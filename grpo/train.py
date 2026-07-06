import time
from concurrent.futures import ThreadPoolExecutor

import torch
import torch.optim as optim

from shared import data, persistence
from shared.buyer import make_buyer
from shared.judge import make_judge
from shared.env import NegotiationEnv
from shared.seeding import seed_everything
from .agent import GRPOAgent
from .config import GRPOConfig
from .loss import group_advantages, completion_mask, grpo_loss_batch, group_keep_indices

LOG_FIELDS = ["update", "mean_reward", "deal_rate", "policy_loss", "kl", "entropy",
              "clip_frac", "grad_norm", "buyer_fails", "judge_fails",
              "gen_s", "api_s", "update_s", "wall_time_s"]


def _score_close(scenario, close, buyer, cfg, judge):
    """Returns (reward, agreed_price, transcript, failed); failed completions are unscorable."""
    env = NegotiationEnv([scenario], buyer, cfg, single_turn=True, judge=judge)
    env.reset(seed=cfg.seed)   # tied to cfg.seed so buyer sampling varies across training seeds
    _obs, reward, _t, _tr, info = env.step(close)
    return (reward, info.get("agreed_price"),
            [[role, text] for role, text in env.turns],
            bool(info.get("buyer_failed") or info.get("judge_failed")))


def collect_group(agent, scenario, buyer, cfg, pool, judge, gen_seed=None):
    """Generate G closes for one scenario, score them concurrently, return a group dict."""
    t0 = time.time()
    ctx_ids, gen_ids = agent.generate_group(scenario["system"], scenario["seed"], cfg.group_size,
                                            seed=gen_seed)
    gen_s = time.time() - t0
    mask = completion_mask(gen_ids, agent.tokenizer.eos_token_id)
    closes = agent.decode(gen_ids)

    t1 = time.time()
    scored = list(pool.map(lambda c: _score_close(scenario, c, buyer, cfg, judge), closes))
    api_s = time.time() - t1
    rewards = [s[0] for s in scored]
    deals = [s[1] is not None for s in scored]
    transcripts = [s[2] for s in scored]
    failed = [s[3] for s in scored]

    # drop failed completions; skip the scenario if too few survive for a stable baseline
    keep = group_keep_indices(failed)
    if keep is None:
        return None
    if len(keep) < len(closes):
        idx = torch.tensor(keep, device=gen_ids.device)
        gen_ids, mask = gen_ids.index_select(0, idx), mask.index_select(0, idx)
        rewards = [rewards[i] for i in keep]
        deals = [deals[i] for i in keep]
        transcripts = [transcripts[i] for i in keep]
        closes = [closes[i] for i in keep]

    with torch.no_grad():
        old_logps, _ = agent.evaluate_batch(ctx_ids, gen_ids, mask)
        ref_logps = agent.evaluate_ref_batch(ctx_ids, gen_ids, mask)
    return {
        "ctx_ids": ctx_ids, "gen_ids": gen_ids, "mask": mask,
        "advantages": group_advantages(rewards).to(agent.device),
        "old_logps": old_logps.detach(), "ref_logps": ref_logps.detach(),
        "rewards": rewards, "deals": deals, "gen_s": gen_s, "api_s": api_s,
        "scenario_id": scenario["id"], "closes": closes, "transcripts": transcripts,
    }


def log_group_transcripts(cfg, groups, update):
    """One JSONL record per completion: transcript, reward, and group-relative advantage."""
    for gi, g in enumerate(groups):
        advs = g["advantages"].detach().cpu().tolist()
        for ci in range(len(g["rewards"])):
            persistence.log_transcript(cfg, {
                "update": update + 1,
                "group": gi,
                "completion": ci,
                "scenario_id": g["scenario_id"],
                "reward": round(g["rewards"][ci], 4),
                "advantage": round(advs[ci], 4),
                "deal": g["deals"][ci],
                "close": g["closes"][ci],
                "turns": g["transcripts"][ci],
            })


def grpo_update(agent, optimizer, groups, cfg):
    sums = {"policy": 0.0, "kl": 0.0, "entropy": 0.0, "clip": 0.0}
    grad_norm = 0.0
    n = len(groups)
    for _epoch in range(cfg.n_epochs):
        optimizer.zero_grad()
        for g in groups:
            new_logps, entropy = agent.evaluate_batch(g["ctx_ids"], g["gen_ids"], g["mask"])
            loss_vec, policy, kl, ent, clip = grpo_loss_batch(
                new_logps, g["old_logps"], g["ref_logps"], entropy, g["advantages"], g["mask"],
                clip_eps=cfg.clip_eps, kl_coef=cfg.kl_coef, ent_coef=cfg.ent_coef)
            # 1/G then 1/n; mean not sum keeps step size independent of group_size
            (loss_vec.mean() / n).backward()
            with torch.no_grad():
                # per-completion [G]; mean for scalar diagnostics only
                sums["policy"] += float(policy.mean())
                sums["kl"] += float(kl.mean())
                sums["entropy"] += float(ent.mean())
                sums["clip"] += float(clip.mean())
        grad_norm = torch.nn.utils.clip_grad_norm_(agent.lora_parameters(), cfg.max_grad_norm).item()
        optimizer.step()
    d = n * cfg.n_epochs
    return {"policy_loss": round(sums["policy"] / d, 4), "kl": round(sums["kl"] / d, 4),
            "entropy": round(sums["entropy"] / d, 4), "clip_frac": round(sums["clip"] / d, 4),
            "grad_norm": round(grad_norm, 4)}


def _parse_args():
    import argparse
    ap = argparse.ArgumentParser(description="Single-closing-turn GRPO training.")
    ap.add_argument("--updates", type=int, default=None, help="override num_updates")
    ap.add_argument("--episodes", type=int, default=None, help="override episodes_per_update")
    ap.add_argument("--group-size", type=int, default=None, help="override group_size (G)")
    ap.add_argument("--seed", type=int, default=None, help="override cfg.seed (for multi-seed runs)")
    ap.add_argument("--preview", action="store_true",
                    help="tiny smoke run (default 3 updates x 4 scenarios x group 4), local-only: "
                         "no HF push, no pod stop, isolated *_preview run dir.")
    return ap.parse_args()


def main():
    import random
    args = _parse_args()
    cfg = GRPOConfig()
    if args.preview:
        cfg.num_updates = args.updates or 3
        cfg.episodes_per_update = args.episodes or 4
        cfg.group_size = args.group_size or 4
        persistence.mark_preview(cfg)
    else:
        if args.updates is not None:
            cfg.num_updates = args.updates
        if args.episodes is not None:
            cfg.episodes_per_update = args.episodes
        if args.group_size is not None:
            cfg.group_size = args.group_size
    if args.seed is not None:
        cfg.seed = args.seed
        cfg.seed_in_path = True
        cfg.__post_init__()            # re-derive output_dir + hf_repo_id with the _s{seed} suffix

    seed_everything(cfg.seed)
    persistence.install_signal_handlers()
    persistence.write_run_config(cfg)
    print(f"GRPO {'PREVIEW ' if args.preview else ''}-> {cfg.output_dir}  "
          f"({cfg.num_updates} updates x {cfg.episodes_per_update} scenarios x group {cfg.group_size})",
          flush=True)

    agent = GRPOAgent(cfg)
    # explicit weight_decay=0.0: AdamW defaults to 0.01
    optimizer = optim.AdamW(agent.lora_parameters(), lr=cfg.learning_rate, weight_decay=0.0)
    start = persistence.load_checkpoint(agent, optimizer, cfg) if cfg.resume else 0

    scenarios = data.load_grpo_examples(cfg)
    random.Random(cfg.seed).shuffle(scenarios)
    buyer = make_buyer(cfg, "train")
    judge = make_judge(cfg)
    pool = ThreadPoolExecutor(max_workers=cfg.group_size)

    uploaded_ok = False
    last_step = start          # advances only on real gradient steps; resume retrains skipped range
    empty_streak = 0
    for update in range(start, cfg.num_updates):
        if persistence.stop_requested():
            print("  stop requested; finishing after update %d" % update, flush=True)
            break
        t0 = time.time()
        base = cfg.seed * 1_000_000 + update * cfg.episodes_per_update
        # gen_seed = base + e: reproducible across resume, decorrelated within an update
        groups = [collect_group(agent, scenarios[(base + e) % len(scenarios)], buyer, cfg, pool,
                                judge, gen_seed=base + e)
                  for e in range(cfg.episodes_per_update)]
        groups = [g for g in groups if g is not None]   # drop scenarios skipped on buyer-API outage
        if not groups:
            empty_streak += 1
            print(f"update {update + 1}: all groups skipped (buyer/judge API; {empty_streak} empty "
                  "in a row), skipping update", flush=True)
            if persistence.outage_abort(empty_streak, cfg.abort_after_empty_updates):
                persistence.save_checkpoint(agent, optimizer, cfg, last_step)
                print(f"  {empty_streak} consecutive empty updates — sustained buyer/judge outage. "
                      f"Aborting (resume retrains from update {last_step}).", flush=True)
                raise SystemExit(7)
            continue
        empty_streak = 0

        tu = time.time()
        metrics = grpo_update(agent, optimizer, groups, cfg)
        last_step = update + 1
        update_s = round(time.time() - tu, 1)

        rewards = [r for g in groups for r in g["rewards"]]
        deals = [d for g in groups for d in g["deals"]]
        metrics.update({
            "update": update + 1,
            "mean_reward": round(sum(rewards) / max(len(rewards), 1), 4),
            "deal_rate": round(sum(deals) / max(len(deals), 1), 4),
            "gen_s": round(sum(g["gen_s"] for g in groups), 1),
            "api_s": round(sum(g["api_s"] for g in groups), 1),
            "update_s": update_s, "wall_time_s": round(time.time() - t0, 1),
            "buyer_fails": buyer.failures,            # cumulative; diff rows for per-update
            "judge_fails": getattr(judge, "api_failures", 0),
        })
        persistence.log_row(cfg, LOG_FIELDS, metrics)
        log_group_transcripts(cfg, groups, update)
        print(f"update {update + 1}/{cfg.num_updates}  reward={metrics['mean_reward']}  "
              f"deals={metrics['deal_rate']}  kl={metrics['kl']}  clip={metrics['clip_frac']}  "
              f"ent={metrics['entropy']}  [gen={metrics['gen_s']}s api={metrics['api_s']}s "
              f"upd={update_s}s tot={metrics['wall_time_s']}s]", flush=True)

        if (update + 1) % cfg.checkpoint_every == 0:
            persistence.save_checkpoint(agent, optimizer, cfg, update + 1)
            uploaded_ok = persistence.push_to_hub(cfg, update + 1)

    persistence.save_checkpoint(agent, optimizer, cfg, last_step)
    agent.save_adapter()                       # uniform lora_final for the eval harness
    uploaded_ok = persistence.push_to_hub(cfg, last_step)
    persistence.maybe_stop_pod(cfg, uploaded_ok)
    if cfg.hf_repo_id and not uploaded_ok:
        # only this push mirrors lora_final; non-zero exit makes the orchestrator retry it
        print("  final HF push failed; exiting non-zero so the orchestrator retries the push.",
              flush=True)
        raise SystemExit(8)
    if persistence.stop_requested():
        # interrupted: checkpointed for resume, but the unit is not complete
        raise SystemExit(130)


if __name__ == "__main__":
    main()
