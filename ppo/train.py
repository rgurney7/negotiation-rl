import argparse
import os
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch
import torch.optim as optim

from shared import data, persistence, model as model_mod
from shared.buyer import make_buyer
from shared.judge import make_judge
from shared.env import make_envs
from shared.eval_harness import run_eval, make_greedy_generate_fn
from shared.seeding import seed_everything
from .config import PPOConfig
from .gae import compute_gae
from .rollout import truncate_at_close, drop_failed_episodes, episode_failed
from .agent import PPOAgent

LOG_FIELDS = ["update", "deals", "no_deals", "mean_reward", "mean_price_ratio",
              "policy_loss", "value_loss", "entropy", "approx_kl", "clip_frac",
              "grad_norm", "critic_grad_norm", "explained_variance", "mean_resp_len",
              "buyer_fails", "judge_fails", "gen_s", "api_s", "update_s", "wall_time_s"]


def _mean(xs):
    return sum(xs) / len(xs) if xs else 0.0


def _env_reset(env_and_seed):
    env, seed = env_and_seed
    return env.reset(seed=seed)


def _env_step(env_and_text):
    env, text = env_and_text
    return env.step(text)


def collect_rollouts(agent, envs, cfg, sampling_params, pool, update):
    """Run B episodes in lockstep. Returns (episodes, final_infos, timings)."""
    B = len(envs)
    t0 = time.time()
    base_seed = cfg.seed * 1_000_000 + update * B
    obs = list(pool.map(_env_reset, [(envs[i], base_seed + i) for i in range(B)]))
    api_s = time.time() - t0
    gen_s = 0.0
    seller_prompts = [e.get_seller_prompt() for e in envs]

    episodes = [[] for _ in range(B)]
    final_infos = [{} for _ in range(B)]
    active = [True] * B

    for _t in range(cfg.num_turns):
        idx = [i for i in range(B) if active[i]]
        if not idx:
            break
        tg = time.time()
        prompts = [agent.build_prompt(seller_prompts[i], obs[i]) for i in idx]
        # offset the sampling seed by turn index so turns don't restart from the same RNG state
        turn_params = dict(sampling_params)
        base_sp = turn_params.get("seed")
        if base_sp is not None:
            turn_params["seed"] = base_sp + _t
        gens = agent.generate_batch(prompts, turn_params)
        with torch.no_grad():
            evals = [agent.evaluate(g.prompt_ids, g.gen_ids) for g in gens]
        gen_s += time.time() - tg

        ta = time.time()
        step_args = [(envs[i], gens[k].text) for k, i in enumerate(idx)]
        step_results = list(pool.map(_env_step, step_args))
        api_s += time.time() - ta

        for k, i in enumerate(idx):
            log_probs, _entropy, value = evals[k]
            next_obs, reward, terminated, truncated, info = step_results[k]
            episodes[i].append({
                "prompt_ids": gens[k].prompt_ids, "gen_ids": gens[k].gen_ids,
                "old_log_probs": log_probs.detach(), "value": float(value),
                "reward": float(reward), "resp_len": int(gens[k].gen_ids.shape[1]),
            })
            obs[i] = next_obs
            if terminated or truncated:
                active[i] = False
                final_infos[i] = info or {}

    return episodes, final_infos, {"gen_s": round(gen_s, 1), "api_s": round(api_s, 1)}


def compute_advantages(episodes, cfg):
    """Per-episode GAE, then flatten and batch-normalize advantages."""
    samples = []
    for steps in episodes:
        if not steps:
            continue
        rewards = torch.tensor([s["reward"] for s in steps], dtype=torch.float32)
        values = torch.tensor([s["value"] for s in steps], dtype=torch.float32)
        adv, ret = compute_gae(rewards, values, cfg.gamma, cfg.gae_lambda)
        for j, s in enumerate(steps):
            samples.append({**s, "advantage": float(adv[j]), "return": float(ret[j])})

    if cfg.normalize_advantages and len(samples) > 1:
        advs = torch.tensor([s["advantage"] for s in samples])
        mean, std = advs.mean().item(), advs.std().item() + 1e-8
        for s in samples:
            s["advantage"] = (s["advantage"] - mean) / std
    return samples


def _explained_variance(values, returns):
    values, returns = np.asarray(values), np.asarray(returns)
    var = returns.var()
    return float("nan") if var == 0 else float(1.0 - (returns - values).var() / var)


def ppo_update(agent, optimizer, samples, cfg):
    """ppo_epochs passes; gradients accumulate across all samples in a pass, one step per pass."""
    n = len(samples)
    sums = {"policy": 0.0, "value": 0.0, "entropy": 0.0, "kl": 0.0, "clip": 0.0}
    grad_norm = critic_grad_norm = 0.0
    last_values, last_returns = [], []

    for _epoch in range(cfg.ppo_epochs):
        optimizer.zero_grad()
        last_values, last_returns = [], []
        for s in samples:
            new_lp, entropy, value = agent.evaluate(s["prompt_ids"], s["gen_ids"])
            old_lp = s["old_log_probs"]
            adv, ret = s["advantage"], s["return"]

            ratio = torch.exp(new_lp - old_lp)                       # token-level
            p1 = ratio * adv
            p2 = torch.clamp(ratio, 1.0 - cfg.clip_eps, 1.0 + cfg.clip_eps) * adv
            policy_loss = -torch.min(p1, p2).mean()
            value_loss = (ret - value) ** 2
            entropy_bonus = entropy.mean()

            loss = policy_loss + cfg.value_coef * value_loss - cfg.entropy_coef * entropy_bonus
            (loss / n).backward()

            with torch.no_grad():
                sums["policy"] += policy_loss.item()
                sums["value"] += value_loss.item()
                sums["entropy"] += entropy_bonus.item()
                sums["kl"] += (old_lp - new_lp).mean().item()
                sums["clip"] += ((ratio - 1.0).abs() > cfg.clip_eps).float().mean().item()
                last_values.append(float(value))
                last_returns.append(ret)

        # clip separately: a combined norm would let a large critic grad throttle the policy
        grad_norm = torch.nn.utils.clip_grad_norm_(agent.lora_parameters(), cfg.max_grad_norm).item()
        critic_grad_norm = torch.nn.utils.clip_grad_norm_(
            agent.critic_parameters(), cfg.max_grad_norm).item()
        optimizer.step()

    d = n * cfg.ppo_epochs
    return {"policy_loss": round(sums["policy"] / d, 4), "value_loss": round(sums["value"] / d, 4),
            "entropy": round(sums["entropy"] / d, 4), "approx_kl": round(sums["kl"] / d, 4),
            "clip_frac": round(sums["clip"] / d, 4), "grad_norm": round(grad_norm, 4),
            "critic_grad_norm": round(critic_grad_norm, 4),
            "explained_variance": round(_explained_variance(last_values, last_returns), 4)}


def log_episode_transcripts(cfg, envs, episodes, infos, update):
    """One JSONL record per episode; must run before the next reset overwrites env.turns."""
    for i, steps in enumerate(episodes):
        info = infos[i] or {}
        reward = sum(s["reward"] for s in steps) if steps else 0.0
        persistence.log_transcript(cfg, {
            "update": update + 1,
            "episode": i,
            "scenario_id": info.get("scenario"),
            "listing_price": info.get("listing_price"),
            "agreed_price": info.get("agreed_price"),
            "deal": info.get("agreed_price") is not None,
            "close_turn": info.get("close_turn"),         # judge's deal turn
            "buyer_failed": bool(info.get("buyer_failed")),  # episode dropped: buyer API outage
            "judge_failed": bool(info.get("judge_failed")),  # episode dropped: both judges down
            "trained_turns": len(steps),                  # steps kept after post-deal truncation
            "reward": round(reward, 4),
            "resp_lens": [s["resp_len"] for s in steps],
            "values": [round(s["value"], 4) for s in steps],
            "turns": [[role, text] for role, text in envs[i].turns],
        })


def rollout_stats(episodes, infos):
    # exclude API-failed episodes so an outage doesn't read as a wave of no-deals
    live = [info for info in infos if not episode_failed(info)]
    ep_rewards = [sum(s["reward"] for s in steps) for steps in episodes if steps]
    deals = [info for info in live if info.get("agreed_price") is not None]
    ratios = [info["agreed_price"] / info["listing_price"] for info in deals if info.get("listing_price")]
    resp_lens = [s["resp_len"] for steps in episodes for s in steps]
    return {"deals": len(deals), "no_deals": len(live) - len(deals),
            "mean_reward": round(_mean(ep_rewards), 4), "mean_price_ratio": round(_mean(ratios), 4),
            "mean_resp_len": round(_mean(resp_lens), 1)}


def checkpoint_eval(agent, cfg, scenarios, buyer, judge):
    """Greedy eval on a fixed val subset (disjoint from test); restores training mode after."""
    gen = make_greedy_generate_fn(agent.model, agent.tokenizer, cfg)
    metrics, _ = run_eval(cfg, gen, scenarios, buyer, judge=judge)
    model_mod.for_training(agent.model)          # eval toggled the model to inference; undo it
    return metrics


def _parse_args():
    ap = argparse.ArgumentParser(description="Multi-turn PPO training.")
    ap.add_argument("--updates", type=int, default=None, help="override num_updates")
    ap.add_argument("--episodes", type=int, default=None, help="override num_parallel_episodes (B)")
    ap.add_argument("--seed", type=int, default=None, help="override cfg.seed (for multi-seed runs)")
    ap.add_argument("--no-ckpt-eval", action="store_true",
                    help="disable the D4 checkpoint-eval trajectory (saves grade-API budget/time)")
    ap.add_argument("--preview", action="store_true",
                    help="tiny smoke run (default 3 updates x 4 episodes), local-only: no HF push, "
                         "no pod stop, isolated *_preview run dir. Combine with --updates/--episodes.")
    return ap.parse_args()


def main():
    args = _parse_args()
    cfg = PPOConfig()
    if args.preview:
        cfg.num_updates = args.updates or 3
        cfg.num_parallel_episodes = args.episodes or 4
        cfg.checkpoint_eval_every = 0
        persistence.mark_preview(cfg)
    else:
        if args.updates is not None:
            cfg.num_updates = args.updates
        if args.episodes is not None:
            cfg.num_parallel_episodes = args.episodes
    if args.no_ckpt_eval:
        cfg.checkpoint_eval_every = 0
    if args.seed is not None:
        cfg.seed = args.seed
        cfg.seed_in_path = True
        cfg.__post_init__()            # re-derive output_dir + hf_repo_id with the _s{seed} suffix

    seed_everything(cfg.seed)
    persistence.install_signal_handlers()
    persistence.write_run_config(cfg)
    print(f"PPO {'PREVIEW ' if args.preview else ''}-> {cfg.output_dir}  "
          f"({cfg.num_updates} updates x {cfg.num_parallel_episodes} episodes)", flush=True)

    agent = PPOAgent(cfg)
    # explicit weight_decay=0.0: AdamW defaults to 0.01
    optimizer = optim.AdamW([
        {"params": agent.lora_parameters(), "lr": cfg.policy_lr},
        {"params": agent.critic_parameters(), "lr": cfg.critic_lr},
    ], weight_decay=0.0)
    start = persistence.load_checkpoint(agent, optimizer, cfg) if cfg.resume else 0

    buyer = make_buyer(cfg, "train")
    judge = make_judge(cfg)
    envs = make_envs(cfg, data.load_ppo_scenarios(cfg), buyer,
                     cfg.num_parallel_episodes, single_turn=False, judge=judge)
    pool = ThreadPoolExecutor(max_workers=cfg.num_parallel_episodes)

    # missing grade key / val slice disables the diagnostic instead of aborting the run
    ckpt_eval_scen = ckpt_eval_buyer = None
    if cfg.checkpoint_eval_every and cfg.checkpoint_eval_n:
        try:
            ckpt_eval_scen = data.load_val50(cfg)[:cfg.checkpoint_eval_n]
            ckpt_eval_buyer = make_buyer(cfg, "grade")
            print(f"  D4 checkpoint-eval: {len(ckpt_eval_scen)} val scenarios every "
                  f"{cfg.checkpoint_eval_every} updates -> val_trajectory.jsonl", flush=True)
        except Exception as e:
            ckpt_eval_scen = ckpt_eval_buyer = None
            print(f"  D4 checkpoint-eval disabled ({e}); training continues.", flush=True)

    uploaded_ok = False
    last_step = start          # advances only on real gradient steps; resume retrains skipped range
    empty_streak = 0
    for update in range(start, cfg.num_updates):
        if persistence.stop_requested():
            print("  stop requested; finishing after update %d" % update, flush=True)
            break
        t0 = time.time()
        # stride (num_turns+1) per update so per-turn offsets never collide across updates
        sampling_params = agent.make_sampling_params(seed=cfg.seed * 100_000 + update * (cfg.num_turns + 1))
        episodes, infos, timings = collect_rollouts(agent, envs, cfg, sampling_params, pool, update)
        episodes = drop_failed_episodes(episodes, infos)     # drop buyer-API-failed episodes
        episodes = truncate_at_close(episodes, infos, cfg)   # drop post-deal turns before GAE
        samples = compute_advantages(episodes, cfg)
        if not samples:
            empty_streak += 1
            print(f"update {update + 1}: no samples collected ({empty_streak} empty in a row), "
                  "skipping", flush=True)
            if persistence.outage_abort(empty_streak, cfg.abort_after_empty_updates):
                persistence.save_checkpoint(agent, optimizer, cfg, last_step)
                print(f"  {empty_streak} consecutive empty updates — sustained buyer/judge outage. "
                      f"Aborting (resume retrains from update {last_step}).", flush=True)
                raise SystemExit(7)
            continue
        empty_streak = 0

        tu = time.time()
        metrics = ppo_update(agent, optimizer, samples, cfg)
        last_step = update + 1
        update_s = round(time.time() - tu, 1)

        metrics.update(rollout_stats(episodes, infos))
        metrics.update(timings)
        metrics["update_s"] = update_s
        metrics["update"] = update + 1
        metrics["wall_time_s"] = round(time.time() - t0, 1)
        metrics["buyer_fails"] = buyer.failures            # cumulative; diff rows for per-update
        metrics["judge_fails"] = getattr(judge, "api_failures", 0)
        persistence.log_row(cfg, LOG_FIELDS, metrics)
        log_episode_transcripts(cfg, envs, episodes, infos, update)
        print(f"update {update + 1}/{cfg.num_updates}  reward={metrics['mean_reward']}  "
              f"deals={metrics['deals']}/{cfg.num_parallel_episodes}  "
              f"price_ratio={metrics['mean_price_ratio']}  kl={metrics['approx_kl']}  "
              f"ent={metrics['entropy']}  ev={metrics['explained_variance']}  "
              f"[gen={metrics['gen_s']}s api={metrics['api_s']}s upd={update_s}s "
              f"tot={metrics['wall_time_s']}s]", flush=True)

        if ckpt_eval_scen and (update + 1) % cfg.checkpoint_eval_every == 0:
            # diagnostic only: a failure here must not kill training
            try:
                tv = time.time()
                vm = checkpoint_eval(agent, cfg, ckpt_eval_scen, ckpt_eval_buyer, judge)
                vm.update({"update": update + 1, "eval_s": round(time.time() - tv, 1)})
                persistence.append_jsonl(os.path.join(cfg.output_dir, "val_trajectory.jsonl"), vm)
                print(f"  [val@{update + 1}] n={vm['n']} deal_rate={vm['deal_rate']} "
                      f"mean_reward={vm['mean_reward']} ({vm['eval_s']}s)", flush=True)
            except Exception as e:
                ckpt_eval_scen = ckpt_eval_buyer = None
                model_mod.for_training(agent.model)      # eval may have died mid-toggle
                print(f"  D4 checkpoint-eval failed ({type(e).__name__}: {e}); disabling it, "
                      "training continues.", flush=True)

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
