"""Shared eval path for base/SFT/PPO/GRPO."""

from .env import NegotiationEnv


def run_eval(cfg, generate_fn, scenarios, buyer, judge=None, max_consecutive_failures=10):
    """Returns (metrics, per_scenario), per_scenario mapping id -> {reward, agreed_price, ratio,
    deal, turns}. max_consecutive_failures stops a sustained buyer/judge outage; 0 disables."""
    per = {}
    consecutive_failed = 0
    for i, sc in enumerate(scenarios):
        env = NegotiationEnv([sc], buyer, cfg, single_turn=False, judge=judge)
        env.reset(seed=i)
        reward, info = 0.0, {}
        for _ in range(cfg.num_turns):
            seller_text = generate_fn(env.get_seller_prompt(), env.obs())
            _obs, reward, terminated, truncated, info = env.step(seller_text)
            if terminated or truncated:
                break
        if info.get("buyer_failed") or info.get("judge_failed"):
            # omit the scenario rather than score a fabricated no-deal
            consecutive_failed += 1
            if max_consecutive_failures and consecutive_failed >= max_consecutive_failures:
                print(f"  eval circuit breaker: {consecutive_failed} consecutive buyer/judge "
                      f"failures — sustained outage; stopping after {i + 1}/{len(scenarios)} "
                      "scenarios.", flush=True)
                break
            continue
        consecutive_failed = 0
        agreed = info.get("agreed_price")
        listing = info.get("listing_price", sc["listing"])
        per[sc["id"]] = {
            "reward": reward,
            "agreed_price": agreed,
            "ratio": (agreed / listing) if agreed else None,
            "deal": agreed is not None,
            "turns": [list(t) for t in env.turns],
        }
    return _aggregate(per), per


def _aggregate(per):
    rewards = [v["reward"] for v in per.values()]
    deals = [v for v in per.values() if v["deal"]]
    ratios = [v["ratio"] for v in deals if v["ratio"] is not None]
    n = max(len(per), 1)
    return {
        "n": len(per),
        "mean_reward": round(sum(rewards) / n, 4),
        "deal_rate": round(len(deals) / n, 4),
        "mean_ratio": round(sum(ratios) / len(ratios), 4) if ratios else None,
    }


def paired_win_rate(per_a, per_b):
    """Fraction of shared scenarios where A's reward exceeds B's (ties = 0.5)."""
    ids = set(per_a) & set(per_b)
    if not ids:
        return None
    wins = sum((per_a[i]["reward"] > per_b[i]["reward"]) +
               0.5 * (per_a[i]["reward"] == per_b[i]["reward"]) for i in ids)
    return round(wins / len(ids), 4)


def _sign_test_p(wins, losses):
    """Two-sided exact binomial sign test (p=0.5) over decisive pairs. 1.0 when none are decisive."""
    from math import comb
    n = wins + losses
    if n == 0:
        return 1.0
    k = min(wins, losses)
    tail = sum(comb(n, j) for j in range(k + 1)) / (2 ** n)
    return round(min(1.0, 2 * tail), 4)


def _norm_sf(x):
    """Upper-tail standard-normal survival function, via erf (no scipy)."""
    import math
    return 0.5 * math.erfc(x / math.sqrt(2.0))


def wilcoxon_signed_rank_p(diffs):
    """Two-sided Wilcoxon signed-rank p over paired differences, normal approximation with tie +
    continuity correction (zeros dropped). 1.0 when no pair is nonzero."""
    import math
    from collections import Counter
    nz = [d for d in diffs if d != 0]
    n = len(nz)
    if n == 0:
        return 1.0
    order = sorted(range(n), key=lambda i: abs(nz[i]))
    ranks = [0.0] * n                                   # average ranks for ties in |d|
    i = 0
    while i < n:
        j = i
        while j + 1 < n and abs(nz[order[j + 1]]) == abs(nz[order[i]]):
            j += 1
        avg = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    w_plus = sum(ranks[i] for i in range(n) if nz[i] > 0)
    mean_w = n * (n + 1) / 4.0
    tie_term = sum(t ** 3 - t for t in Counter(abs(d) for d in nz).values())
    var_w = n * (n + 1) * (2 * n + 1) / 24.0 - tie_term / 48.0
    if var_w <= 0:
        return 1.0
    z = (w_plus - mean_w)
    z = (z - math.copysign(0.5, z)) / math.sqrt(var_w)  # continuity correction
    return round(min(1.0, 2.0 * _norm_sf(abs(z))), 6)


def paired_stats(per_a, per_b):
    """Paired A-vs-B over shared scenarios: win-rate, wins/losses/ties, sign-test p, Wilcoxon p."""
    ids = set(per_a) & set(per_b)
    if not ids:
        return None
    wins = losses = ties = 0
    diffs = []
    for i in ids:
        da, db = per_a[i]["reward"], per_b[i]["reward"]
        diffs.append(da - db)
        if da > db:
            wins += 1
        elif da < db:
            losses += 1
        else:
            ties += 1
    return {"n": len(ids), "wins": wins, "losses": losses, "ties": ties,
            "win_rate": round((wins + 0.5 * ties) / len(ids), 4),
            "p_value": _sign_test_p(wins, losses),
            "wilcoxon_p": wilcoxon_signed_rank_p(diffs)}


def compare_methods(results, common_exclude=("base",)):
    """results: {method -> per_scenario} -> per-method aggregates, pairwise win-rates, win_stats.
    metrics_common re-aggregates over scenarios shared by the trained methods; common_exclude
    methods don't shrink that common set."""
    methods = list(results)
    core = [m for m in methods if m not in common_exclude] or methods
    common = set.intersection(*[set(results[m]) for m in core]) if core else set()
    metrics_common = {}
    for m in methods:
        ids = common if m in core else common & set(results[m])
        metrics_common[m] = _aggregate({i: results[m][i] for i in ids})
    return {
        "methods": methods,
        "metrics": {m: _aggregate(results[m]) for m in methods},
        "common_n": len(common),
        "metrics_common": metrics_common,
        "win_rate": {f"{a}_vs_{b}": paired_win_rate(results[a], results[b])
                     for a in methods for b in methods if a != b},
        "win_stats": {f"{a}_vs_{b}": paired_stats(results[a], results[b])
                      for i, a in enumerate(methods) for b in methods[i + 1:]},
    }


def make_greedy_generate_fn(model, tokenizer, cfg):
    """Wrap a loaded policy as generate_fn(system, obs) -> seller_text, greedy."""
    import torch
    from . import model as model_mod

    try:
        from unsloth import FastModel
        FastModel.for_inference(model)
    except Exception:
        pass
    model.eval()      # the transformers-fallback tier never hits FastModel.for_inference

    def generate_fn(system, obs):
        prompt = model_mod.build_prompt(tokenizer, system, obs)
        inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(model.device)
        with torch.no_grad():
            # a generation_config repetition penalty applies even under greedy decoding; neutralize it
            out = model.generate(**inputs, max_new_tokens=cfg.max_new_tokens, do_sample=False,
                                 repetition_penalty=1.0, pad_token_id=tokenizer.pad_token_id)
        text = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        return text.strip()

    return generate_fn


def evaluate_adapter(cfg, scenarios, buyer, adapter_dir=None, judge=None):
    """Load base (+ LoRA adapter if given; None = baseline) and run the eval. Needs GPU."""
    from . import model as model_mod
    from .judge import make_judge

    model, tokenizer = model_mod.load_base(cfg)
    if adapter_dir:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter_dir)
    generate_fn = make_greedy_generate_fn(model, tokenizer, cfg)
    return run_eval(cfg, generate_fn, scenarios, buyer, judge=judge or make_judge(cfg))
