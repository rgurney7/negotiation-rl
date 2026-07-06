"""Held-out eval of base + each method's adapter on identical scenarios; writes per-method JSONs + comparison.json."""

import argparse
import hashlib
import json
import os

from shared.config import SharedConfig
from shared import data, persistence
from shared.eval_harness import evaluate_adapter, compare_methods
from shared.buyer import make_buyer
from shared.judge import make_judge


# fixed sample seed, not the run seed: every method and training seed must score the same set
EVAL_SAMPLE_SEED = 20260701


def sample_distinct_listings(scenarios, limit):
    """Seeded sample of `limit` scenarios, one per distinct listing (repeated listings are correlated)."""
    import random
    if not limit or limit >= len(scenarios):
        return scenarios
    first_per_listing = {}
    for s in scenarios:
        first_per_listing.setdefault((s.get("title"), s.get("listing"), s.get("description")), s)
    firsts = list(first_per_listing.values())
    if len(firsts) >= limit:
        chosen = {id(s) for s in random.Random(EVAL_SAMPLE_SEED).sample(firsts, limit)}
    else:
        print(f"  WARN: only {len(firsts)} distinct listings < limit {limit}; taking one row per "
              "listing plus repeats to fill (pseudoreplication returns).", flush=True)
        chosen = {id(s) for s in firsts}
        for s in scenarios:                       # top up with repeats in file order
            if len(chosen) >= limit:
                break
            chosen.add(id(s))
    return [s for s in scenarios if id(s) in chosen]


def _scenario_sig(scenarios):
    """Order-independent fingerprint of the scenario set; checked before reusing a cached eval."""
    ids = sorted(str(s["id"]) if isinstance(s, dict) else str(s) for s in scenarios)
    return hashlib.sha1("|".join(ids).encode()).hexdigest()[:16]


def drop_cold_opens(scenarios):
    """Drop scenarios with no buyer opener; the policies never train on that state."""
    from shared import render
    kept = [s for s in scenarios
            if any(r == "buyer" for r, _ in render.parse_seed(s.get("seed") or ""))]
    if len(kept) < len(scenarios):
        print(f"  excluded {len(scenarios) - len(kept)} cold-open scenario(s) (no buyer opener) "
              f"from the {len(scenarios)}-scenario pool", flush=True)
    return kept


def adapter_dirs(methods, seed=None):
    """Resolve each method's lora_final dir from its config (seeded _s{seed} dir when seed is set)."""
    from sft.config import SFTConfig
    from grpo.config import GRPOConfig
    from ppo.config import PPOConfig
    kw = {"seed": seed, "seed_in_path": True} if seed is not None else {}
    cfgs = {"sft": SFTConfig(**kw), "grpo": GRPOConfig(**kw), "ppo": PPOConfig(**kw)}
    return {m: os.path.join(cfgs[m].output_dir, "lora_final") for m in methods}


def _free_gpu():
    import gc
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def run(cfg, dirs, scenarios, buyer, out_dir, judge=None, include_base=True, evaluate=evaluate_adapter,
        min_coverage=0.9, split=None, limit=None):
    """Evaluate base + each adapter on the same scenarios; below min_coverage a method exits non-zero without persisting."""
    results = {}
    n_req = len(scenarios)
    scenario_sig = _scenario_sig(scenarios)
    targets = ([("base", None)] if include_base else []) + list(dirs.items())
    for name, adir in targets:
        out_path = os.path.join(out_dir, f"{name}_eval.json")
        if os.path.exists(out_path):
            # reuse a cached eval only for the same request shape AND the same scenario set
            saved = json.load(open(out_path))
            if (saved.get("n_requested") == n_req and saved.get("split") == split
                    and saved.get("limit") == limit and saved.get("scenario_sig") == scenario_sig):
                results[name] = saved.get("per_scenario", {})
                print(f"  {name}: reusing {out_path} (resume)", flush=True)
                continue
            reason = ("scenario set" if saved.get("scenario_sig") not in (None, scenario_sig)
                      else "request shape")
            print(f"  {name}: cached eval is a different {reason} "
                  f"(saved split={saved.get('split')} limit={saved.get('limit')} "
                  f"n={saved.get('n_requested')} sig={saved.get('scenario_sig')} vs "
                  f"{split}/{limit}/{n_req}/{scenario_sig}) — re-scoring.", flush=True)
        metrics, per = evaluate(cfg, scenarios, buyer, adapter_dir=adir, judge=judge)
        if n_req and len(per) < min_coverage * n_req:
            print(f"  {name}: only {len(per)}/{n_req} scenarios scored (< {min_coverage:.0%} floor) — "
                  "that's an API outage, not a result. NOT persisting; exiting non-zero so the "
                  "orchestrator retries this unit (already-scored methods resume from disk).",
                  flush=True)
            raise SystemExit(6)
        results[name] = per
        persistence.write_json_atomic(
            out_path, {"adapter_dir": adir, "metrics": metrics, "per_scenario": per,
                       "n_requested": n_req, "split": split, "limit": limit,
                       "scenario_sig": scenario_sig})
        print(f"  {name}: {metrics}", flush=True)
        _free_gpu()
    comparison = compare_methods(results)
    persistence.write_json_atomic(os.path.join(out_dir, "comparison.json"), comparison)
    return comparison


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--methods", nargs="+", default=["sft", "grpo", "ppo"],
                    choices=["sft", "grpo", "ppo"])
    ap.add_argument("--split", default="test", choices=["test", "validation"])
    ap.add_argument("--no-base", action="store_true", help="skip the un-adapted baseline")
    ap.add_argument("--out", default=None, help="output dir (default <runs>/eval[_s{seed}])")
    ap.add_argument("--seed", type=int, default=None,
                    help="evaluate this seed's adapters; writes to eval_s{seed} so seeds don't "
                         "clobber. base is seed-independent (greedy gen + temp-0 grader): eval it "
                         "once and pass --no-base for later seeds.")
    ap.add_argument("--limit", type=int, default=None,
                    help="seeded sample of N DISTINCT listings, one scenario each (fixed sample "
                         "seed: identical set for every method and every training seed)")
    ap.add_argument("--min-coverage", type=float, default=0.9,
                    help="abort non-zero if a method scores fewer than this fraction of the "
                         "requested scenarios (outage guard; 0 disables)")
    args = ap.parse_args()

    # max_seq_length 4096 so long multi-turn eval contexts never truncate, same for every method
    cfg = SharedConfig(run_name="eval", max_seq_length=4096,
                       seed=args.seed if args.seed is not None else 42,
                       seed_in_path=args.seed is not None)
    out_dir = args.out or cfg.output_dir
    scenarios = drop_cold_opens(data.load_eval_pool(cfg, split=args.split))
    if args.limit:
        scenarios = sample_distinct_listings(scenarios, args.limit)
    print(f"Eval on {len(scenarios)} {args.split} scenarios -> {out_dir}", flush=True)

    dirs = adapter_dirs(args.methods, seed=args.seed)
    for m, d in dirs.items():
        if not os.path.isdir(d):
            print(f"  WARN: {m} adapter not found at {d}", flush=True)

    buyer = make_buyer(cfg, "grade")
    judge = make_judge(cfg)
    comparison = run(cfg, dirs, scenarios, buyer, out_dir, judge=judge, include_base=not args.no_base,
                     min_coverage=args.min_coverage, split=args.split, limit=args.limit)

    print("\n=== COMPARISON ===", flush=True)
    # headline is over the scenarios shared by all methods; own-set metrics are in {method}_eval.json
    print(f"  (descriptive metrics over the {comparison.get('common_n')} scenarios shared by all "
          f"methods)", flush=True)
    for m, agg in comparison["metrics_common"].items():
        print(f"  {m:5s}  deal_rate={agg['deal_rate']}  mean_reward={agg['mean_reward']}  "
              f"mean_ratio={agg['mean_ratio']}", flush=True)
    for pair, st in comparison["win_stats"].items():
        if st:
            decisive = st["wins"] + st["losses"]
            print(f"  {pair}: win_rate={st['win_rate']}  "
                  f"(wins {st['wins']}/{decisive} decisive, ties {st['ties']}, "
                  f"sign_p={st['p_value']}, wilcoxon_p={st.get('wilcoxon_p')})", flush=True)


if __name__ == "__main__":
    main()
