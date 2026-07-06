"""Aggregate the per-seed eval JSONs into the across-seed comparison + primary contrast."""

import argparse
import json
import os

from shared.config import ROOT
from shared.eval_harness import paired_stats, _aggregate

CI_MIN_SEEDS = 5   # below this, report a descriptive range instead of a bootstrap CI


def _mean(xs):
    return sum(xs) / len(xs) if xs else None


def iqm(xs):
    """Interquartile mean: trims floor(0.25 N) from each tail."""
    xs = sorted(float(x) for x in xs)
    n = len(xs)
    if n == 0:
        return None
    lo = int(0.25 * n)
    core = xs[lo:n - lo] or xs
    return sum(core) / len(core)


def bootstrap_ci(xs, pct=95, n_boot=10000, seed=0):
    """Deterministic percentile bootstrap CI; callers gate on N >= CI_MIN_SEEDS."""
    import random
    xs = [float(x) for x in xs]
    n = len(xs)
    if n == 0:
        return (None, None)
    rng = random.Random(seed)
    means = sorted(_mean([xs[rng.randrange(n)] for _ in range(n)]) for _ in range(n_boot))
    a = (100 - pct) / 2.0
    lo = means[int(a / 100 * n_boot)]
    hi = means[min(n_boot - 1, int((100 - a) / 100 * n_boot))]
    return (lo, hi)


def describe(values):
    """Across-seed summary of one per-seed scalar metric."""
    xs = [float(v) for v in values if v is not None]
    n = len(xs)
    out = {"n": n, "values": [round(x, 4) for x in xs], "mean": round(_mean(xs), 4) if n else None}
    if n >= 2:
        m = _mean(xs)
        out["std"] = round((sum((x - m) ** 2 for x in xs) / (n - 1)) ** 0.5, 4)
        out["min"], out["max"] = round(min(xs), 4), round(max(xs), 4)
    if n >= CI_MIN_SEEDS:
        lo, hi = bootstrap_ci(xs)
        out["ci95"] = [round(lo, 4), round(hi, 4)]
        out["iqm"] = round(iqm(xs), 4)
    elif n >= 1:
        out["interval"] = f"descriptive range only (N={n} < {CI_MIN_SEEDS}: too few seeds for a bootstrap CI/IQM)"
    return out


def paired_contrast_across_seeds(per_seed_diffs):
    """Direction + sign-consistency of per-seed diffs (A - B); CI only at N >= CI_MIN_SEEDS."""
    diffs = [float(d) for d in per_seed_diffs if d is not None]
    n = len(diffs)
    pos = sum(1 for d in diffs if d > 0)
    neg = sum(1 for d in diffs if d < 0)
    out = {"n": n, "diffs": [round(d, 4) for d in diffs],
           "mean_diff": round(_mean(diffs), 4) if n else None,
           "all_same_sign": n > 0 and (pos == n or neg == n),
           "direction": "A>B" if pos > neg else ("B>A" if neg > pos else "mixed/tied")}
    if n >= CI_MIN_SEEDS:
        lo, hi = bootstrap_ci(diffs)
        out["ci95"] = [round(lo, 4), round(hi, 4)]
    return out


# --- file IO + CLI ---

def _eval_dir(eval_root, seed):
    return os.path.join(eval_root, f"eval_s{seed}")


def load_seed_eval(eval_root, seed, method):
    path = os.path.join(_eval_dir(eval_root, seed), f"{method}_eval.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def _per_seed_metrics(eval_root, seed, methods):
    """({method: metrics}, common_n): trained methods re-aggregated on their shared scenario set; base restricted to its coverage of it."""
    trained = list(methods)
    loaded = {m: load_seed_eval(eval_root, seed, m) for m in trained + ["base"]}
    loaded = {m: r for m, r in loaded.items() if r is not None}
    if not loaded:
        return {}, None
    trained_ps = {m: loaded[m]["per_scenario"] for m in trained
                  if m in loaded and loaded[m].get("per_scenario")}
    out = {m: r["metrics"] for m, r in loaded.items()}          # fallback: stored own-set metrics
    common_n = None
    if trained_ps:
        common = set.intersection(*[set(p) for p in trained_ps.values()])
        common_n = len(common)
        for m in trained_ps:
            out[m] = _aggregate({i: trained_ps[m][i] for i in common})
        base_ps = loaded["base"]["per_scenario"] if loaded.get("base", {}).get("per_scenario") else None
        if base_ps:
            base_common = {i: base_ps[i] for i in common if i in base_ps}
            if base_common:
                out["base"] = _aggregate(base_common)
    return out, common_n


def aggregate(eval_root, seeds, methods, primary):
    """Build the across-seed summary + the primary paired contrast."""
    metric_keys = ("mean_reward", "deal_rate", "mean_ratio")
    loaded = {s: _per_seed_metrics(eval_root, s, methods) for s in seeds}
    per_seed_metrics = {s: loaded[s][0] for s in seeds}
    per_seed_common_n = {s: loaded[s][1] for s in seeds}
    by_method = {}
    for m in list(methods) + ["base"]:
        recs = [(s, per_seed_metrics[s][m]) for s in seeds if m in per_seed_metrics[s]]
        if not recs:
            continue
        by_method[m] = {
            "seeds_found": [s for s, _ in recs],
            **{k: describe([mt.get(k) for _, mt in recs]) for k in metric_keys},
        }

    a, b = primary
    per_seed, diffs = [], []
    for s in seeds:
        ms = per_seed_metrics[s]
        if a not in ms or b not in ms:
            continue
        # diff over the same shared scenario set as by_method, not the stored own-set means
        diff = ms[a].get("mean_reward") - ms[b].get("mean_reward")
        ra, rb = load_seed_eval(eval_root, s, a), load_seed_eval(eval_root, s, b)
        within = paired_stats(ra.get("per_scenario"), rb.get("per_scenario")) \
            if ra and rb and ra.get("per_scenario") and rb.get("per_scenario") else None
        per_seed.append({"seed": s, "mean_reward_diff": round(diff, 4),
                         "common_n": per_seed_common_n.get(s), "within_seed_paired": within})
        diffs.append(diff)

    return {
        "eval_root": eval_root,
        "seeds_requested": list(seeds),
        "methods": list(methods),
        "n_seeds": len(seeds),
        "ci_eligible": len(seeds) >= CI_MIN_SEEDS,
        "stats_note": ("N>=5: bootstrap CI / IQM reported." if len(seeds) >= CI_MIN_SEEDS else
                       f"N={len(seeds)}<{CI_MIN_SEEDS}: descriptive ranges only; the headline is the "
                       "within-seed Wilcoxon paired test + across-seed sign-consistency."),
        "per_seed_common_n": per_seed_common_n,
        "by_method": by_method,
        "primary_contrast": {
            "pair": f"{a}_vs_{b}",
            "per_seed": per_seed,
            "across_seed": paired_contrast_across_seeds(diffs),
        },
    }


def main():
    ap = argparse.ArgumentParser(description="Aggregate per-seed evals into the headline comparison.")
    ap.add_argument("--seeds", type=int, nargs="+", required=True, help="seeds to aggregate")
    ap.add_argument("--methods", nargs="+", default=["sft", "grpo", "ppo"],
                    choices=["sft", "grpo", "ppo"])
    ap.add_argument("--primary", nargs=2, default=["ppo", "sft"],
                    metavar=("A", "B"), help="pre-registered primary contrast A vs B")
    ap.add_argument("--eval-root", default=os.path.join(ROOT, "runs"),
                    help="dir holding eval_s{seed}/ subdirs (default <repo>/runs)")
    ap.add_argument("--out", default=None, help="output path (default <eval-root>/aggregate.json)")
    args = ap.parse_args()

    result = aggregate(args.eval_root, args.seeds, args.methods, tuple(args.primary))
    out = args.out or os.path.join(args.eval_root, "aggregate.json")
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out, "w") as f:
        json.dump(result, f, indent=2)
    print(f"wrote {out}", flush=True)

    if not result["by_method"]:
        # exit non-zero so the orchestrator withholds the completion marker
        print(f"  ERROR: no eval JSONs found under {args.eval_root} (eval_s*/); aggregate is empty.",
              flush=True)
        raise SystemExit(2)

    print(f"\n=== ACROSS-SEED ({result['n_seeds']} seeds) ===", flush=True)
    print(f"  {result['stats_note']}", flush=True)
    for m, agg in result["by_method"].items():
        mr = agg["mean_reward"]
        ci = f"  ci95={mr['ci95']}" if "ci95" in mr else ""
        print(f"  {m:5s}  mean_reward={mr['mean']} (seeds {mr['values']}){ci}  "
              f"deal_rate={agg['deal_rate']['mean']}", flush=True)
    pc = result["primary_contrast"]
    ac = pc["across_seed"]
    print(f"\n  PRIMARY {pc['pair']}: per-seed diffs {ac['diffs']}  "
          f"all_same_sign={ac['all_same_sign']}  direction={ac['direction']}", flush=True)
    for row in pc["per_seed"]:
        w = row["within_seed_paired"]
        wp = w.get("wilcoxon_p") if w else None
        print(f"    seed {row['seed']}: Δreward={row['mean_reward_diff']}  within-seed wilcoxon_p={wp}",
              flush=True)


if __name__ == "__main__":
    main()
