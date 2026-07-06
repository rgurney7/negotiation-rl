"""Pod-side orchestrator for the multi-seed final run; pod teardown is local_killer.py's job."""

import argparse
import json
import os
import subprocess
import sys
import time

from shared.config import ROOT, runs_base
from shared import persistence

# sentinels live on the /workspace volume, not the repo clone, so resume state survives restarts
STATE_DIR = os.path.join(runs_base(), "orchestrator")
DEFAULT_RESULTS_REPO = "ShallowLearning/negotiation-results"


def sentinel_path(unit):
    return os.path.join(STATE_DIR, f"{unit}.done")


def done_units():
    if not os.path.isdir(STATE_DIR):
        return set()
    return {n[:-5] for n in os.listdir(STATE_DIR) if n.endswith(".done")}


def mark_done(unit):
    os.makedirs(STATE_DIR, exist_ok=True)
    with open(sentinel_path(unit), "w") as f:
        f.write(unit)
        f.flush()
        os.fsync(f.fileno())


def eval_unit_name(seed, methods, split, eval_limit):
    """Sentinel name for one seed's eval, parameterized by what was evaluated."""
    lim = f"L{eval_limit}" if eval_limit else "full"
    return f"eval_s{seed}_{'-'.join(sorted(methods))}_{split}_{lim}"


def plan_units(seeds, methods, done, split="test", eval_limit=None):
    """Ordered (kind, method, seed, unit) list of units still to run: all training, then one eval per seed."""
    plan = []
    for s in seeds:
        for m in methods:
            unit = f"{m}_s{s}"
            if unit not in done:
                plan.append(("train", m, s, unit))
    for s in seeds:
        unit = eval_unit_name(s, methods, split, eval_limit)
        if unit not in done:
            plan.append(("eval", None, s, unit))
    return plan


def aggregate_covers(agg, seeds, methods):
    """True iff aggregate.json covers every trained method over exactly the requested seeds (base exempt)."""
    by_method = agg.get("by_method") or {}
    want = set(seeds)
    for m in methods:
        if set(by_method.get(m, {}).get("seeds_found", [])) != want:
            return False
    return True


def _run(cmd, env):
    print(f"\n>>> {' '.join(cmd)}", flush=True)
    return subprocess.run(cmd, env=env).returncode


def _run_with_retry(cmd, env, tries=2, retry_sleep=300):
    rc = 1
    for attempt in range(1, tries + 1):
        rc = _run(cmd, env)
        if rc == 0:
            return 0
        print(f"  unit rc={rc} (attempt {attempt}/{tries})", flush=True)
        if attempt < tries and retry_sleep:
            print(f"  sleeping {retry_sleep}s before the retry (outage-shaped failures need time "
                  "to clear)...", flush=True)
            time.sleep(retry_sleep)
    return rc


def _eval_cmd(seed, split, first_seed, methods, eval_limit=None):
    # eval only the trained methods; eval_methods defaults to all three
    cmd = [sys.executable, "-m", "eval_methods", "--seed", str(seed), "--split", split,
           "--methods", *methods]
    if eval_limit:
        cmd += ["--limit", str(eval_limit)]
    if seed != first_seed:
        cmd.append("--no-base")     # base is seed-independent; eval it once
    return cmd


def _verify_openai_key():
    """Smoke the grade model with the real call shape (no temperature: gpt-5 family 400s on it)."""
    key = os.environ.get("OPEN_AI_API_KEY") or os.environ.get("OPENAI_API_KEY")
    if not key:
        raise SystemExit("OPEN_AI_API_KEY missing: the grade buyer needs it at eval time. "
                         "Set it in the pod .env before launching.")
    from openai import OpenAI
    from shared.config import SharedConfig
    model = SharedConfig().grade_buyer_model
    try:
        resp = OpenAI(api_key=key, timeout=30.0).chat.completions.create(
            model=model,
            messages=[{"role": "system", "content": "Reply with one word."},
                      {"role": "user", "content": "Say ok."}],
            reasoning_effort="none",
            max_completion_tokens=16,
        )
        if not resp.choices:
            raise RuntimeError("completion returned no choices")
        print(f"  OpenAI grade buyer OK (model {model} answered)", flush=True)
    except Exception as e:
        raise SystemExit(f"OpenAI grade-buyer smoke failed for model {model} "
                         f"({type(e).__name__}: {str(e)[:120]}) — fix the key or model access "
                         "before spending GPU-hours on a run whose eval cannot score.")


def main():
    ap = argparse.ArgumentParser(description="Final multi-seed run orchestrator (pod-side).")
    ap.add_argument("--seeds", type=int, nargs="+", required=True)
    ap.add_argument("--methods", nargs="+", default=["sft", "grpo", "ppo"],
                    choices=["sft", "grpo", "ppo"])
    ap.add_argument("--primary", nargs=2, default=["ppo", "sft"], metavar=("A", "B"))
    ap.add_argument("--split", default="test", choices=["test", "validation"])
    ap.add_argument("--eval-limit", type=int, default=None,
                    help="seeded sample of N DISTINCT listings per method (the plan's cost model "
                         "uses 150; None = the full cold-open-filtered test split, ~3.3x the "
                         "cost/time). Set this to keep the run inside the wall-clock ceiling.")
    ap.add_argument("--max-hours", type=float, default=72.0,
                    help="soft cap: stop launching NEW units past this (hard ceiling is the local "
                         "killer). Pilot-measured N=3 is ~50h (15.5h/seed + eval), so the cap must "
                         "clear that with margin while staying under the killer ceiling minus one unit.")
    ap.add_argument("--results-repo", default=DEFAULT_RESULTS_REPO,
                    help="HF repo for the _ALL_DONE marker + aggregate.json the local killer polls")
    ap.add_argument("--no-hf-check", action="store_true",
                    help="OFFLINE DRY RUN ONLY: skips the fail-fast HF smoke pushes AND the "
                         "stale-completion-marker clear. Never use on a real pod run — though the "
                         "killer's absent->present arming would still catch a stale marker.")
    args = ap.parse_args()

    try:                                            # pick up HF_TOKEN/API keys from the repo .env
        from dotenv import load_dotenv
        load_dotenv(os.path.join(ROOT, ".env"))
    except Exception:
        pass

    env = dict(os.environ, NEG_ORCHESTRATED="1")   # trainers must not self-teardown the pod
    deadline = time.time() + args.max_hours * 3600 if args.max_hours else None
    first_seed = args.seeds[0]

    marker = persistence.done_marker_name(args.seeds, args.methods)

    # fail fast: verify HF access before spending hours of training
    if not args.no_hf_check:
        from types import SimpleNamespace
        from ppo.config import PPOConfig
        persistence.verify_hf_token(PPOConfig())
        persistence.verify_hf_token(SimpleNamespace(hf_repo_id=args.results_repo))
        # a stale marker from a prior same-shaped run would trigger the killer's first poll
        persistence.clear_done_markers(args.results_repo, [marker, "_ALL_DONE"])
        # validate both buyer keys now; the OpenAI key isn't otherwise exercised until eval
        if not os.environ.get("GOOGLE_API_KEY"):
            raise SystemExit("GOOGLE_API_KEY missing: the train buyer + judge need it from the "
                             "first RL unit. Set it in the pod .env before launching.")
        _verify_openai_key()

    if os.path.isdir("/workspace") and not os.environ.get("HF_HOME", "").startswith("/workspace"):
        print("  WARN: HF_HOME is not on /workspace — the model cache sits on the ephemeral "
              "container disk and re-downloads (~8GB) after any pod restart. "
              "Recommended: export HF_HOME=/workspace/hf_cache", flush=True)

    print(f"Completion marker for this run: {marker}\n"
          f"Start the LOCAL killer with matching args, e.g.:\n"
          f"  python local_killer.py --pod-id <id> --seeds {' '.join(map(str, args.seeds))}"
          + (f" --methods {' '.join(args.methods)}" if args.methods != ["sft", "grpo", "ppo"] else ""),
          flush=True)

    if args.eval_limit is None:
        print("  WARN: --eval-limit is unset -> full split eval (~13-17h per seed), so N=3 can run "
              "~90h+ with a retry and approach the killer's 96h ceiling. The recommended recipe is "
              "--eval-limit 150; if you really want the full split, raise the killer --max-hours.",
              flush=True)

    plan = plan_units(args.seeds, args.methods, done_units(), args.split, args.eval_limit)
    print(f"Orchestrating {len(plan)} remaining units over seeds {args.seeds}: "
          f"{[u for _, _, _, u in plan]}", flush=True)

    stopped_early = False
    for kind, method, seed, unit in plan:
        if deadline and time.time() > deadline:
            print(f"  soft wall-clock reached; not launching {unit} (local killer owns the hard stop).",
                  flush=True)
            stopped_early = True
            break
        if kind == "train":
            rc = _run_with_retry([sys.executable, "-m", f"{method}.train", "--seed", str(seed)], env)
        else:
            rc = _run_with_retry(_eval_cmd(seed, args.split, first_seed, args.methods, args.eval_limit), env)
        if rc != 0:
            # sentinel not written, so a restart re-runs exactly this unit
            print(f"  FAILED {unit} (rc={rc}); aborting. Fix + re-run; completed units are skipped.",
                  flush=True)
            raise SystemExit(rc)
        mark_done(unit)
        if kind == "eval" and not args.no_hf_check:
            # warn-only early mirror; the gated end-of-run mirror below is the real gate
            if not persistence.push_eval_results(args.results_repo, runs_base(), [seed]):
                print(f"  (early mirror of eval_s{seed} failed; the gated end-of-run mirror retries)",
                      flush=True)

    if stopped_early:
        print("\nORCHESTRATOR STOPPED at the soft wall-clock; re-run to finish the rest.", flush=True)
        return

    # aggregate from the same base dir eval wrote to; aggregate's <repo>/runs default is wrong on a pod
    eval_root = runs_base()
    aggregate_json = os.path.join(eval_root, "aggregate.json")
    rc = _run([sys.executable, os.path.join(ROOT, "aggregate_seeds.py"),
               "--seeds", *map(str, args.seeds),
               "--methods", *args.methods, "--primary", *args.primary,
               "--eval-root", eval_root, "--out", aggregate_json], env)

    # gate the marker on full seed x method coverage; on failure the pod stays up
    ok = rc == 0
    if ok:
        try:
            agg = json.load(open(aggregate_json))
            ok = aggregate_covers(agg, args.seeds, args.methods)
            if not ok:
                print(f"  aggregate coverage incomplete: "
                      f"{ {m: agg.get('by_method', {}).get(m, {}).get('seeds_found') for m in args.methods} } "
                      f"vs requested seeds {args.seeds}.", flush=True)
        except Exception as e:
            print(f"  could not read {aggregate_json}: {e}", flush=True)
            ok = False
    if not ok:
        print(f"  aggregation failed or incomplete under {eval_root} (rc={rc}); NOT pushing {marker} "
              "so the pod stays up and the results aren't lost. Check the eval dirs.", flush=True)
        raise SystemExit(rc or 3)

    # mirror all raw per-seed evals to HF before the marker, so teardown can't erase them
    mirrored = False
    for attempt in range(1, 4):
        mirrored = persistence.push_eval_results(args.results_repo, eval_root, args.seeds)
        if mirrored:
            break
        print(f"  eval mirror attempt {attempt}/3 failed; retrying in 60s...", flush=True)
        time.sleep(60)
    if not mirrored:
        print(f"  could not mirror the raw eval dirs after 3 attempts; NOT pushing {marker}. "
              "The pod stays up — fix connectivity and re-run (all units resume).", flush=True)
        raise SystemExit(4)
    if not persistence.push_done_marker(args.results_repo, payload_path=aggregate_json, name=marker):
        print(f"  completion-marker push failed; the killer will NOT tear down. Re-run to retry "
              "(everything else is complete and resumes instantly).", flush=True)
        raise SystemExit(5)
    print(f"\nORCHESTRATOR COMPLETE — pushed {marker}; local killer may remove the pod.", flush=True)


if __name__ == "__main__":
    main()
