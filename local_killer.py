"""Local watchdog: removes the pod when run_final's completion marker appears on HF, or at a wall-clock ceiling."""

import argparse
import os
import subprocess
import sys
import time

from shared.persistence import done_marker_name


def _load_env():
    try:
        from dotenv import load_dotenv, find_dotenv
        load_dotenv(find_dotenv(usecwd=True))
        root_env = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
        if os.path.exists(root_env):
            load_dotenv(root_env)
    except Exception:
        pass


def verify_local_hf_auth():
    if len(os.environ.get("HF_TOKEN", "").strip()) < 20:
        raise SystemExit("HF_TOKEN missing/too short in the LOCAL env; the killer could never see "
                         "the completion marker and would bill to the ceiling. Fix .env first.")
    from huggingface_hub import HfApi
    try:
        who = HfApi(token=os.environ["HF_TOKEN"]).whoami()
        print(f"  local HF token OK (user: {who.get('name', '?')})", flush=True)
    except Exception as e:
        raise SystemExit(f"local HF_TOKEN failed authentication ({type(e).__name__}); the killer "
                         "would poll forever seeing 'not done'. Fix the token, then restart.")


def marker_present(repo_id, name, state):
    """True/False on a successful poll; None on a failed poll (a failed poll carries no information)."""
    try:
        from huggingface_hub import HfApi
        api = HfApi(token=os.environ.get("HF_TOKEN"))
        files = set(api.list_repo_files(repo_id, repo_type="model"))
        state["errors"] = 0
        return name in files
    except Exception as e:
        state["errors"] = state.get("errors", 0) + 1
        if state["errors"] in (1, 3) or state["errors"] % 6 == 0:
            print(f"  WARN: {state['errors']} consecutive poll failure(s) "
                  f"({type(e).__name__}) — a broken poll looks identical to 'not done yet'. "
                  "If this persists, check HF_TOKEN / --results-repo.", flush=True)
        return None


def should_fire(present, state, allow_preexisting=False):
    """Fire only on an absent -> present transition, so a stale same-shape marker can't fire."""
    if present is None:
        return False
    if allow_preexisting:
        return present
    if not present:
        state["armed"] = True
        return False
    if state.get("armed"):
        return True
    if not state.get("warned_stale"):
        state["warned_stale"] = True
        print("  marker already present at the first poll — likely a STALE marker from a prior "
              "same-shaped run. NOT tearing down; run_final's startup clear removes it (it must "
              "disappear then reappear). If the run truly finished before this watchdog started, "
              "restart with --allow-preexisting-marker.", flush=True)
    return False


def verify_runpodctl(pod_id):
    """Check runpodctl is usable and the pod id is actually listed, before we depend on it at fire time."""
    import shutil
    if not shutil.which("runpodctl"):
        raise SystemExit("runpodctl not on PATH — the killer could never remove the pod. "
                         "Install it (brew install runpod/runpodctl/runpodctl), then restart.")
    if not os.environ.get("RUNPOD_API_KEY"):
        raise SystemExit("RUNPOD_API_KEY missing from the LOCAL env — the killer could never "
                         "remove the pod. Fix .env, then restart.")
    proc = subprocess.run(["runpodctl", "get", "pod", pod_id], capture_output=True, text=True)
    if proc.returncode != 0:
        raise SystemExit(f"runpodctl get pod {pod_id} failed (rc={proc.returncode}) — bad "
                         "RUNPOD_API_KEY or CLI error. Fix it, then restart; a broken teardown "
                         "discovered at fire time bills until you notice.")
    # exact ID-column match: `runpodctl get pod <bogus>` exits 0 with a header-only table
    ids = {line.split()[0] for line in (proc.stdout or "").splitlines()
           if line.split() and line.split()[0] != "ID"}
    if pod_id not in ids:
        raise SystemExit(f"pod id {pod_id} is not a row in `runpodctl get pod` output (found: "
                         f"{sorted(ids) or 'none'}) — likely a typo or a pod that isn't up yet. The "
                         "killer removes by exact id, so a wrong id would never stop billing. Fix "
                         "--pod-id, then restart.")
    print(f"  runpodctl OK (pod {pod_id} listed)", flush=True)


def remove_pod(pod_id, tries=3):
    for attempt in range(1, tries + 1):
        print(f"  removing pod {pod_id} (halts all billing), attempt {attempt}/{tries} ...", flush=True)
        try:
            rc = subprocess.run(["runpodctl", "remove", "pod", pod_id]).returncode
        except OSError as e:
            print(f"  runpodctl unavailable ({e}); cannot remove automatically.", flush=True)
            break
        print(f"  runpodctl remove pod -> rc={rc}", flush=True)
        if rc == 0:
            return True
        time.sleep(30)
    print(f"  !!! could not remove pod {pod_id} — REMOVE IT MANUALLY, it is still billing:\n"
          f"    runpodctl remove pod {pod_id}", flush=True)
    return False


def main():
    ap = argparse.ArgumentParser(description="Local watchdog that removes the key-less final-run pod.")
    ap.add_argument("--pod-id", required=True, help="the pod id to remove (from runpodctl get pod)")
    ap.add_argument("--seeds", type=int, nargs="+", default=None,
                    help="the SAME seeds you gave run_final — derives this run's marker name")
    ap.add_argument("--methods", nargs="+", default=["sft", "grpo", "ppo"],
                    choices=["sft", "grpo", "ppo"],
                    help="the SAME methods you gave run_final (default all three)")
    ap.add_argument("--marker", default=None,
                    help="explicit marker name (overrides --seeds/--methods derivation)")
    ap.add_argument("--results-repo", default="ShallowLearning/negotiation-results",
                    help="HF repo the orchestrator pushes the completion marker to")
    ap.add_argument("--max-hours", type=float, default=96.0,
                    help="hard ceiling: remove the pod regardless. This is a HANG backstop, not the "
                         "budget gate — normal completion fires the marker and tears down "
                         "immediately, so a generous ceiling costs nothing but must sit WELL above the "
                         "orchestrator soft cap + one unit (~12h) so it never kills mid-eval.")
    ap.add_argument("--poll-seconds", type=int, default=300, help="how often to check the done marker")
    ap.add_argument("--no-marker", action="store_true",
                    help="ignore the HF marker; remove strictly on the wall-clock ceiling")
    ap.add_argument("--allow-preexisting-marker", action="store_true",
                    help="honor a marker that is ALREADY present at the first poll (use when "
                         "restarting the killer after the run finished; default requires an "
                         "absent->present transition so a stale same-shape marker can't fire)")
    args = ap.parse_args()

    if args.no_marker:
        marker = None
    elif args.marker:
        marker = args.marker
    elif args.seeds:
        marker = done_marker_name(args.seeds, args.methods)
    else:
        ap.error("pass --seeds (matching run_final) so the killer watches THIS run's marker, "
                 "or --marker/--no-marker explicitly")

    _load_env()
    verify_runpodctl(args.pod_id)
    if marker is not None:
        verify_local_hf_auth()
    deadline = time.time() + args.max_hours * 3600
    print(f"watching pod {args.pod_id}: marker {marker or '(disabled)'} in {args.results_repo} "
          f"OR {args.max_hours}h ceiling", flush=True)

    state = {}
    while True:
        if time.time() > deadline:
            print("  wall-clock ceiling reached.", flush=True)
            # non-zero exit if removal failed: the pod is still billing
            return 0 if remove_pod(args.pod_id) else 1
        if marker is not None:
            present = marker_present(args.results_repo, marker, state)
            if should_fire(present, state, args.allow_preexisting_marker):
                print(f"  {marker} found; run complete.", flush=True)
                return 0 if remove_pod(args.pod_id) else 1
        remaining = int(deadline - time.time())
        print(f"  ... not done yet; {remaining // 60} min until the ceiling. Sleeping "
              f"{args.poll_seconds}s.", flush=True)
        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    sys.exit(main())
