"""Checkpointing, HF mirroring, and pod teardown; `agent` is anything with .save(dir)/.load(dir)."""

import csv
import json
import os

# stop flag set by the signal handler; trainers opt in via install_signal_handlers()
_STOP = {"flag": False}


def install_signal_handlers():
    """SIGTERM/SIGINT set the stop flag so the loop can checkpoint at the next boundary.
    Guarded: signal registration only works in the main thread."""
    import signal

    def _handler(signum, frame):
        _STOP["flag"] = True
        print(f"  signal {signum} received; will checkpoint + stop at the next loop boundary.",
              flush=True)

    try:
        signal.signal(signal.SIGTERM, _handler)
        signal.signal(signal.SIGINT, _handler)
    except Exception as e:
        print(f"  WARN: could not install signal handlers: {e}", flush=True)


def stop_requested():
    return _STOP["flag"]


def outage_abort(consecutive_empty, limit):
    """True when consecutive fully-dropped updates indicate a sustained outage. limit<=0 disables."""
    return limit > 0 and consecutive_empty >= limit


def write_run_config(cfg):
    import dataclasses
    payload = dataclasses.asdict(cfg)
    payload["git_commit"] = _git_commit()
    write_json_atomic(os.path.join(cfg.output_dir, "run_config.json"), payload)


def _git_commit():
    try:
        import subprocess
        return subprocess.check_output(["git", "rev-parse", "HEAD"],
                                       stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return "unknown"


def log_row(cfg, fields, metrics):
    new = not os.path.exists(cfg.log_file)
    os.makedirs(os.path.dirname(cfg.log_file), exist_ok=True)
    with open(cfg.log_file, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        if new:
            w.writeheader()
        w.writerow({k: metrics.get(k, "") for k in fields})
        f.flush()
        os.fsync(f.fileno())


def append_jsonl(path, record):
    """Append one fsync'd JSON line; creates parent dirs."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(record, default=str) + "\n")
        f.flush()
        os.fsync(f.fileno())


def log_transcript(cfg, record):
    append_jsonl(cfg.transcript_file, record)


def write_json_atomic(path, obj):
    """Atomic JSON write (temp file + fsync + os.replace)."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2, default=str)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def _fsync_dir(path):
    """Best-effort directory fsync; not permitted everywhere, failures ignored."""
    try:
        fd = os.open(path, os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    except Exception:
        pass


def mark_preview(cfg):
    # isolate a smoke run: own run dir, no HF push, never stop the pod
    cfg.run_name = cfg.run_name.rstrip("/") + "_preview"
    cfg.hf_repo_id = ""
    cfg.stop_pod_on_finish = False
    cfg.checkpoint_every = max(1, min(cfg.checkpoint_every,
                                      getattr(cfg, "num_updates", cfg.checkpoint_every)))
    cfg.__post_init__()


def save_checkpoint(agent, optimizer, cfg, step):
    import torch

    ckpt = os.path.join(cfg.checkpoint_dir, f"step_{step:04d}")
    os.makedirs(ckpt, exist_ok=True)
    agent.save(ckpt)
    if optimizer is not None:
        torch.save(optimizer.state_dict(), os.path.join(ckpt, "optimizer.pt"))
    # sentinel written last: presence proves the dir is complete; resume skips dirs without it
    with open(os.path.join(ckpt, "_COMPLETE"), "w") as f:
        f.write(str(step))
        f.flush()
        os.fsync(f.fileno())
    _fsync_dir(ckpt)                            # persist the step dir's entries before the pointer
    # advance the latest pointer atomically so it never names a partial dir
    latest = os.path.join(cfg.checkpoint_dir, "latest.txt")
    tmp = latest + ".tmp"
    with open(tmp, "w") as f:
        f.write(str(step))
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, latest)
    _fsync_dir(cfg.checkpoint_dir)
    print(f"  checkpoint saved: step_{step:04d}", flush=True)
    _prune_old_checkpoints(cfg)


def _prune_old_checkpoints(cfg):
    """Keep only the K newest step_* dirs; K<=0 keeps all. Eval loads lora_final, so pruning is safe."""
    import shutil
    k = getattr(cfg, "keep_last_k_checkpoints", 0)
    if not k or k <= 0 or not os.path.isdir(cfg.checkpoint_dir):
        return
    steps = []
    for name in os.listdir(cfg.checkpoint_dir):
        if name.startswith("step_") and os.path.isdir(os.path.join(cfg.checkpoint_dir, name)):
            try:
                steps.append((int(name[len("step_"):]), name))
            except ValueError:
                pass
    for _, name in sorted(steps)[:-k]:         # everything but the K highest
        shutil.rmtree(os.path.join(cfg.checkpoint_dir, name), ignore_errors=True)


def load_checkpoint(agent, optimizer, cfg):
    import torch
    if not os.path.isdir(cfg.checkpoint_dir):
        return 0
    # resume from the highest step_* dir with the _COMPLETE sentinel and the adapter file;
    # latest.txt is not trusted
    best = None
    for name in os.listdir(cfg.checkpoint_dir):
        if not name.startswith("step_"):
            continue
        ckpt = os.path.join(cfg.checkpoint_dir, name)
        if not os.path.isdir(ckpt):
            continue
        if not os.path.exists(os.path.join(ckpt, "_COMPLETE")):
            continue
        if not os.path.exists(os.path.join(ckpt, "lora.safetensors")):
            continue
        try:
            step = int(name[len("step_"):])
        except ValueError:
            continue
        if best is None or step > best[0]:
            best = (step, ckpt)
    if best is None:
        return 0
    step, ckpt = best
    agent.load(ckpt)
    opt_path = os.path.join(ckpt, "optimizer.pt")
    if optimizer is not None and os.path.exists(opt_path):
        optimizer.load_state_dict(torch.load(opt_path, map_location=agent.device))
    print(f"  resumed from step {step}", flush=True)
    return step


def valid_hf_trainer_checkpoint(output_dir):
    """Newest complete HF Trainer checkpoint-* dir (parseable trainer_state.json), or None."""
    import re
    if not os.path.isdir(output_dir):
        return None
    candidates = []
    for name in os.listdir(output_dir):
        m = re.fullmatch(r"checkpoint-(\d+)", name)
        d = os.path.join(output_dir, name)
        if m and os.path.isdir(d):
            candidates.append((int(m.group(1)), d))
    # trainer_state.json must also parse: a torn save would crash resume identically on every retry
    for step, d in sorted(candidates, reverse=True):
        state = os.path.join(d, "trainer_state.json")
        if not os.path.exists(state):
            continue
        try:
            with open(state) as f:
                json.load(f)
        except Exception:
            print(f"  WARN: {state} is unparseable (torn save); skipping to an earlier checkpoint.",
                  flush=True)
            continue
        return d
    return None


def push_to_hub(cfg, step):
    # mirror the run dir to a private HF repo; returns False on failure
    if not cfg.hf_repo_id:
        return False
    try:
        from huggingface_hub import HfApi
        api = HfApi(token=os.environ.get("HF_TOKEN"))
        api.create_repo(cfg.hf_repo_id, private=True, exist_ok=True, repo_type="model")
        # skip the heavy resume-only artifacts; they live on the volume and eval only reads lora_final
        api.upload_folder(folder_path=cfg.output_dir, repo_id=cfg.hf_repo_id,
                          repo_type="model", commit_message=f"step {step}",
                          ignore_patterns=["checkpoints/*/optimizer.pt", "checkpoints/*/critic.pt"])
        print(f"  pushed to {cfg.hf_repo_id} @ step {step}", flush=True)
        return True
    except Exception as e:
        print(f"  WARN: HF push failed: {e}", flush=True)
        return False


def _should_teardown(cfg, uploaded_ok, pod_id, orchestrated):
    """Whether a trainer may remove the pod itself: never when orchestrated (the orchestrator owns
    teardown); else only if stop_pod_on_finish, on a pod, and the final push landed (or no mirror)."""
    if orchestrated:
        return False
    if not (cfg.stop_pod_on_finish and pod_id):
        return False
    if cfg.hf_repo_id and not uploaded_ok:
        return False
    return True


def hf_token_present(token=None):
    """True if HF_TOKEN looks usable; checks length only (real tokens are ~37 chars), no network."""
    token = os.environ.get("HF_TOKEN", "") if token is None else token
    return len(token.strip()) >= 20


def verify_hf_token(cfg):
    """Preflight HF auth check: token present + a 1-file smoke push. Raises on hard failure;
    False when pushing is disabled."""
    if not cfg.hf_repo_id:
        return False
    if not hf_token_present():
        raise RuntimeError(
            f"HF_TOKEN missing or too short (len={len(os.environ.get('HF_TOKEN', ''))}); refusing to "
            "start — every checkpoint push would fail and the run would lose its artifacts.")
    import io
    from huggingface_hub import HfApi
    api = HfApi(token=os.environ["HF_TOKEN"])
    api.create_repo(cfg.hf_repo_id, private=True, exist_ok=True, repo_type="model")
    api.upload_file(path_or_fileobj=io.BytesIO(b"ok"), path_in_repo=".hf_smoke",
                    repo_id=cfg.hf_repo_id, repo_type="model", commit_message="preflight smoke")
    print(f"  HF token verified; smoke push to {cfg.hf_repo_id} OK.", flush=True)
    return True


def done_marker_name(seeds, methods):
    """Completion-marker name from sorted seeds+methods, so markers from differently-shaped runs
    never collide."""
    return (f"_ALL_DONE_s{'-'.join(str(s) for s in sorted(seeds))}"
            f"_{'-'.join(sorted(methods))}")


def clear_done_markers(repo_id, names):
    """Delete stale completion markers at startup. Missing repo/marker is fine; a marker that
    exists but can't be deleted raises."""
    if not repo_id:
        return
    from huggingface_hub import HfApi
    api = HfApi(token=os.environ.get("HF_TOKEN"))
    try:
        files = set(api.list_repo_files(repo_id, repo_type="model"))
    except Exception as e:
        # only a missing repo means nothing-to-clear; any other failure must raise, since a
        # stale marker might still exist
        if type(e).__name__ == "RepositoryNotFoundError":
            return
        raise
    for name in names:
        if name not in files:
            continue
        print(f"  stale completion marker {name} found in {repo_id} (prior run); deleting it so the "
              "local killer can't fire on it.", flush=True)
        api.delete_file(path_in_repo=name, repo_id=repo_id, repo_type="model",
                        commit_message="clear stale completion marker")
    left = set(api.list_repo_files(repo_id, repo_type="model")) & set(names)
    if left:
        raise RuntimeError(
            f"could not delete stale completion marker(s) {sorted(left)} from {repo_id}; refusing to "
            "start — the local killer would remove this pod on its next poll.")


def push_eval_results(repo_id, eval_root, seeds):
    """Mirror each seed's eval_s{seed}/ dir to the results repo. Call before push_done_marker;
    True only when every seed's dir existed and uploaded."""
    if not repo_id:
        return False
    seeds = list(seeds)
    dirs = {s: os.path.join(eval_root, f"eval_s{s}") for s in seeds}
    missing = [s for s, d in dirs.items() if not os.path.isdir(d)]
    if missing:
        print(f"  WARN: eval dir(s) missing for seed(s) {missing} under {eval_root}; nothing "
              "mirrored.", flush=True)
        return False
    try:
        from huggingface_hub import HfApi
        api = HfApi(token=os.environ.get("HF_TOKEN"))
        api.create_repo(repo_id, private=True, exist_ok=True, repo_type="model")
        for s in seeds:
            api.upload_folder(folder_path=dirs[s], path_in_repo=f"eval_s{s}", repo_id=repo_id,
                              repo_type="model", commit_message=f"raw eval s{s}")
        print(f"  mirrored {len(seeds)}/{len(seeds)} eval dir(s) to {repo_id}", flush=True)
        return True
    except Exception as e:
        print(f"  WARN: could not mirror eval dirs: {e}", flush=True)
        return False


def push_done_marker(repo_id, payload_path=None, name="_ALL_DONE"):
    """Upload the payload first, then the completion marker; no marker (returns False) if the
    payload is missing or fails to upload."""
    if not repo_id:
        return False
    try:
        import io
        from huggingface_hub import HfApi
        api = HfApi(token=os.environ.get("HF_TOKEN"))
        api.create_repo(repo_id, private=True, exist_ok=True, repo_type="model")
        if payload_path is not None:
            if not os.path.exists(payload_path):
                print(f"  WARN: aggregate payload {payload_path} missing; NOT pushing {name} so the "
                      "pod stays up and the results aren't lost.", flush=True)
                return False
            api.upload_file(path_or_fileobj=payload_path, path_in_repo=os.path.basename(payload_path),
                            repo_id=repo_id, repo_type="model", commit_message="aggregate")
        api.upload_file(path_or_fileobj=io.BytesIO(b"done"), path_in_repo=name,
                        repo_id=repo_id, repo_type="model", commit_message="run complete")
        print(f"  pushed {name} to {repo_id}", flush=True)
        return True
    except Exception as e:
        print(f"  WARN: could not push done marker: {e}", flush=True)
        return False


def maybe_stop_pod(cfg, uploaded_ok):
    pod_id = os.environ.get("RUNPOD_POD_ID")
    orchestrated = os.environ.get("NEG_ORCHESTRATED") == "1"
    if not _should_teardown(cfg, uploaded_ok, pod_id, orchestrated):
        if orchestrated:
            print("  orchestrated run: teardown deferred to run_final.py / the local killer.", flush=True)
        elif cfg.hf_repo_id and not uploaded_ok and pod_id:
            print("  NOT removing pod: final HF upload did not succeed.", flush=True)
        return
    # remove, not stop: `stop` keeps the volume billing; `remove` deletes the pod and halts billing.
    print("  training complete; removing pod to halt billing.", flush=True)
    os.system(f"runpodctl remove pod {pod_id}")
