#!/usr/bin/env bash
# Sanitized SFT re-eval, run on the pod. Order of operations is deliberate: cheap preflights
# (buyer + judge one-call smoke) run before any model download or GPU time, per-scenario pacing
# is on (NEG_EVAL_VERBOSE), each seed's results mirror to HF the moment they exist, and every
# eval attempt runs under a hard timeout so a hang costs one attempt, not the whole run.
set -u
cd "$(dirname "$0")"

RESULTS_REPO="ShallowLearning/negotiation-results"
export NEG_EVAL_VERBOSE=1

mark() {  # mark <path_in_repo> [message]
  M="$1" RESULTS_REPO="$RESULTS_REPO" python - <<'EOF'
import io
import os
from huggingface_hub import HfApi
api = HfApi(token=os.environ.get("HF_TOKEN"))
api.upload_file(path_or_fileobj=io.BytesIO(b"x"), path_in_repo=os.environ["M"],
                repo_id=os.environ["RESULTS_REPO"], repo_type="model")
print(f"marker {os.environ['M']} pushed", flush=True)
EOF
}

push_log() {
  RESULTS_REPO="$RESULTS_REPO" python - <<'EOF' || true
import os
from huggingface_hub import HfApi
api = HfApi(token=os.environ.get("HF_TOKEN"))
if os.path.exists("/workspace/sanitized_eval.log"):
    api.upload_file(path_or_fileobj="/workspace/sanitized_eval.log",
                    path_in_repo="sanitized_eval.log",
                    repo_id=os.environ["RESULTS_REPO"], repo_type="model",
                    commit_message="heartbeat")
EOF
}

pip install -q huggingface_hub 2>/dev/null || true
mark "_SANITIZED_STARTED"
(
  while true; do
    sleep 300
    date -u "+heartbeat %Y-%m-%dT%H:%M:%SZ" >> /workspace/sanitized_eval.log
    push_log
  done
) &

grep -vE 'causal-conv1d|flash-linear-attention' requirements.lock > /tmp/req-eval.txt
pip install --no-build-isolation -r /tmp/req-eval.txt

# Preflight: one real buyer call and one real judge call from THIS box before anything
# expensive. The first run burned 10 GPU-hours before the first API warning surfaced.
python - <<'EOF'
import sys
sys.path.insert(0, ".")
from shared.config import SharedConfig
from shared.judge import make_judge
from shared.buyer import make_buyer
cfg = SharedConfig(run_name="preflight")
sc = {"id": "smoke", "listing": 100.0, "title": "Test lamp", "description": "A lamp.",
      "buyer_target": 70.0, "reserve": 50.0}
turns = [("buyer", "Would you take $70 for the lamp?"),
         ("seller", "Yes, $70 works. It is a deal."),
         ("buyer", "Great, deal at $70!")]
verdict = make_judge(cfg)(turns, sc)
assert verdict and verdict[0] == 70.0, f"judge smoke returned {verdict}"
print("preflight judge OK", flush=True)
reply = make_buyer(cfg, "grade").reply(
    [("buyer", "Is this still available?"), ("seller", "Yes it is. Asking $100.")], sc, seed=0)
assert reply, "buyer smoke returned no reply"
print("preflight buyer OK", flush=True)
EOF
if [ $? -ne 0 ]; then
  echo "PREFLIGHT FAILED"; push_log; mark "_PREFLIGHT_FAILED"; exit 1
fi

python - <<'EOF'
from huggingface_hub import snapshot_download
for n in (1, 2, 3):
    snapshot_download(f"ShallowLearning/negotiation-sft-qwen3.5-4b-s{n}",
                      local_dir=f"/workspace/sft_runs_s{n}", allow_patterns=["lora_final/*"])
    print(f"adapter s{n} in place", flush=True)
EOF

for N in 1 2 3; do
  for attempt in 1 2 3; do
    timeout 14400 python -m eval_methods --methods sft --no-base --seed "$N" --split test \
      --limit 150 --sanitize-leak --out "/workspace/eval_sanitized_s$N" && break
    echo "seed $N attempt $attempt failed or timed out; retrying in 120s"
    push_log
    sleep 120
  done
  N="$N" RESULTS_REPO="$RESULTS_REPO" python - <<'EOF' || true
import os
from huggingface_hub import HfApi
n, repo = os.environ["N"], os.environ["RESULTS_REPO"]
api = HfApi(token=os.environ.get("HF_TOKEN"))
api.upload_folder(folder_path=f"/workspace/eval_sanitized_s{n}", path_in_repo=f"eval_sanitized_s{n}",
                  repo_id=repo, repo_type="model", commit_message=f"sanitized eval s{n}")
print(f"uploaded eval_sanitized_s{n}", flush=True)
EOF
  push_log
done

mark "_SANITIZED_DONE"
echo "ALL_SANITIZED_DONE"
push_log
