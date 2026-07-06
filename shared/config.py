"""SharedConfig holds values constant across PPO/GRPO/SFT; each method subclasses it."""

import os
from dataclasses import dataclass, field

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def runs_base():
    """Root for all run outputs: /workspace on RunPod, else <repo>/runs."""
    return "/workspace" if os.path.isdir("/workspace") else os.path.join(ROOT, "runs")


@dataclass
class SharedConfig:
    # Qwen3.5-4B is multimodal; load_base narrows to the text tower. NEG_MODEL overrides.
    model_name: str = field(default_factory=lambda: os.environ.get("NEG_MODEL", "Qwen/Qwen3.5-4B"))
    dtype: str = "bfloat16"
    load_in_4bit: bool = False            # 16-bit LoRA on a 48GB A40; flip True only <=24GB
    trust_remote_code: bool = True
    max_seq_length: int = 2048            # PPO overrides to 4096
    # Kernels (causal-conv1d + fla) are a speed bonus, not a correctness requirement; missing
    # ones warn and run slow. True hard-fails instead.
    require_fast_kernels: bool = False

    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.0
    # all-linear LoRA, held constant across SFT/GRPO/PPO
    target_modules: tuple = ("q_proj", "k_proj", "v_proj", "o_proj",
                             "gate_proj", "up_proj", "down_proj")

    reserve_fraction: float = 0.5         # reserve = 0.5 * listing; the loaders assert the data agrees.
    max_new_tokens: int = 100
    temperature: float = 1.0
    top_p: float = 1.0

    # train and grade buyers are different model families; reasoning off on both
    train_buyer_model: str = "gemini-2.5-flash"
    grade_buyer_model: str = "gpt-5.4-nano"   # eval only
    buyer_max_chars: int = 500
    num_turns: int = 8                    # seller turns per episode (fixed horizon, no early stop)

    # reward judge: transcript -> {deal_reached, agreed_price, close_turn}; "" -> deterministic judge
    judge_model: str = "gemini-3.1-flash-lite"
    # backup judge, tried when the primary is unusable; if it also fails the sample is
    # dropped, never scored. "" disables.
    judge_backup_model: str = "gemini-3.5-flash"

    # abort (non-zero, resumable) after this many consecutive fully-dropped updates; 0 disables
    abort_after_empty_updates: int = 5

    seed: int = 42

    data_dir: str = os.path.join(ROOT, "data")

    # Persistence / RunPod (shared/persistence.py).
    run_name: str = "run"                 # subclass sets ppo_runs / grpo_runs / sft_runs
    checkpoint_every: int = 5
    keep_last_k_checkpoints: int = 3      # prune older step_* dirs; 0 = keep all
    resume: bool = True
    hf_repo_id: str = ""                  # subclass sets the durable mirror repo; "" = no push
    stop_pod_on_finish: bool = True
    # thread the seed into run_name + hf_repo_id so multi-seed runs never clobber each other
    seed_in_path: bool = False

    def __post_init__(self):
        # idempotent: a repeat __post_init__ must not double-append; runs before paths derive from run_name
        if self.seed_in_path and self.run_name and not self.run_name.endswith(f"_s{self.seed}"):
            self.run_name = f"{self.run_name}_s{self.seed}"
            if self.hf_repo_id and not self.hf_repo_id.endswith(f"-s{self.seed}"):
                self.hf_repo_id = f"{self.hf_repo_id}-s{self.seed}"
        self.output_dir = os.path.join(runs_base(), self.run_name)
        self.checkpoint_dir = os.path.join(self.output_dir, "checkpoints")
        self.log_file = os.path.join(self.output_dir, "training_log.csv")
        # one JSONL record per episode/completion: full transcript + reward
        self.transcript_file = os.path.join(self.output_dir, "transcripts.jsonl")
        self.slices_dir = os.path.join(self.data_dir, "slices")
        self.judge_cache_path = os.path.join(self.data_dir, "judge_cache.jsonl")

    @property
    def torch_dtype(self):
        import torch
        return torch.bfloat16 if self.dtype == "bfloat16" else torch.float16
