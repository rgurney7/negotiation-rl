from dataclasses import dataclass

from shared.config import SharedConfig


@dataclass
class SFTConfig(SharedConfig):
    run_name: str = "sft_runs"
    hf_repo_id: str = "ShallowLearning/negotiation-sft-qwen3.5-4b"
    # max_seq_length stays at the 2048 default; per-turn examples are short

    epochs: int = 3
    learning_rate: float = 2e-4
    batch_size: int = 2
    grad_accum: int = 4                   # effective batch = 8
    warmup_ratio: float = 0.05
    weight_decay: float = 0.01
