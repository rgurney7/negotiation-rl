from dataclasses import dataclass

from shared.config import SharedConfig


@dataclass
class GRPOConfig(SharedConfig):
    run_name: str = "grpo_runs"
    hf_repo_id: str = "ShallowLearning/negotiation-grpo-qwen3.5-4b"
    max_seq_length: int = 2048            # trains on the single closing turn

    # rollout shape
    num_updates: int = 150
    episodes_per_update: int = 8          # scenarios per update
    group_size: int = 8                   # G completions per scenario
    n_epochs: int = 2                     # off-policy reuse per update

    # objective
    clip_eps: float = 0.2
    kl_coef: float = 0.04                 # k3 KL toward the frozen reference
    ent_coef: float = 0.01                # token-mean entropy
    learning_rate: float = 5e-5
    max_grad_norm: float = 1.0
