from dataclasses import dataclass

from shared.config import SharedConfig


@dataclass
class PPOConfig(SharedConfig):
    run_name: str = "ppo_runs"
    hf_repo_id: str = "ShallowLearning/negotiation-ppo-qwen3.5-4b"
    max_seq_length: int = 4096

    # rollout shape
    num_updates: int = 150
    num_parallel_episodes: int = 8        # episodes run in lockstep

    # PPO objective
    ppo_epochs: int = 3
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_eps: float = 0.2
    value_coef: float = 0.5
    entropy_coef: float = 0.003           # token-level
    max_grad_norm: float = 0.5
    normalize_advantages: bool = True
    truncate_after_deal: bool = True      # don't train on turns after the judge's deal turn
    # no KL penalty; clipping only

    # optimizer (dual LR)
    policy_lr: float = 1e-5
    critic_lr: float = 3e-4

    # periodic greedy eval on a fixed val subset (disjoint from test); 0 disables
    checkpoint_eval_every: int = 30
    checkpoint_eval_n: int = 25
