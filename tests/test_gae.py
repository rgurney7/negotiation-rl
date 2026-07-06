"""GAE over one episode with sparse terminal reward (needs torch)."""

import os
import sys

import torch

HERE = os.path.dirname(__file__)
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from ppo.gae import compute_gae  # noqa: E402


def test_returns_equal_adv_plus_values():
    rewards = torch.tensor([0.0, 0.0, 1.0])
    values = torch.tensor([0.3, 0.5, 0.7])
    adv, ret = compute_gae(rewards, values, gamma=0.99, gae_lambda=0.95)
    assert torch.allclose(ret, adv + values)


def test_single_step_terminal():
    # One-turn episode: last turn is terminal (bootstrap 0), so adv = reward - value.
    rewards = torch.tensor([0.8])
    values = torch.tensor([0.5])
    adv, ret = compute_gae(rewards, values, 0.99, 0.95)
    assert abs(adv[0].item() - (0.8 - 0.5)) < 1e-6
    assert abs(ret[0].item() - 0.8) < 1e-6


def test_terminal_reward_propagates_backwards():
    # Zero reward until the end; positive terminal reward -> positive advantage at the last turn.
    rewards = torch.tensor([0.0, 0.0, 0.0, 1.0])
    values = torch.zeros(4)
    adv, _ = compute_gae(rewards, values, gamma=1.0, gae_lambda=1.0)
    # With gamma=lambda=1 and zero values, every turn's advantage == the terminal reward.
    assert torch.allclose(adv, torch.ones(4))


def _run_all():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn(); print(f"  ok  {fn.__name__}")
    print(f"PASS  {len(fns)} gae tests")


if __name__ == "__main__":
    _run_all()
