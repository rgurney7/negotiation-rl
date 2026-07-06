import torch


def compute_gae(rewards, values, gamma, gae_lambda):
    """rewards, values: 1-D tensors of length T. Returns (advantages, returns)."""
    T = len(rewards)
    advantages = torch.zeros_like(rewards)
    last_gae = 0.0
    for t in reversed(range(T)):
        nonterminal = 1.0 if t < T - 1 else 0.0
        next_value = values[t + 1] if t < T - 1 else 0.0
        delta = rewards[t] + gamma * next_value * nonterminal - values[t]
        last_gae = delta + gamma * gae_lambda * nonterminal * last_gae
        advantages[t] = last_gae
    returns = advantages + values
    return advantages, returns
