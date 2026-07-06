"""Token-level GRPO objective + k3 KL."""

import torch


def group_advantages(rewards):
    # threshold on std (not std+eps): a near-flat group is noise, not signal
    rewards = torch.as_tensor(rewards, dtype=torch.float32)
    if rewards.numel() <= 1:
        return torch.zeros_like(rewards)
    mean, std = rewards.mean(), rewards.std()
    if float(std) < 1e-4:
        return torch.zeros_like(rewards)
    return (rewards - mean) / std


def group_keep_indices(failed, min_survivors=2):
    """Indices of scorable completions, or None if fewer than min_survivors (skip the scenario)."""
    keep = [i for i, f in enumerate(failed) if not f]
    return keep if len(keep) >= min_survivors else None


def completion_mask(generated_ids, eos_token_id):
    # Real tokens (1) vs right-padding (0), [G, T]. Keeps through the first EOS, correct when pad==eos.
    is_eos = (generated_ids == eos_token_id).long()
    prior_eos = is_eos.cumsum(dim=1) - is_eos
    return prior_eos == 0


def masked_mean(values, mask, dim=-1):
    # Zero masked positions first so a pad inf/NaN can't leak in via 0*inf.
    bool_mask = mask.bool()
    values = torch.where(bool_mask, values, torch.zeros_like(values))
    return values.sum(dim) / bool_mask.to(values.dtype).sum(dim).clamp(min=1.0)


def grpo_loss(new_logps, old_logps, ref_logps, entropy, advantage,
              clip_eps=0.2, kl_coef=0.04, ent_coef=0.01):
    # Per-completion reference. Returns (loss, policy_loss, kl, mean_entropy, clip_frac).
    ratio = torch.exp((new_logps - old_logps).clamp(-20, 20))
    clipped = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps)
    policy_loss = -torch.min(ratio * advantage, clipped * advantage).mean()

    log_ratio = (ref_logps - new_logps).clamp(-20, 20)      # pi_ref / pi_theta
    kl = (torch.exp(log_ratio) - log_ratio - 1).mean()      # k3, always >= 0

    mean_entropy = entropy.mean()
    clip_frac = ((ratio < 1 - clip_eps) | (ratio > 1 + clip_eps)).float().mean()
    loss = policy_loss + kl_coef * kl - ent_coef * mean_entropy
    return loss, policy_loss, kl, mean_entropy, clip_frac


def grpo_loss_batch(new_logps, old_logps, ref_logps, entropy, advantages, mask,
                    clip_eps=0.2, kl_coef=0.04, ent_coef=0.01):
    # Whole group at once: [G, T] padded tokens, advantages [G], mask marks real tokens.
    # Returns per-completion [G] vectors; matches grpo_loss when the mask is all ones.
    adv = advantages.unsqueeze(1)
    ratio = torch.exp((new_logps - old_logps).clamp(-20, 20))
    clipped = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps)
    policy_loss = -masked_mean(torch.min(ratio * adv, clipped * adv), mask)

    log_ratio = (ref_logps - new_logps).clamp(-20, 20)
    kl = masked_mean(torch.exp(log_ratio) - log_ratio - 1, mask)

    mean_entropy = masked_mean(entropy, mask)
    clip_frac = masked_mean(((ratio < 1 - clip_eps) | (ratio > 1 + clip_eps)).float(), mask)
    loss = policy_loss + kl_coef * kl - ent_coef * mean_entropy
    return loss, policy_loss, kl, mean_entropy, clip_frac
