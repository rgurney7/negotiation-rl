"""GRPO objective: batch/per-completion equivalence, k3 KL, masking, advantages (needs torch)."""

import os
import sys

import torch

HERE = os.path.dirname(__file__)
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from grpo.loss import (group_advantages, completion_mask, masked_mean,  # noqa: E402
                       grpo_loss, grpo_loss_batch, group_keep_indices)


def test_group_advantages_flat_group_is_zero():
    adv = group_advantages([0.5, 0.5, 0.5])
    assert torch.allclose(adv, torch.zeros(3))


def test_group_keep_indices_all_pass():
    assert group_keep_indices([False, False, False, False]) == [0, 1, 2, 3]


def test_group_keep_indices_drops_failed():
    assert group_keep_indices([False, True, False, True]) == [0, 2]


def test_group_keep_indices_skips_group_below_min_survivors():
    assert group_keep_indices([True, True, True, False]) is None      # 1 survivor < 2 -> skip
    assert group_keep_indices([True, True, True, True]) is None       # 0 survivors -> skip
    assert group_keep_indices([False, False, True, True]) == [0, 1]   # exactly 2 -> keep


def test_group_advantages_standardized():
    adv = group_advantages([0.0, 1.0])
    assert abs(adv.mean().item()) < 1e-6
    assert adv[1] > adv[0]


def test_completion_mask_drops_after_first_eos():
    eos = 7
    gen = torch.tensor([[1, 2, 7, 7, 3]])     # first eos at index 2; rest is padding
    m = completion_mask(gen, eos)
    assert m.tolist() == [[True, True, True, False, False]]


def test_masked_mean_ignores_padding():
    vals = torch.tensor([[1.0, 2.0, 999.0]])
    mask = torch.tensor([[1, 1, 0]])
    assert abs(masked_mean(vals, mask).item() - 1.5) < 1e-6


def test_k3_kl_nonnegative():
    torch.manual_seed(0)
    new = torch.randn(4, 5)
    ref = torch.randn(4, 5)
    mask = torch.ones(4, 5)
    adv = torch.zeros(4)
    _loss, _pl, kl, _e, _c = grpo_loss_batch(new, new.clone(), ref, torch.ones(4, 5), adv, mask)
    assert (kl >= -1e-6).all()        # batched KL is a per-completion [G] vector


def test_batch_matches_per_completion():
    torch.manual_seed(1)
    G, T = 3, 6
    new = torch.randn(G, T)
    old = torch.randn(G, T)
    ref = torch.randn(G, T)
    ent = torch.rand(G, T)
    adv = torch.tensor([-0.5, 0.0, 1.2])
    mask = torch.ones(G, T)

    bl, bpl, bkl, bent, bclip = grpo_loss_batch(new, old, ref, ent, adv, mask)
    # batched returns per-completion vectors; compare each to the scalar reference
    for i in range(G):
        l, pl, kl, e, c = grpo_loss(new[i], old[i], ref[i], ent[i], adv[i])
        assert torch.allclose(bl[i], l, atol=1e-5)
        assert torch.allclose(bpl[i], pl, atol=1e-5)
        assert torch.allclose(bkl[i], kl, atol=1e-5)
        assert torch.allclose(bent[i], e, atol=1e-5)
        assert torch.allclose(bclip[i], c, atol=1e-5)


def test_batch_ignores_masked_padding():
    # padded positions must not affect any output, even when their values are inf
    torch.manual_seed(2)
    G, T = 2, 5
    new = torch.randn(G, T)
    old = torch.randn(G, T)
    ref = torch.randn(G, T)
    ent = torch.rand(G, T)
    adv = torch.tensor([0.7, -0.3])
    mask = torch.tensor([[1, 1, 1, 0, 0], [1, 1, 1, 1, 0]])
    clean = grpo_loss_batch(new, old, ref, ent, adv, mask)

    new2, old2, ref2, ent2 = (t.clone() for t in (new, old, ref, ent))
    for t in (new2, old2, ref2, ent2):           # corrupt only the masked-out tail of each row
        t[0, 3:] = float("inf")
        t[1, 4:] = float("inf")
    dirty = grpo_loss_batch(new2, old2, ref2, ent2, adv, mask)

    for a, b in zip(clean, dirty):
        assert torch.isfinite(b).all(), "padding leaked a non-finite value into the loss"
        assert torch.allclose(a, b, atol=1e-5), "padding changed the loss"


def _run_all():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn(); print(f"  ok  {fn.__name__}")
    print(f"PASS  {len(fns)} grpo-loss tests")


if __name__ == "__main__":
    _run_all()
