"""Reward shape + deterministic price extractor."""

import os
import sys

HERE = os.path.dirname(__file__)
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from shared.reward import price_reward, extract_agreed_price  # noqa: E402


def test_price_reward_shape():
    # reserve=50, ceiling=100
    assert price_reward(None, 50, 100) == 0.0           # no deal
    assert price_reward(40, 50, 100) == -1.0            # below reserve
    assert price_reward(50, 50, 100) == 0.0             # at reserve
    assert price_reward(75, 50, 100) == 0.5             # midpoint
    assert price_reward(100, 50, 100) == 1.0            # at ceiling
    assert price_reward(120, 50, 100) == 1.0            # clipped above ceiling


def test_extract_basic_accept_with_number():
    turns = [("buyer", "Can you do $80?"), ("seller", "I can do 90."), ("buyer", "Deal, 90 it is.")]
    assert extract_agreed_price(turns, listing=100) == 90.0


def test_extract_accept_no_number_falls_back_to_last_offer():
    # Buyer accepts with no number -> use the last in-band price on the table (seller's 85).
    turns = [("buyer", "How about $70?"), ("seller", "I'll take 85."), ("buyer", "Okay, sounds good.")]
    assert extract_agreed_price(turns, listing=100) == 85.0


def test_extract_no_acceptance_is_no_deal():
    turns = [("buyer", "Would you take $60?"), ("seller", "No, lowest is 90."), ("buyer", "Too much for me.")]
    assert extract_agreed_price(turns, listing=100) is None


def test_extract_k_suffix():
    turns = [("buyer", "I can give you 13k."), ("seller", "Deal.")]
    assert extract_agreed_price(turns, listing=15000) == 13000.0


def test_extract_out_of_band_price_ignored():
    # "5" is below the 0.1*listing floor; deal with no in-band number falls back to buyer_target
    turns = [("buyer", "is the model number 5 the right one?"), ("seller", "yes"), ("buyer", "ok deal")]
    assert extract_agreed_price(turns, listing=100, buyer_target=72) == 72.0


def test_extract_accept_no_number_no_offer_uses_target():
    turns = [("seller", "Want to buy it?"), ("buyer", "Sure, it's a deal.")]
    assert extract_agreed_price(turns, listing=100, buyer_target=72) == 72.0


def _run_all():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print(f"  ok  {fn.__name__}")
    print(f"PASS  {len(fns)} reward tests")


if __name__ == "__main__":
    _run_all()
