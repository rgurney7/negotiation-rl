"""truncate_at_close: drop post-deal steps, move the terminal reward to the close step."""

import os
import sys
import types

HERE = os.path.dirname(__file__)
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from ppo.rollout import truncate_at_close, drop_failed_episodes  # noqa: E402


def _steps(rewards):
    """Step dicts with only the fields truncate reads."""
    return [{"reward": float(r), "value": 0.0, "resp_len": 1} for r in rewards]


def _cfg(on=True):
    return types.SimpleNamespace(truncate_after_deal=on)


def test_truncates_and_moves_reward():
    # deal sealed at turn 3; reward landed at the last turn pre-truncation
    eps = [_steps([0, 0, 0, 0, 0, 0, 0, 0.8])]
    infos = [{"agreed_price": 120.0, "close_step": 3, "reward": 0.8}]
    out = truncate_at_close(eps, infos, _cfg())
    assert len(out[0]) == 3                                  # turns 4-8 dropped
    assert out[0][-1]["reward"] == 0.8                       # reward moved onto the close step
    assert all(s["reward"] == 0.0 for s in out[0][:-1])      # earlier turns stay zero


def test_no_deal_keeps_full_horizon():
    eps = [_steps([0] * 8)]
    infos = [{"agreed_price": None, "close_step": None, "reward": 0.0}]
    out = truncate_at_close(eps, infos, _cfg())
    assert len(out[0]) == 8


def test_close_at_last_turn_keeps_all():
    eps = [_steps([0] * 7 + [0.5])]
    infos = [{"agreed_price": 90.0, "close_step": 8, "reward": 0.5}]
    out = truncate_at_close(eps, infos, _cfg())
    assert len(out[0]) == 8 and out[0][-1]["reward"] == 0.5


def test_clamps_out_of_range_close():
    # an out-of-range close index clamps rather than raising
    eps = [_steps([0, 0, 0, 0.5])]
    infos = [{"agreed_price": 90.0, "close_step": 99, "reward": 0.5}]
    out = truncate_at_close(eps, infos, _cfg())
    assert len(out[0]) == 4 and out[0][-1]["reward"] == 0.5


def test_flag_off_is_noop():
    eps = [_steps([0] * 7 + [0.8])]
    infos = [{"agreed_price": 120.0, "close_step": 3, "reward": 0.8}]
    out = truncate_at_close(eps, infos, _cfg(on=False))
    assert len(out[0]) == 8 and out[0][-1]["reward"] == 0.8


def test_mixed_batch():
    eps = [_steps([0, 0, 0, 0.7]), _steps([0] * 4), _steps([0, 0, 0, 0.9])]
    infos = [
        {"agreed_price": 80.0, "close_step": 2, "reward": 0.7},   # deal at turn 2
        {"agreed_price": None, "close_step": None, "reward": 0.0}, # no deal
        {"agreed_price": 95.0, "close_step": 4, "reward": 0.9},    # deal at last turn
    ]
    out = truncate_at_close(eps, infos, _cfg())
    assert [len(e) for e in out] == [2, 4, 4]
    assert out[0][-1]["reward"] == 0.7 and out[2][-1]["reward"] == 0.9


def test_drop_failed_episodes_empties_only_failed():
    eps = [_steps([0, 0, 0.7]), _steps([0, 0, 0, 0.5]), _steps([0, 0.9])]
    infos = [{"agreed_price": 80.0},
             {"buyer_failed": True, "agreed_price": None},   # buyer API outage -> drop
             {"agreed_price": 95.0}]
    out = drop_failed_episodes(eps, infos)
    assert [len(e) for e in out] == [3, 0, 2]               # middle episode emptied
    assert out[0] is eps[0] and out[2] is eps[2]            # survivors untouched


def test_drop_failed_then_truncate_skips_dropped():
    # truncate_at_close leaves an emptied episode as []
    eps = [_steps([0, 0, 0, 0.8]), _steps([0, 0, 0, 0.0])]
    infos = [{"agreed_price": 120.0, "close_step": 2, "reward": 0.8},
             {"buyer_failed": True, "agreed_price": None, "close_step": None}]
    eps = drop_failed_episodes(eps, infos)
    out = truncate_at_close(eps, infos, _cfg())
    assert [len(e) for e in out] == [2, 0]
    assert out[0][-1]["reward"] == 0.8


def test_drop_failed_all_clear_is_noop():
    eps = [_steps([0, 0.7]), _steps([0, 0.9])]
    infos = [{"agreed_price": 80.0}, {"agreed_price": 95.0}]
    out = drop_failed_episodes(eps, infos)
    assert [len(e) for e in out] == [2, 2]


def test_drop_failed_episodes_drops_judge_failed_too():
    # a judge outage is a drop, not a real no-deal
    eps = [_steps([0, 0, 0.7]), _steps([0, 0, 0.0]), _steps([0, 0.9])]
    infos = [{"agreed_price": 80.0},
             {"judge_failed": True, "agreed_price": None},   # both judges down -> drop
             {"agreed_price": 95.0}]
    out = drop_failed_episodes(eps, infos)
    assert [len(e) for e in out] == [3, 0, 2]               # middle episode emptied


def _run_all():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print(f"  ok  {fn.__name__}")
    print(f"PASS  {len(fns)} ppo-truncate tests")


if __name__ == "__main__":
    _run_all()
