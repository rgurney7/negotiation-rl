"""The runtime renderer must reproduce the committed data slices exactly; parse_seed must round-trip."""

import json
import os
import sys

HERE = os.path.dirname(__file__)
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from shared import render  # noqa: E402

SLICES = os.path.join(ROOT, "data", "slices")
DATA = os.path.join(ROOT, "data")


def _rows(path, n=None):
    with open(path) as f:
        rows = [json.loads(line) for line in f if line.strip()]
    return rows[:n] if n else rows


def test_seed_roundtrip_ppo():
    for r in _rows(os.path.join(SLICES, "ppo.jsonl")):
        assert render.render_transcript(render.parse_seed(r["user"])) == r["user"], r["did"]


def test_seed_roundtrip_eval():
    """Eval seeds follow the PPO seed convention."""
    for r in _rows(os.path.join(DATA, "eval_pool.jsonl")):
        assert render.render_transcript(render.parse_seed(r["first_buyer_msg"])) == r["first_buyer_msg"], r["sid"]


def test_seed_roundtrip_grpo():
    for r in _rows(os.path.join(SLICES, "grpo.jsonl")):
        assert render.render_transcript(render.parse_seed(r["user"])) == r["user"], r["did"]


def test_sft_user_roundtrip():
    """Same renderer, all cut points."""
    for r in _rows(os.path.join(SLICES, "sft.jsonl")):
        assert render.render_transcript(render.parse_seed(r["user"])) == r["user"], (r["did"], r["turn_index"])


def test_header_and_marker_match_slices():
    r = _rows(os.path.join(SLICES, "ppo.jsonl"), 1)[0]
    assert r["user"].startswith(render.TRANSCRIPT_HEADER + "\n")
    assert r["user"].rstrip().endswith(render.TURN_MARKER)


def test_seller_prompt_matches_slice():
    for r in _rows(os.path.join(DATA, "eval_pool.jsonl"), 200):
        rebuilt = render.seller_prompt(r["listing"], r["title"], r["description"])
        assert rebuilt == r["seller_prompt"], r["sid"]


def _run_all():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print(f"  ok  {fn.__name__}")
    print(f"PASS  {len(fns)} render-parity tests")


if __name__ == "__main__":
    _run_all()
