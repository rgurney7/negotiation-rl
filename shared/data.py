"""Slice loaders; normalize slices into one scenario dict shape."""

import json
import os


def _load_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def _assert_reserve(rows, reserve_fraction):
    for r in rows:
        if "reserve" in r and "listing" in r:
            assert abs(r["reserve"] - reserve_fraction * r["listing"]) < 1e-6, (
                f"reserve {r['reserve']} != {reserve_fraction}*{r['listing']} in {r.get('did') or r.get('sid')}")


def _scenario(*, sid, system, seed, listing, reserve, buyer_target, title, description, **extra):
    s = {"id": sid, "system": system, "seed": seed, "listing": float(listing),
         "reserve": float(reserve), "buyer_target": float(buyer_target),
         "title": title, "description": description}
    s.update(extra)
    return s


def load_ppo_scenarios(cfg):
    """ppo.jsonl -> full self-play scenarios (seed = recorded buyer opener)."""
    rows = _load_jsonl(os.path.join(cfg.slices_dir, "ppo.jsonl"))
    _assert_reserve(rows, cfg.reserve_fraction)
    return [_scenario(sid=r["did"], system=r["system"], seed=r["user"], listing=r["listing"],
                      reserve=r["reserve"], buyer_target=r["buyer_target"], title=r["title"],
                      description=r["description"], seed_through_index=r.get("seed_through_index"))
            for r in rows]


def load_grpo_examples(cfg):
    """grpo.jsonl -> single-closing-turn scenarios. reference_close is log-only, never a target."""
    rows = _load_jsonl(os.path.join(cfg.slices_dir, "grpo.jsonl"))
    _assert_reserve(rows, cfg.reserve_fraction)
    return [_scenario(sid=r["did"], system=r["system"], seed=r["user"], listing=r["listing"],
                      reserve=r["reserve"], buyer_target=r["buyer_target"], title=r["title"],
                      description=r["description"], close_type=r.get("close_type"),
                      reference_close=r.get("reference_close"))
            for r in rows]


def load_eval_pool(cfg, split=None):
    """eval_pool.jsonl -> shared multi-turn eval scenarios; split filters 'validation'/'test'."""
    rows = _load_jsonl(os.path.join(cfg.data_dir, "eval_pool.jsonl"))
    if split:
        rows = [r for r in rows if r["split"] == split]
    _assert_reserve(rows, cfg.reserve_fraction)
    return [_scenario(sid=r["sid"], system=r["seller_prompt"], seed=r["first_buyer_msg"],
                      listing=r["listing"], reserve=r["reserve"], buyer_target=r["buyer_target"],
                      title=r["title"], description=r["description"], split=r["split"],
                      category=r.get("category"))
            for r in rows]


def load_val50(cfg):
    """val50.jsonl -> the 50-scenario dev subset for checkpoint/HP selection."""
    rows = _load_jsonl(os.path.join(cfg.slices_dir, "val50.jsonl"))
    _assert_reserve(rows, cfg.reserve_fraction)
    return [_scenario(sid=r["sid"], system=r["seller_prompt"], seed=r["first_buyer_msg"],
                      listing=r["listing"], reserve=r["reserve"], buyer_target=r["buyer_target"],
                      title=r["title"], description=r["description"], split=r["split"])
            for r in rows]


def load_sft_examples(cfg):
    """sft.jsonl -> per-turn chat examples ({system, user, assistant})."""
    rows = _load_jsonl(os.path.join(cfg.slices_dir, "sft.jsonl"))
    examples = [{"messages": [
        {"role": "system", "content": r["system"]},
        {"role": "user", "content": r["user"]},
        {"role": "assistant", "content": r["assistant"]},
    ]} for r in rows]
    assert examples, "sft.jsonl produced 0 examples"
    return examples
