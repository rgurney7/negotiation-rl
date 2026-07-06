"""Seed-in-path threading and the across-seed aggregator's pure stats (no GPU, no API)."""

import json
import os
import sys
import tempfile

HERE = os.path.dirname(__file__)
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from shared.config import SharedConfig                                   # noqa: E402
from shared import persistence                                          # noqa: E402
from shared.eval_harness import wilcoxon_signed_rank_p, _sign_test_p    # noqa: E402
from aggregate_seeds import (describe, iqm, bootstrap_ci,               # noqa: E402
                             paired_contrast_across_seeds, aggregate)


# --- seed-in-path threading ---

def test_seed_in_path_suffixes_run_and_repo():
    c = SharedConfig(run_name="ppo_runs", hf_repo_id="org/repo", seed=7, seed_in_path=True)
    assert c.run_name == "ppo_runs_s7"
    assert c.output_dir.endswith("ppo_runs_s7")
    assert c.hf_repo_id == "org/repo-s7"


def test_seed_in_path_is_idempotent():
    c = SharedConfig(run_name="ppo_runs", hf_repo_id="org/repo", seed=7, seed_in_path=True)
    c.__post_init__()                       # re-deriving must NOT double-append
    c.__post_init__()
    assert c.run_name == "ppo_runs_s7" and c.hf_repo_id == "org/repo-s7"


def test_seed_in_path_off_is_unchanged():
    c = SharedConfig(run_name="ppo_runs", seed=7, seed_in_path=False)
    assert c.run_name == "ppo_runs" and c.output_dir.endswith("ppo_runs")


def test_distinct_seeds_get_distinct_dirs():
    a = SharedConfig(run_name="ppo_runs", seed=1, seed_in_path=True)
    b = SharedConfig(run_name="ppo_runs", seed=2, seed_in_path=True)
    assert a.output_dir != b.output_dir and a.checkpoint_dir != b.checkpoint_dir


def test_mark_preview_then_seed_no_double_suffix():
    c = SharedConfig(run_name="ppo_runs")
    persistence.mark_preview(c)             # -> ppo_runs_preview, hf_repo_id cleared
    c.seed = 7
    c.seed_in_path = True
    c.__post_init__()
    assert c.run_name == "ppo_runs_preview_s7"


# --- across-seed stats honesty ---

def test_describe_small_n_withholds_ci_iqm():
    d = describe([0.6, 0.5, 0.55])
    assert d["n"] == 3 and "ci95" not in d and "iqm" not in d and "interval" in d
    assert d["mean"] == round((0.6 + 0.5 + 0.55) / 3, 4)
    assert d["min"] == 0.5 and d["max"] == 0.6


def test_describe_large_n_reports_ci_and_iqm():
    d = describe([0.50, 0.52, 0.54, 0.56, 0.58])
    assert d["n"] == 5 and "ci95" in d and "iqm" in d
    lo, hi = d["ci95"]
    assert lo <= d["mean"] <= hi


def test_iqm_trims_outlier_seed():
    # N=5 -> trim floor(0.25*5)=1 from each tail -> mean of the middle three (all 0.5)
    assert abs(iqm([0.0, 0.5, 0.5, 0.5, 10.0]) - 0.5) < 1e-9


def test_bootstrap_ci_deterministic_and_bounded():
    xs = [0.40, 0.50, 0.60, 0.55, 0.45]
    assert bootstrap_ci(xs) == bootstrap_ci(xs)              # fixed RNG -> reproducible
    lo, hi = bootstrap_ci(xs)
    assert min(xs) <= lo <= hi <= max(xs)


def test_paired_contrast_sign_consistency():
    c = paired_contrast_across_seeds([0.1, 0.05, 0.2])
    assert c["all_same_sign"] and c["direction"] == "A>B" and "ci95" not in c   # N=3 -> no CI
    m = paired_contrast_across_seeds([0.1, -0.05, 0.2])
    assert not m["all_same_sign"] and m["direction"] == "A>B"


# --- Wilcoxon (magnitude-sensitive) ---

def test_wilcoxon_all_positive_is_significant():
    assert wilcoxon_signed_rank_p([0.2, 0.3, 0.5, 0.1, 0.4, 0.25, 0.35, 0.15]) < 0.05


def test_wilcoxon_symmetric_is_not_significant():
    assert wilcoxon_signed_rank_p([0.2, -0.2, 0.3, -0.3, 0.1, -0.1]) > 0.5


def test_wilcoxon_no_nonzero_pairs_is_one():
    assert wilcoxon_signed_rank_p([0, 0, 0]) == 1.0


def test_wilcoxon_beats_sign_test_when_margins_are_lopsided():
    # 5 large wins, 3 tiny losses: magnitude-weighting gives a strictly smaller p than the sign test
    diffs = [0.9, 0.8, 0.7, 0.85, 0.95, -0.01, -0.02, -0.015]
    wins = sum(d > 0 for d in diffs)
    losses = sum(d < 0 for d in diffs)
    assert wilcoxon_signed_rank_p(diffs) < _sign_test_p(wins, losses)


# --- aggregate() end-to-end over synthetic per-seed eval files ---

def _write_eval(root, seed, method, per_scenario):
    d = os.path.join(root, f"eval_s{seed}")
    os.makedirs(d, exist_ok=True)
    rewards = [v["reward"] for v in per_scenario.values()]
    deals = [v for v in per_scenario.values() if v["deal"]]
    metrics = {"n": len(per_scenario),
               "mean_reward": round(sum(rewards) / len(rewards), 4),
               "deal_rate": round(len(deals) / len(per_scenario), 4),
               "mean_ratio": None}
    with open(os.path.join(d, f"{method}_eval.json"), "w") as f:
        json.dump({"adapter_dir": None, "metrics": metrics, "per_scenario": per_scenario}, f)


def test_aggregate_end_to_end_small_n():
    def ps(reward, deal):
        return {str(i): {"reward": reward, "agreed_price": (90 if deal else None),
                         "ratio": (0.5 if deal else None), "deal": deal} for i in range(10)}
    with tempfile.TemporaryDirectory() as root:
        for s in (1, 2, 3):
            _write_eval(root, s, "ppo", ps(0.6, True))     # ppo consistently beats sft
            _write_eval(root, s, "sft", ps(0.4, True))
            _write_eval(root, s, "grpo", ps(0.5, True))
            _write_eval(root, s, "base", ps(0.3, False))
        res = aggregate(root, [1, 2, 3], ["sft", "grpo", "ppo"], ("ppo", "sft"))

    assert res["n_seeds"] == 3 and res["ci_eligible"] is False
    assert set(res["by_method"]) == {"sft", "grpo", "ppo", "base"}
    assert res["by_method"]["ppo"]["mean_reward"]["mean"] == 0.6
    assert "ci95" not in res["by_method"]["ppo"]["mean_reward"]          # honest at N=3
    pc = res["primary_contrast"]
    assert pc["pair"] == "ppo_vs_sft" and len(pc["per_seed"]) == 3
    assert pc["across_seed"]["all_same_sign"] and pc["across_seed"]["direction"] == "A>B"
    assert pc["per_seed"][0]["within_seed_paired"]["wilcoxon_p"] is not None


def test_aggregate_recomputes_by_method_on_common_set_under_drop():
    """by_method compares all methods over the shared scenarios after a drop."""
    def ps(reward_by_id):
        return {i: {"reward": r, "agreed_price": 90, "ratio": 0.5, "deal": True}
                for i, r in reward_by_id.items()}
    with tempfile.TemporaryDirectory() as root:
        sft = ps({str(i): (0.4 if i < 9 else -1.0) for i in range(10)})   # id "9" is an outlier
        ppo = ps({str(i): 0.6 for i in range(9)})                         # dropped id "9"
        grpo = ps({str(i): 0.5 for i in range(10)})
        _write_eval(root, 1, "sft", sft)
        _write_eval(root, 1, "ppo", ppo)
        _write_eval(root, 1, "grpo", grpo)
        res = aggregate(root, [1], ["sft", "grpo", "ppo"], ("ppo", "sft"))
    # sft over its full 10 rows would be 0.26; over the 9 SHARED rows it is 0.4.
    assert res["by_method"]["sft"]["mean_reward"]["mean"] == 0.4
    assert res["by_method"]["ppo"]["mean_reward"]["mean"] == 0.6


def test_primary_contrast_uses_common_set_not_own_set_means():
    """The primary Δreward subtracts common-set means, not own-set means (0.2 here, not 0.34)."""
    def ps(reward_by_id):
        return {i: {"reward": r, "agreed_price": 90, "ratio": 0.5, "deal": True}
                for i, r in reward_by_id.items()}
    with tempfile.TemporaryDirectory() as root:
        _write_eval(root, 1, "sft", ps({str(i): (0.4 if i < 9 else -1.0) for i in range(10)}))
        _write_eval(root, 1, "ppo", ps({str(i): 0.6 for i in range(9)}))       # dropped id "9"
        _write_eval(root, 1, "grpo", ps({str(i): 0.5 for i in range(10)}))
        res = aggregate(root, [1], ["sft", "grpo", "ppo"], ("ppo", "sft"))
    assert res["primary_contrast"]["per_seed"][0]["mean_reward_diff"] == 0.2


def test_base_excluded_from_common_set_intersection():
    """base (evaluated on seed 1 only) must not shrink the trained methods' common set."""
    from aggregate_seeds import _per_seed_metrics
    def ps(ids, reward):
        return {i: {"reward": reward, "agreed_price": 90, "ratio": 0.5, "deal": True} for i in ids}
    with tempfile.TemporaryDirectory() as root:
        allids = [str(i) for i in range(10)]
        _write_eval(root, 1, "sft", ps(allids, 0.4))
        _write_eval(root, 1, "ppo", ps(allids, 0.6))
        _write_eval(root, 1, "grpo", ps(allids, 0.5))
        _write_eval(root, 1, "base", ps(allids[1:], 0.3))     # base dropped id "0"
        out, common_n = _per_seed_metrics(root, 1, ["sft", "grpo", "ppo"])
    assert common_n == 10 and out["ppo"]["n"] == 10 and out["sft"]["n"] == 10
    assert out["base"]["n"] == 9                               # base kept only its own coverage


def test_aggregate_records_per_seed_common_n():
    """aggregate.json carries per-seed common_n denominators."""
    def ps(n):
        return {str(i): {"reward": 0.5, "agreed_price": 90, "ratio": 0.5, "deal": True}
                for i in range(n)}
    with tempfile.TemporaryDirectory() as root:
        for s in (1, 2):
            _write_eval(root, s, "ppo", ps(10))
            _write_eval(root, s, "sft", ps(10))
        res = aggregate(root, [1, 2], ["sft", "ppo"], ("ppo", "sft"))
    assert res["per_seed_common_n"] == {1: 10, 2: 10}
    assert all(row["common_n"] == 10 for row in res["primary_contrast"]["per_seed"])


def test_aggregate_empty_when_no_eval_files():
    """No eval JSONs yields empty by_method and an empty primary contrast."""
    with tempfile.TemporaryDirectory() as root:
        res = aggregate(root, [1, 2, 3], ["sft", "grpo", "ppo"], ("ppo", "sft"))
    assert res["by_method"] == {} and res["primary_contrast"]["per_seed"] == []


def _run_all():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print(f"  ok  {fn.__name__}")
    print(f"PASS  {len(fns)} seed-path/aggregator tests")


if __name__ == "__main__":
    _run_all()
