"""Eval comparison driver tests with a stub evaluate (no GPU, no API)."""

import json
import os
import sys
import tempfile

HERE = os.path.dirname(__file__)
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from shared.config import SharedConfig        # noqa: E402
from shared import persistence, eval_harness  # noqa: E402
import eval_methods                           # noqa: E402


def _per(rewards):
    out = {}
    for i, r in enumerate(rewards):
        deal = r > 0
        out[f"s{i}"] = {"reward": r, "deal": deal,
                        "ratio": 0.9 if deal else None,
                        "agreed_price": 90.0 if deal else None}
    return out


def test_write_json_atomic_roundtrips():
    p = os.path.join(tempfile.mkdtemp(), "sub", "x.json")
    persistence.write_json_atomic(p, {"a": 1, "b": [1, 2]})
    assert json.load(open(p)) == {"a": 1, "b": [1, 2]}
    assert not os.path.exists(p + ".tmp")          # temp file cleaned up


def test_compare_methods_matrix():
    results = {"a": _per([1.0, 1.0, -1.0]), "b": _per([-1.0, -1.0, 1.0])}
    comp = eval_harness.compare_methods(results)
    assert set(comp["methods"]) == {"a", "b"}
    assert comp["metrics"]["a"]["n"] == 3
    assert comp["metrics"]["a"]["deal_rate"] == round(2 / 3, 4)
    assert comp["win_rate"]["a_vs_b"] == round(2 / 3, 4)   # a wins 2 of 3
    assert comp["win_rate"]["b_vs_a"] == round(1 / 3, 4)


def test_compare_methods_common_set_under_differential_drop():
    """metrics_common re-aggregates every method over only the shared scenarios."""
    a = _per([1.0, 1.0, 1.0]); del a["s0"]         # a dropped s0
    b = _per([1.0, 1.0, 1.0]); del b["s2"]         # b dropped s2
    comp = eval_harness.compare_methods({"a": a, "b": b})
    assert comp["metrics"]["a"]["n"] == 2 and comp["metrics"]["b"]["n"] == 2   # own sets differ
    assert comp["common_n"] == 1                    # only s1 is shared
    assert comp["metrics_common"]["a"]["n"] == 1 and comp["metrics_common"]["b"]["n"] == 1


def test_paired_stats_sign_test():
    a, b = _per([1.0, 1.0, 1.0]), _per([-1.0, -1.0, -1.0])
    st = eval_harness.paired_stats(a, b)
    assert (st["n"], st["wins"], st["losses"], st["ties"]) == (3, 3, 0, 0)
    assert st["win_rate"] == 1.0
    assert st["p_value"] == round(2 * 0.5 ** 3, 4)         # exact two-sided sign test, k=0
    tied = eval_harness.paired_stats(a, a)                  # all ties
    assert tied["win_rate"] == 0.5 and tied["ties"] == 3 and tied["p_value"] == 1.0


def test_run_orchestration_with_stub_evaluate():
    cfg = SharedConfig(run_name="eval_test")
    out = tempfile.mkdtemp()
    canned = {
        None:               ({"n": 3, "deal_rate": 0.0}, _per([-1.0, -1.0, -1.0])),
        "/x/sft/lora_final": ({"n": 3}, _per([1.0, 1.0, -1.0])),
        "/x/ppo/lora_final": ({"n": 3}, _per([1.0, 1.0, 1.0])),
    }

    def stub_eval(c, scenarios, buyer, adapter_dir=None, judge=None):
        return canned[adapter_dir]

    dirs = {"sft": "/x/sft/lora_final", "ppo": "/x/ppo/lora_final"}
    comp = eval_methods.run(cfg, dirs, scenarios=[1, 2, 3], buyer=None, out_dir=out,
                            judge=None, include_base=True, evaluate=stub_eval)

    for name in ["base_eval.json", "sft_eval.json", "ppo_eval.json", "comparison.json"]:
        assert os.path.exists(os.path.join(out, name)), name
    saved = json.load(open(os.path.join(out, "comparison.json")))
    assert set(saved["methods"]) == {"base", "sft", "ppo"}
    assert saved["win_rate"]["ppo_vs_base"] == 1.0        # ppo wins all 3 vs base
    assert comp["metrics"]["ppo"]["deal_rate"] == 1.0
    assert comp["metrics"]["base"]["deal_rate"] == 0.0


def test_run_skips_already_scored_method_on_resume():
    """A completed {method}_eval.json is reused on restart, not re-scored."""
    cfg = SharedConfig(run_name="eval_resume_test")
    out = tempfile.mkdtemp()
    persistence.write_json_atomic(
        os.path.join(out, "sft_eval.json"),
        {"adapter_dir": "/x/sft/lora_final", "metrics": {"n": 3}, "per_scenario": _per([1.0, 1.0, -1.0]),
         "n_requested": 3, "split": None, "limit": None,
         "scenario_sig": eval_methods._scenario_sig([1, 2, 3])})
    called = []

    def stub_eval(c, scenarios, buyer, adapter_dir=None, judge=None):
        called.append(adapter_dir)
        return ({"n": 3}, _per([1.0, 1.0, 1.0]))

    dirs = {"sft": "/x/sft/lora_final", "ppo": "/x/ppo/lora_final"}
    comp = eval_methods.run(cfg, dirs, scenarios=[1, 2, 3], buyer=None, out_dir=out, judge=None,
                            include_base=False, evaluate=stub_eval)
    assert "/x/sft/lora_final" not in called       # sft reused from disk, not re-evaluated
    assert "/x/ppo/lora_final" in called           # ppo (no prior file) still evaluated
    assert set(comp["methods"]) == {"sft", "ppo"}  # both present in the comparison


def test_resume_rejects_cached_eval_of_different_request_shape():
    """A cached eval is reused only for the same (split, limit, n) request."""
    cfg = SharedConfig(run_name="eval_shape_test")
    out = tempfile.mkdtemp()
    persistence.write_json_atomic(
        os.path.join(out, "sft_eval.json"),
        {"adapter_dir": "/x/sft/lora_final", "metrics": {"n": 3}, "per_scenario": _per([1.0] * 3),
         "n_requested": 3, "split": "test", "limit": 150})
    called = []

    def stub_eval(c, scenarios, buyer, adapter_dir=None, judge=None):
        called.append(adapter_dir)
        return ({"n": 10}, _per([1.0] * 10))

    eval_methods.run(cfg, {"sft": "/x/sft/lora_final"}, scenarios=list(range(10)), buyer=None,
                     out_dir=out, judge=None, include_base=False, evaluate=stub_eval,
                     split="test", limit=None)
    assert "/x/sft/lora_final" in called            # different shape -> re-scored, not reused
    saved = json.load(open(os.path.join(out, "sft_eval.json")))
    assert saved["n_requested"] == 10 and saved["limit"] is None   # file now records the new shape


def test_resume_rejects_cached_eval_of_different_scenario_set():
    """Same shape but a different scenario set (by scenario_sig) forces a re-score."""
    cfg = SharedConfig(run_name="eval_sig_test")
    out = tempfile.mkdtemp()
    persistence.write_json_atomic(
        os.path.join(out, "sft_eval.json"),
        {"adapter_dir": "/x/sft/lora_final", "metrics": {"n": 3}, "per_scenario": _per([1.0] * 3),
         "n_requested": 3, "split": "test", "limit": None,
         "scenario_sig": eval_methods._scenario_sig([7, 8, 9])})   # a different set of 3
    called = []

    def stub_eval(c, scenarios, buyer, adapter_dir=None, judge=None):
        called.append(adapter_dir)
        return ({"n": 3}, _per([1.0] * 3))

    eval_methods.run(cfg, {"sft": "/x/sft/lora_final"}, scenarios=[1, 2, 3], buyer=None, out_dir=out,
                     judge=None, include_base=False, evaluate=stub_eval, split="test", limit=None)
    assert "/x/sft/lora_final" in called            # same shape, different set -> re-scored
    saved = json.load(open(os.path.join(out, "sft_eval.json")))
    assert saved["scenario_sig"] == eval_methods._scenario_sig([1, 2, 3])   # rewritten to the real set


def test_compare_methods_base_drop_does_not_shrink_common_set():
    """A base-only drop must not shrink the trained methods' common set."""
    a, b = _per([1.0, 1.0, 1.0]), _per([1.0, 1.0, -1.0])
    base = _per([0.5, 0.5, 0.5]); del base["s1"]        # base-only drop
    comp = eval_harness.compare_methods({"base": base, "a": a, "b": b})
    assert comp["common_n"] == 3                        # trained methods keep all 3
    assert comp["metrics_common"]["a"]["n"] == 3
    assert comp["metrics_common"]["base"]["n"] == 2     # common intersected with base coverage


def test_drop_cold_opens_excludes_seedless_scenarios():
    """Rows with no buyer opener in the seed are filtered out."""
    warm = {"id": "w", "seed": "Negotiation Transcript:\n[Buyer]: hi, 50?\n\n[Your Turn]:"}
    cold = {"id": "c", "seed": "Negotiation Transcript:\n\n[Your Turn]:"}
    empty = {"id": "e", "seed": ""}
    got = eval_methods.drop_cold_opens([warm, cold, empty])
    assert [s["id"] for s in got] == ["w"]


def test_sample_distinct_listings_one_row_per_listing_deterministic():
    """--limit N yields N scenarios over N distinct listings, deterministically."""
    scenarios = [{"sid": f"s{l}_{r}", "title": f"item{l}", "listing": 100.0 + l,
                  "description": f"desc{l}"} for l in range(10) for r in range(3)]
    got = eval_methods.sample_distinct_listings(scenarios, 5)
    assert len(got) == 5
    idents = {(s["title"], s["listing"], s["description"]) for s in got}
    assert len(idents) == 5                                     # one row per distinct listing
    again = eval_methods.sample_distinct_listings(list(scenarios), 5)
    assert [s["sid"] for s in got] == [s["sid"] for s in again]  # deterministic
    order = [s["sid"] for s in scenarios]
    assert sorted(order.index(s["sid"]) for s in got) == [order.index(s["sid"]) for s in got]
    assert eval_methods.sample_distinct_listings(scenarios, 999) == scenarios   # no-op past pool size
    topped = eval_methods.sample_distinct_listings(scenarios, 12)   # 12 > 10 distinct -> top up
    assert len(topped) == 12
    assert len({(s["title"], s["listing"], s["description"]) for s in topped}) == 10


def test_run_aborts_below_coverage_floor_without_persisting():
    """A gutted eval is never written: the file's existence means 'complete' to the resume path."""
    cfg = SharedConfig(run_name="eval_floor_test")
    out = tempfile.mkdtemp()

    def stub_eval(c, scenarios, buyer, adapter_dir=None, judge=None):
        if adapter_dir == "/x/ppo/lora_final":
            return ({"n": 1}, _per([1.0]))                  # 1 of 10 scored = outage, not a result
        return ({"n": 10}, _per([1.0] * 10))

    dirs = {"sft": "/x/sft/lora_final", "ppo": "/x/ppo/lora_final"}
    try:
        eval_methods.run(cfg, dirs, scenarios=list(range(10)), buyer=None, out_dir=out,
                         judge=None, include_base=False, evaluate=stub_eval)
        raise AssertionError("expected SystemExit")
    except SystemExit as e:
        assert e.code == 6
    assert os.path.exists(os.path.join(out, "sft_eval.json"))       # healthy method persisted
    assert not os.path.exists(os.path.join(out, "ppo_eval.json"))   # gutted eval NOT persisted
    saved = json.load(open(os.path.join(out, "sft_eval.json")))
    assert saved["n_requested"] == 10


def test_pod_eval_root_matches_writer_and_reader():
    """On a pod, the eval writer and the aggregator reader must resolve the same /workspace dir."""
    import os as _os
    from shared import config as _config
    import aggregate_seeds
    real_isdir = _os.path.isdir
    _os.path.isdir = lambda p: True if p == "/workspace" else real_isdir(p)
    try:
        writer = SharedConfig(run_name="eval", seed=1, seed_in_path=True).output_dir
        reader = aggregate_seeds._eval_dir(_config.runs_base(), 1)
        assert writer == reader == os.path.join("/workspace", "eval_s1")
    finally:
        _os.path.isdir = real_isdir


def _run_all():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print(f"  ok  {fn.__name__}")
    print(f"PASS  {len(fns)} eval-driver tests")


if __name__ == "__main__":
    _run_all()
