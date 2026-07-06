"""run_final's pure orchestration logic (no GPU, no API, no subprocess)."""

import os
import sys
import tempfile

HERE = os.path.dirname(__file__)
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from run_final import plan_units, _eval_cmd, eval_unit_name, aggregate_covers   # noqa: E402
from shared.persistence import done_marker_name, outage_abort, push_eval_results  # noqa: E402

ALL3 = ["sft", "grpo", "ppo"]
EVAL_ALL3 = "grpo-ppo-sft_test_full"          # eval-unit suffix (methods sorted) for the default plan


def _units(plan):
    return [u for _, _, _, u in plan]


def test_plan_all_training_then_eval_per_seed():
    plan = plan_units([1, 2], ALL3, done=set())
    assert _units(plan) == ["sft_s1", "grpo_s1", "ppo_s1", "sft_s2", "grpo_s2", "ppo_s2",
                            f"eval_s1_{EVAL_ALL3}", f"eval_s2_{EVAL_ALL3}"]
    kinds = [k for k, _, _, _ in plan]
    assert kinds[-2:] == ["eval", "eval"] and set(kinds[:-2]) == {"train"}


def test_plan_skips_completed_units_on_resume():
    done = {"sft_s1", "grpo_s1", "ppo_s1", "sft_s2", f"eval_s1_{EVAL_ALL3}"}
    plan = plan_units([1, 2], ALL3, done=done)
    assert _units(plan) == ["grpo_s2", "ppo_s2", f"eval_s2_{EVAL_ALL3}"]


def test_plan_probe_single_seed_single_method():
    plan = plan_units([7], ["ppo"], done=set())
    assert _units(plan) == ["ppo_s7", "eval_s7_ppo_test_full"]


def test_probe_sentinel_does_not_poison_final_run():
    """A single-method run's eval sentinel must not satisfy a later run's differently-shaped eval unit."""
    probe_done = {u for _, _, _, u in plan_units([1], ["ppo"], done=set())}
    final = plan_units([1, 2, 3], ALL3, done=probe_done, eval_limit=150)
    evals = [u for u in _units(final) if u.startswith("eval_s1")]
    assert evals == ["eval_s1_grpo-ppo-sft_test_L150"]      # seed-1 eval re-runs in the new shape
    assert "ppo_s1" not in _units(final)                    # training is still reused


def test_eval_unit_name_encodes_params():
    a = eval_unit_name(1, ["ppo"], "test", None)
    b = eval_unit_name(1, ALL3, "test", None)
    c = eval_unit_name(1, ALL3, "test", 150)
    d = eval_unit_name(1, ALL3, "validation", 150)
    assert len({a, b, c, d}) == 4                           # any param change -> a different unit


def test_done_marker_name_is_run_scoped():
    """Marker names encode seeds+methods so one run's marker can't trigger another's killer."""
    probe = done_marker_name([1], ["ppo"])
    final = done_marker_name([1, 2, 3], ALL3)
    assert probe == "_ALL_DONE_s1_ppo" and final == "_ALL_DONE_s1-2-3_grpo-ppo-sft"
    assert done_marker_name([3, 1, 2], ["ppo", "sft", "grpo"]) == final   # order-insensitive
    assert probe != final
    assert done_marker_name([1, 2, 3], ALL3) == final       # both sides derive identically


def test_aggregate_covers_requires_full_seed_coverage():
    """The gate demands every trained method over every requested seed."""
    full = {"by_method": {"sft": {"seeds_found": [1, 2, 3]}, "ppo": {"seeds_found": [1, 2, 3]},
                          "base": {"seeds_found": [1]}}}     # base is first-seed-only by design
    assert aggregate_covers(full, [1, 2, 3], ["sft", "ppo"])
    partial = {"by_method": {"sft": {"seeds_found": [2, 3]}, "ppo": {"seeds_found": [1, 2, 3]}}}
    assert not aggregate_covers(partial, [1, 2, 3], ["sft", "ppo"])
    assert not aggregate_covers({"by_method": {}}, [1], ["ppo"])
    assert aggregate_covers(full, [1, 2, 3], ["ppo"])        # probe shape: only ppo required


def test_outage_abort_threshold():
    """A sustained buyer/judge outage must abort, not burn the update budget and exit 0."""
    assert not outage_abort(4, 5)
    assert outage_abort(5, 5) and outage_abort(9, 5)
    assert not outage_abort(100, 0)                          # 0 disables


def test_push_eval_results_missing_dir_fails_before_network():
    """A missing seed dir fails the local check before any HfApi call."""
    with tempfile.TemporaryDirectory() as root:
        os.makedirs(os.path.join(root, "eval_s1"))
        assert push_eval_results("org/repo", root, [1, 2]) is False   # seed 2's dir missing


def test_killer_arming_requires_absent_then_present():
    """The killer arms only after seeing the marker absent once; a stale marker never fires it."""
    from local_killer import should_fire
    stale = {}
    assert should_fire(True, stale) is False            # present at first poll -> stale, hold fire
    assert should_fire(True, stale) is False            # still present -> still holding
    assert should_fire(False, stale) is False           # run_final cleared it -> ARMS
    assert should_fire(True, stale) is True             # reappears -> genuine completion, fire
    fresh = {}
    assert should_fire(False, fresh) is False           # normal run: absent from the start -> arms
    assert should_fire(True, fresh) is True
    late = {}
    assert should_fire(True, late, allow_preexisting=True) is True   # explicit late-restart override
    # a failed poll (present=None) carries no information: it must neither fire nor arm
    blip = {}
    assert should_fire(None, blip) is False             # failed first poll -> no arming
    assert should_fire(True, blip) is False             # stale marker after the blip -> holds fire
    assert should_fire(None, blip, allow_preexisting=True) is False  # override still needs a real poll


def test_eval_cmd_base_once():
    # base is seed-independent: only the first seed evaluates it; the rest pass --no-base
    first = _eval_cmd(1, "test", first_seed=1, methods=ALL3)
    later = _eval_cmd(2, "test", first_seed=1, methods=ALL3)
    assert "--no-base" not in first
    assert "--no-base" in later
    assert "--seed" in first and "2" in later


def test_eval_cmd_evals_only_trained_methods():
    # evaluating untrained methods would crash loading absent adapters
    cmd = _eval_cmd(1, "test", first_seed=1, methods=["ppo"])
    i = cmd.index("--methods")
    assert cmd[i + 1] == "ppo" and "sft" not in cmd and "grpo" not in cmd


def _run_all():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print(f"  ok  {fn.__name__}")
    print(f"PASS  {len(fns)} orchestrator tests")


if __name__ == "__main__":
    _run_all()
