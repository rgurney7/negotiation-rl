"""Pure-helper tests for shared/model.py (no GPU)."""

import os
import sys
import types

HERE = os.path.dirname(__file__)
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from shared import model as model_mod   # noqa: E402


class _Cfg:
    def __init__(self, require):
        self.require_fast_kernels = require


def test_fast_kernels_available_returns_bool():
    assert isinstance(model_mod.fast_kernels_available(), bool)


def test_warn_not_fail_when_missing_and_not_required():
    # kernels absent, require=False -> warn, return False, do NOT raise
    assert model_mod.check_fast_kernels(_Cfg(False), available=False) is False


def test_hard_fail_when_required_and_missing():
    raised = False
    try:
        model_mod.check_fast_kernels(_Cfg(True), available=False)
    except RuntimeError:
        raised = True
    assert raised, "require_fast_kernels=True must hard-fail when the kernels are missing"


def test_ok_when_available_regardless_of_require():
    assert model_mod.check_fast_kernels(_Cfg(True), available=True) is True   # present -> never raises
    assert model_mod.check_fast_kernels(_Cfg(False), available=True) is True


class _Model:
    def __init__(self, config):
        self.config = config


def test_assert_text_only_fires_on_top_level_vision_config():
    # composite HF layout: top-level vision_config, text sub-config with vision_config=None
    text_sub = types.SimpleNamespace(vision_config=None, hidden_size=2560)
    composite = types.SimpleNamespace(vision_config={"hidden_size": 1152}, text_config=text_sub,
                                      get_text_config=lambda: text_sub)
    raised = False
    try:
        model_mod._assert_text_only(_Model(composite))
    except RuntimeError:
        raised = True
    assert raised, "must reject a model whose top-level config has a populated vision_config"


def test_assert_text_only_passes_dense_text_model():
    dense = types.SimpleNamespace(hidden_size=2560)            # no vision_config anywhere
    model_mod._assert_text_only(_Model(dense))                 # must not raise
    model_mod._assert_text_only(_Model(None))                  # no config -> no-op


def _run_all():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print(f"  ok  {fn.__name__}")
    print(f"PASS  {len(fns)} model tests")


if __name__ == "__main__":
    _run_all()
