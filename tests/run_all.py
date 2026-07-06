"""Run every offline test (no GPU, no API)."""

import importlib
import os
import sys

HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.dirname(HERE))

MODULES = ["test_render_parity", "test_reward", "test_gae",
           "test_grpo_loss", "test_masking", "test_judge", "test_env", "test_ppo_truncate",
           "test_persistence", "test_eval", "test_model", "test_seed_aggregate",
           "test_orchestrator"]

if __name__ == "__main__":
    sys.path.insert(0, HERE)
    total = 0
    for name in MODULES:
        mod = importlib.import_module(name)
        fns = [v for k, v in sorted(vars(mod).items()) if k.startswith("test_") and callable(v)]
        for fn in fns:
            fn()
            total += 1
        print(f"  ✓ {name} ({len(fns)})")
    print(f"\nALL OFFLINE TESTS PASS — {total} checks across {len(MODULES)} modules")
