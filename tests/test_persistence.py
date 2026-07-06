"""Transcript log, checkpoint save/load, and mark_preview safety."""

import json
import os
import tempfile

from shared import persistence
from ppo.config import PPOConfig
from sft.config import SFTConfig


class _StubCfg:
    """Only the attribute log_transcript reads."""
    def __init__(self, path):
        self.transcript_file = path


def test_log_transcript_appends_valid_jsonl():
    d = tempfile.mkdtemp()
    # Deliberately a non-existent nested dir: log_transcript must create it.
    cfg = _StubCfg(os.path.join(d, "ppo_runs", "transcripts.jsonl"))
    persistence.log_transcript(cfg, {
        "update": 1, "scenario_id": "abc", "reward": 0.5,
        "turns": [["buyer", "would you take 80?"], ["seller", "I can do 90."]],
    })
    persistence.log_transcript(cfg, {"update": 1, "scenario_id": "xyz", "reward": -1.0, "turns": []})

    rows = [json.loads(line) for line in open(cfg.transcript_file)]
    assert len(rows) == 2, "append-only: two records -> two lines"
    assert rows[0]["scenario_id"] == "abc"
    assert rows[0]["turns"][1] == ["seller", "I can do 90."], "nested turns survive the round-trip"
    assert rows[1]["reward"] == -1.0 and rows[1]["turns"] == []


class _FakeAgent:
    """Minimal agent for save/load_checkpoint: writes the lora.safetensors that resume validates."""
    device = "cpu"

    def __init__(self):
        self.loaded_from = None

    def save(self, d):
        open(os.path.join(d, "lora.safetensors"), "w").write("w")

    def load(self, d):
        self.loaded_from = d


class _CkptCfg:
    def __init__(self, base, keep=0):
        self.checkpoint_dir = os.path.join(base, "checkpoints")
        self.keep_last_k_checkpoints = keep


def test_checkpoint_atomic_resume_and_crash_fallback():
    cfg = _CkptCfg(tempfile.mkdtemp())
    a = _FakeAgent()
    persistence.save_checkpoint(a, None, cfg, 5)
    persistence.save_checkpoint(a, None, cfg, 10)
    assert os.path.exists(os.path.join(cfg.checkpoint_dir, "step_0010", "_COMPLETE"))
    assert persistence.load_checkpoint(a, None, cfg) == 10           # highest valid step

    # a crashed checkpoint (adapter present, NO sentinel) with a lying latest.txt must be skipped
    crash = os.path.join(cfg.checkpoint_dir, "step_0015")
    os.makedirs(crash)
    open(os.path.join(crash, "lora.safetensors"), "w").write("partial")
    open(os.path.join(cfg.checkpoint_dir, "latest.txt"), "w").write("15")
    assert persistence.load_checkpoint(a, None, cfg) == 10           # falls back, ignores the lie
    assert a.loaded_from.endswith("step_0010")

    assert persistence.load_checkpoint(a, None, _CkptCfg(tempfile.mkdtemp())) == 0  # no dir -> 0


def test_optimizer_state_survives_checkpoint_resume():
    """A stepped AdamW's moments must round-trip through a checkpoint."""
    import torch
    cfg = _CkptCfg(tempfile.mkdtemp())
    a = _FakeAgent()
    p = torch.nn.Parameter(torch.zeros(4))
    opt = torch.optim.AdamW([p], lr=1e-3)
    p.grad = torch.ones(4)
    opt.step()                                          # populate exp_avg / exp_avg_sq / step
    ref_avg = opt.state[p]["exp_avg"].clone()

    persistence.save_checkpoint(a, opt, cfg, 7)
    assert os.path.exists(os.path.join(cfg.checkpoint_dir, "step_0007", "optimizer.pt"))

    p2 = torch.nn.Parameter(torch.zeros(4))
    opt2 = torch.optim.AdamW([p2], lr=1e-3)             # a fresh, cold optimizer
    assert not opt2.state                              # nothing accumulated yet
    assert persistence.load_checkpoint(a, opt2, cfg) == 7
    restored = opt2.state[opt2.param_groups[0]["params"][0]]
    assert torch.allclose(restored["exp_avg"], ref_avg)   # momentum restored, not cold-started
    assert float(restored["step"]) == 1.0


def test_prune_keeps_last_k_checkpoints():
    cfg = _CkptCfg(tempfile.mkdtemp(), keep=2)
    a = _FakeAgent()
    for step in (5, 10, 15, 20):
        persistence.save_checkpoint(a, None, cfg, step)
    kept = sorted(n for n in os.listdir(cfg.checkpoint_dir) if n.startswith("step_"))
    assert kept == ["step_0015", "step_0020"], "keep_last_k=2 keeps only the two highest steps"
    assert persistence.load_checkpoint(a, None, cfg) == 20      # resume still finds the latest


def test_prune_disabled_keeps_all():
    cfg = _CkptCfg(tempfile.mkdtemp(), keep=0)                  # 0 -> keep everything
    a = _FakeAgent()
    for step in (5, 10, 15):
        persistence.save_checkpoint(a, None, cfg, step)
    kept = sorted(n for n in os.listdir(cfg.checkpoint_dir) if n.startswith("step_"))
    assert kept == ["step_0005", "step_0010", "step_0015"]


def test_should_teardown_matrix():
    from types import SimpleNamespace as NS
    from shared.persistence import _should_teardown

    def cfg(**kw):
        return NS(**{"stop_pod_on_finish": True, "hf_repo_id": "org/r", **kw})

    assert _should_teardown(cfg(), True, "pod1", True) is False    # orchestrated -> NEVER self-stop
    assert _should_teardown(cfg(), True, "pod1", False) is True    # solo run, push ok, on a pod
    assert _should_teardown(cfg(), False, "pod1", False) is False  # final push failed -> keep pod
    assert _should_teardown(cfg(), True, None, False) is False     # not on a pod
    assert _should_teardown(cfg(stop_pod_on_finish=False), True, "pod1", False) is False
    assert _should_teardown(cfg(hf_repo_id=""), False, "pod1", False) is True  # no mirror -> push n/a


def test_hf_token_present_checks_length_only():
    assert persistence.hf_token_present("h" * 37) is True
    assert persistence.hf_token_present("") is False
    assert persistence.hf_token_present("short") is False


def test_mark_preview_is_safe_and_isolated():
    cfg = PPOConfig()
    real_repo, real_dir = cfg.hf_repo_id, cfg.output_dir
    assert real_repo and not real_dir.endswith("_preview")   # sanity: a real run by default

    cfg.num_updates = 3
    persistence.mark_preview(cfg)

    assert cfg.run_name.endswith("_preview")
    assert cfg.hf_repo_id == "", "preview must NOT mirror to the real HF repo"
    assert cfg.stop_pod_on_finish is False, "preview must NEVER stop the pod"
    assert "ppo_runs_preview" in cfg.output_dir and cfg.output_dir != real_dir
    assert cfg.transcript_file.endswith("transcripts.jsonl")
    assert cfg.transcript_file.startswith(cfg.output_dir), "transcript lands in the preview dir"
    assert 1 <= cfg.checkpoint_every <= cfg.num_updates


def test_mark_preview_handles_config_without_num_updates():
    # SFTConfig has epochs, not num_updates — mark_preview must not crash on it.
    cfg = SFTConfig()
    persistence.mark_preview(cfg)
    assert cfg.run_name.endswith("_preview")
    assert cfg.hf_repo_id == "" and cfg.stop_pod_on_finish is False


def test_sft_valid_resume_checkpoint_skips_torn_dir():
    """A torn checkpoint dir (no trainer_state.json) is skipped for the newest complete one."""
    import json
    import tempfile
    valid_resume_checkpoint = persistence.valid_hf_trainer_checkpoint

    root = tempfile.mkdtemp()
    assert valid_resume_checkpoint(root) is None                     # nothing yet -> fresh start
    good = os.path.join(root, "checkpoint-100")
    os.makedirs(good)
    with open(os.path.join(good, "trainer_state.json"), "w") as f:
        json.dump({"global_step": 100}, f)
    torn = os.path.join(root, "checkpoint-200")
    os.makedirs(torn)                                                # higher step, NO state file
    assert valid_resume_checkpoint(root) == good                     # torn dir skipped
    with open(os.path.join(torn, "trainer_state.json"), "w") as f:
        json.dump({"global_step": 200}, f)
    assert valid_resume_checkpoint(root) == torn                     # complete now -> preferred
