"""Post-rollout trajectory shaping."""


def episode_failed(info):
    """True if the buyer or judge API failed, making the episode unscorable."""
    info = info or {}
    return bool(info.get("buyer_failed") or info.get("judge_failed"))


def drop_failed_episodes(episodes, infos):
    """Replace failed episodes with empty step lists; run before truncate_at_close."""
    return [[] if episode_failed(info) else steps
            for steps, info in zip(episodes, infos)]


def truncate_at_close(episodes, infos, cfg):
    """Drop steps after the deal closed and move the terminal reward onto the close step."""
    if not getattr(cfg, "truncate_after_deal", True):
        return episodes
    out = []
    for steps, info in zip(episodes, infos):
        info = info or {}
        cs = info.get("close_step")
        if steps and cs is not None and info.get("agreed_price") is not None:
            cs = max(1, min(int(cs), len(steps)))
            terminal_reward = info.get("reward", steps[-1]["reward"])
            steps = steps[:cs]
            for s in steps[:-1]:
                s["reward"] = 0.0
            steps[-1]["reward"] = float(terminal_reward)
        out.append(steps)
    return out
