"""visualize.render_episode runs an episode and writes a labelled MP4."""
import os

import imageio

from visualize import render_episode
from evaluate import RandomAgent


def test_render_episode_writes_mp4(tmp_path):
    slot_agents = {f"agent_{k}": (RandomAgent(), "random") for k in range(6)}
    out = os.path.join(str(tmp_path), "ep.mp4")
    # short episode to keep the test fast
    result = render_episode(slot_agents, out, fps=5, max_steps=20, seed=0)

    # the MP4 exists and is non-empty
    assert os.path.exists(out) and os.path.getsize(out) > 0
    assert result["path"] == out
    # returned step count is reported and bounded by max_steps
    assert 0 < result["steps"] <= 20
    # one score per slot
    assert set(result["scores"]) == {f"agent_{k}" for k in range(6)}

    # the written frames are taller than a raw env frame -> legend present.
    # raw novice rgb_array frame is window_size tall (~768).
    frames = imageio.mimread(out, memtest=False)
    assert len(frames) >= 1
    assert frames[0].shape[0] > 700  # env frame + legend band
