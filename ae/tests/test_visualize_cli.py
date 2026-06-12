"""visualize.run_cli builds slot_agents from specs and renders an MP4."""
import os

from visualize import run_cli, VizArgs


def test_run_cli_renders_from_specs(tmp_path):
    out = os.path.join(str(tmp_path), "demo.mp4")
    args = VizArgs(
        agents=["random", "random", "random",
                "random", "random", "random"],
        out=out, fps=5, max_steps=15, seed=0)
    result = run_cli(args)
    assert os.path.exists(out) and os.path.getsize(out) > 0
    assert result["path"] == out
