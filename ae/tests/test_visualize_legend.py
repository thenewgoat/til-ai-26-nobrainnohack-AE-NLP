"""visualize._build_legend renders a color-keyed policy legend band."""
import numpy as np

from visualize import _build_legend, LEGEND_ROW_H


def test_legend_shape_and_content():
    labels = ["ckpt:policy_bc.pt", "scripted:balanced", "scripted:base_rusher",
              "random", "random", "random"]
    band = _build_legend(labels, width=400)
    assert band.dtype == np.uint8
    assert band.ndim == 3 and band.shape[2] == 3
    # width matches the requested frame width
    assert band.shape[1] == 400
    # height is one fixed-height row per label
    assert band.shape[0] == LEGEND_ROW_H * len(labels)
    # not a uniformly blank band — swatches + text add variation
    assert band.std() > 0.0


def test_legend_swatches_use_team_colors():
    """Each row carries its team's swatch color somewhere in that row band."""
    from til_environment.renderer import _team_color
    labels = ["random"] * 6
    band = _build_legend(labels, width=400)
    for k in range(6):
        row = band[k * LEGEND_ROW_H:(k + 1) * LEGEND_ROW_H]
        r, g, b = _team_color(k)
        # the swatch color appears as an exact pixel in this row
        hit = np.any((row[:, :, 0] == r) & (row[:, :, 1] == g)
                     & (row[:, :, 2] == b))
        assert hit, f"team {k} swatch color {(r, g, b)} not found in row"


def test_legend_layers_add_text():
    from visualize import _build_legend
    labels = ["scripted:balanced"] * 6
    plain = _build_legend(labels, width=400)
    withlayers = _build_legend(labels, width=400, layers=["escape_bomb"] * 6)
    assert plain.shape == withlayers.shape
    # the appended layer text adds a non-trivial number of dark pixels
    assert (withlayers < 250).sum() - (plain < 250).sum() > 50


def test_agent_layer_resolver():
    from visualize import _agent_layer
    from evaluate import ScriptedAgent, RandomAgent
    sc = ScriptedAgent("balanced")
    sc.belief.last_layer = "sweep"
    assert _agent_layer(sc) == "sweep"
    sc.belief.last_layer = None
    assert _agent_layer(sc) == ""        # None -> blank
    assert _agent_layer(RandomAgent()) == ""   # no cascade -> blank


def test_render_episode_smoke(tmp_path):
    import os
    from visualize import render_episode
    from evaluate import ScriptedAgent, RandomAgent
    slot_agents = {
        "agent_0": (ScriptedAgent("balanced"), "scripted:balanced"),
        "agent_1": (RandomAgent(), "random"),
        "agent_2": (RandomAgent(), "random"),
        "agent_3": (RandomAgent(), "random"),
        "agent_4": (RandomAgent(), "random"),
        "agent_5": (RandomAgent(), "random"),
    }
    out = str(tmp_path / "smoke.mp4")
    result = render_episode(slot_agents, out, max_steps=8)
    assert os.path.exists(out) and os.path.getsize(out) > 0
    assert 1 <= result["steps"] <= 8
