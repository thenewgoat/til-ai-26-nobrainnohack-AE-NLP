"""Abstraction tile grid: per-channel content from a hand-built belief."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "src", "training"))

import numpy as np

import features as F
from features import FeatureBuilder
from til_environment import bomberman_env
from til_environment.config import default_config


def _build_grid(seed=0):
    cfg = default_config()
    cfg.env.novice = True
    env = bomberman_env.basic_env(env_wrappers=[], cfg=cfg)
    env.reset(seed=seed)
    obs, _, _, _, _ = env.last()
    env.close()
    fb = FeatureBuilder()
    grid, *_ = fb.build(obs)
    return grid, fb


def test_self_channel_marks_agent_location():
    grid, fb = _build_grid()
    loc = fb.belief.location
    assert grid[F.CH_SELF, loc[0], loc[1]] == 1.0
    assert grid[F.CH_SELF].sum() == 1.0


def test_danger_and_overlap_present_when_enemy_bomb_known():
    grid, fb = _build_grid()
    fb.belief.enemy_bombs = {(5, 5): 2}
    g = np.zeros((F.GRID_CHANNELS, F.GRID_SIZE, F.GRID_SIZE), np.float32)
    from scripted.danger import DangerMap
    danger = DangerMap(fb.belief.enemy_bombs, fb.belief)
    fb._fill_danger(g, danger)
    assert g[F.CH_DANGER, 5, 5] > 0.0
    assert g[F.CH_DANGER_OVERLAP, 5, 5] > 0.0
    assert g[F.CH_ENEMY_BOMB, 5, 5] > 0.0


def test_own_and_enemy_bombs_render_to_separate_channels():
    grid, fb = _build_grid()
    g = np.zeros((F.GRID_CHANNELS, F.GRID_SIZE, F.GRID_SIZE), np.float32)
    fb.belief.enemy_bombs = {(3, 3): 1}
    fb.belief.own_bombs = [((9, 9), 4)]
    from scripted.danger import DangerMap
    danger = DangerMap(fb.belief.enemy_bombs, fb.belief)
    fb._fill_danger(g, danger)
    assert g[F.CH_ENEMY_BOMB, 3, 3] > 0.0
    assert g[F.CH_OWN_BOMB, 9, 9] > 0.0
    assert g[F.CH_ENEMY_BOMB, 9, 9] == 0.0
    assert g[F.CH_OWN_BOMB, 3, 3] == 0.0


def test_walls_channels_in_range():
    grid, _ = _build_grid()
    for ch in (F.CH_WALL_R, F.CH_WALL_D, F.CH_WALL_L, F.CH_WALL_U):
        assert grid[ch].min() >= 0.0 and grid[ch].max() <= 1.0


def test_confidence_marks_agent_tile_fresh():
    grid, fb = _build_grid()
    loc = fb.belief.location
    assert grid[F.CH_CONFIDENCE, loc[0], loc[1]] == 1.0


def test_confidence_decays_for_stale_tiles():
    grid, fb = _build_grid()
    # a tile seen at step 0, then queried 8 steps later, decays below 1.0
    g = np.zeros((F.GRID_CHANNELS, F.GRID_SIZE, F.GRID_SIZE), np.float32)
    fb._last_seen[2, 3] = 0
    fb._fill_respawn_confidence(g, step=8)
    assert 0.0 < g[F.CH_CONFIDENCE, 2, 3] < 1.0
    # a never-seen tile stays at 0
    assert g[F.CH_CONFIDENCE, 0, 0] == 0.0


def test_base_mark_signs_our_base_positive_enemy_negative():
    grid, fb = _build_grid()
    g = np.zeros((F.GRID_CHANNELS, F.GRID_SIZE, F.GRID_SIZE), np.float32)
    fb._fill_base_mark(g)
    ox, oy = fb.prior.our_base
    assert g[F.CH_BASE_MARK, ox, oy] == 1.0
    for (bx, by) in fb.belief.live_enemy_bases():
        assert g[F.CH_BASE_MARK, bx, by] == -1.0
