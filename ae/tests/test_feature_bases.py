"""Per-base tokens: effective_hp / is_target / is_doomed / bombs_needed."""
import math
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


def _builder(seed=0):
    cfg = default_config()
    cfg.env.novice = True
    env = bomberman_env.basic_env(env_wrappers=[], cfg=cfg)
    env.reset(seed=seed)
    obs, _, _, _, _ = env.last()
    env.close()
    fb = FeatureBuilder()
    fb.build(obs)
    return fb


def test_base_feats_shape_and_range():
    fb = _builder()
    from scripted.danger import DangerMap
    from scripted.pathfind import build_planner
    danger = DangerMap(fb.belief.enemy_bombs, fb.belief)
    planner = build_planner(fb.belief, danger)
    feats = fb._build_base_feats(planner, None)
    assert feats.shape == (F.NUM_BASES, F.BASE_FIELDS)
    assert np.isfinite(feats).all()
    assert feats.min() >= 0.0
    assert feats.max() <= 1.0


def test_full_hp_base_renders_effective_hp_one():
    fb = _builder()
    # an unseen base is assumed full HP -> effective_hp ratio 1.0
    feats = fb._build_base_feats(None, None)
    for i, base in enumerate(fb.prior.enemy_bases):
        if base not in fb.belief.enemy_base_health:
            assert feats[i, F.BF_EFFECTIVE_HP] == 1.0
            assert feats[i, F.BF_OBSERVED_HP] == 1.0
            assert feats[i, F.BF_BOMBS_NEEDED] == math.ceil(100 / 20) / 5.0


def test_dead_base_marked_is_dead():
    fb = _builder()
    dead = fb.prior.enemy_bases[0]
    fb.belief.dead_bases.add(dead)
    feats = fb._build_base_feats(None, None)
    assert feats[0, F.BF_IS_DEAD] == 1.0


def test_target_base_marked_is_target():
    fb = _builder()
    from scripted.danger import DangerMap
    from scripted.pathfind import build_planner
    danger = DangerMap(fb.belief.enemy_bombs, fb.belief)
    planner = build_planner(fb.belief, danger)
    target = (fb.prior.enemy_bases[2], 100.0, 5)
    feats = fb._build_base_feats(planner, target)
    assert feats[2, F.BF_IS_TARGET] == 1.0
    assert feats[0, F.BF_IS_TARGET] == 0.0


def test_soften_phase_and_leash_track_effective_hp():
    fb = _builder()
    soften_floor = fb.strategy.params.soften_floor  # 60.0
    full = fb.prior.enemy_bases[1]
    crippled = fb.prior.enemy_bases[2]
    # full-HP base is in the soften phase; a crippled base (below floor) is not
    fb.belief.enemy_base_health[crippled] = 0.5  # eff 50 < 60
    feats = fb._build_base_feats(None, target=(crippled, 50.0, 3))
    i_full = fb.prior.enemy_bases.index(full)
    i_crip = fb.prior.enemy_bases.index(crippled)
    assert feats[i_full, F.BF_IS_SOFTEN_PHASE] == 1.0
    assert feats[i_crip, F.BF_IS_SOFTEN_PHASE] == 0.0
    # the Phase-B target (crippled, not softening) gets a non-zero leash;
    # a soften-phase base never leashes
    assert feats[i_crip, F.BF_LEASH_RADIUS] > 0.0
    assert feats[i_full, F.BF_LEASH_RADIUS] == 0.0
