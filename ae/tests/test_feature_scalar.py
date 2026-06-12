"""Scalar token: 10 fields, all normalized to ~[0,1]."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "src", "training"))

import features as F
from features import FeatureBuilder
from til_environment import bomberman_env
from til_environment.config import default_config


def _scalar(seed=0):
    cfg = default_config()
    cfg.env.novice = True
    env = bomberman_env.basic_env(env_wrappers=[], cfg=cfg)
    env.reset(seed=seed)
    obs, _, _, _, _ = env.last()
    env.close()
    fb = FeatureBuilder()
    _, _, _, _, scalar = fb.build(obs)
    return scalar, fb


def test_scalar_shape_and_range():
    scalar_stack, _ = _scalar()
    assert scalar_stack.shape == (F.STACKED_SCALARS,)
    scalar = scalar_stack[:F.FEATURE_SCALARS]   # newest frame
    assert (scalar >= 0.0).all() and (scalar <= 1.0).all()


def test_scalar_step_is_zero_at_reset():
    scalar, _ = _scalar()
    assert scalar[F.SC_STEP] == 0.0


def test_scalar_live_enemy_bases_full_at_start():
    scalar, fb = _scalar()
    assert scalar[F.SC_LIVE_ENEMY_BASES] == \
        len(fb.belief.live_enemy_bases()) / F.NUM_BASES
