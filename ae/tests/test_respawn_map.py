"""respawn_map.json must exist, be 16x16, and hold the seed-88 novice delays."""
import json
from pathlib import Path

import numpy as np


def test_respawn_map_json_shape_and_range():
    path = Path(__file__).resolve().parents[1] / "src" / "respawn_map.json"
    assert path.exists(), "run dump_map.py to generate respawn_map.json"
    grid = np.array(json.loads(path.read_text()), dtype=np.int32)
    assert grid.shape == (16, 16)
    # default tile_respawn_steps=40 -> delays in [max(40//4,5), 40] = [10, 40]
    assert grid.min() >= 10
    assert grid.max() <= 40


def test_respawn_map_matches_env():
    """The dumped map must equal the env's novice respawn_map exactly."""
    from til_environment import bomberman_env
    from til_environment.config import default_config

    cfg = default_config()
    cfg.env.novice = True
    env = bomberman_env.basic_env(env_wrappers=[], cfg=cfg)
    env.reset(seed=0)
    env_map = np.asarray(env.unwrapped.dynamics.respawn_map, dtype=np.int32)
    env.close()

    path = Path(__file__).resolve().parents[1] / "src" / "respawn_map.json"
    dumped = np.array(json.loads(path.read_text()), dtype=np.int32)
    assert np.array_equal(dumped, env_map)
