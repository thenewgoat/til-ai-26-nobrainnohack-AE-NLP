"""The five-tensor feature contract: frozen shapes and constant layout."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "src", "training"))

import features as F


def test_contract_dimensions():
    assert F.GRID_CHANNELS == 17
    assert F.NUM_BASES == 5
    assert F.BASE_FIELDS == 11
    assert F.FEATURE_SCALARS == 10
    assert F.RAW_AGENT_SHAPE == (7, 5, 25)
    assert F.RAW_BASE_SHAPE == (7, 7, 25)
    # frame-stack contract
    assert F.STACK == 5
    assert F.STACKED_GRID_CHANNELS == 85
    assert F.STACKED_SCALARS == 50


def test_grid_channel_indices_unique_and_dense():
    idx = [F.CH_WALL_R, F.CH_WALL_D, F.CH_WALL_L, F.CH_WALL_U, F.CH_SELF,
           F.CH_ENEMY_LIVE, F.CH_ENEMY_FROZEN, F.CH_COLLECTIBLE, F.CH_RESPAWN,
           F.CH_DANGER, F.CH_DANGER_OVERLAP, F.CH_ENEMY_BOMB, F.CH_OWN_BOMB,
           F.CH_PLANNER_DIST, F.CH_DIR_TARGET, F.CH_CONFIDENCE, F.CH_BASE_MARK]
    assert sorted(idx) == list(range(F.GRID_CHANNELS))


def test_base_field_indices_unique_and_dense():
    idx = [F.BF_EFFECTIVE_HP, F.BF_OBSERVED_HP, F.BF_BOMBS_NEEDED,
           F.BF_IS_DOOMED, F.BF_IS_TARGET, F.BF_IS_SOFTEN_PHASE,
           F.BF_LEASH_RADIUS, F.BF_ARRIVAL, F.BF_IN_RANGE_NOW,
           F.BF_OWN_IN_FLIGHT, F.BF_IS_DEAD]
    assert sorted(idx) == list(range(F.BASE_FIELDS))


def test_scalar_indices_unique_and_dense():
    idx = [F.SC_STEP, F.SC_TEAM_BOMBS, F.SC_RESOURCES, F.SC_HEALTH,
           F.SC_BASE_HEALTH, F.SC_FROZEN, F.SC_TEAM_ID, F.SC_FACING,
           F.SC_OWN_BOMBS_IN_FLIGHT, F.SC_LIVE_ENEMY_BASES]
    assert sorted(idx) == list(range(F.FEATURE_SCALARS))
