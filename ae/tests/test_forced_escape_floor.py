import numpy as np

from scripted.belief import Belief
from scripted.danger import DangerMap
from scripted.escape import (ESCAPE_ACTIONS, escape_selector,
                             legal_escape_actions, must_force_escape)
from scripted.geometry import BACKWARD, FORWARD, LEFT, PLACE_BOMB, RIGHT, STAY


class _Prior:
    def __init__(self, grid_size=5):
        self.grid_size = grid_size
        self.wall_between = {}
        self.collectibles = {}


def _belief(loc, facing, grid_size=5):
    b = Belief()
    b.prior = _Prior(grid_size)
    b.location = loc
    b.facing = facing
    b.team_bombs = 3
    return b


def _all_legal():
    return np.ones(6, dtype=bool)


def test_escape_actions_exclude_place_bomb():
    assert PLACE_BOMB not in ESCAPE_ACTIONS
    assert set(ESCAPE_ACTIONS) == {FORWARD, BACKWARD, LEFT, RIGHT, STAY}


def test_legal_escape_actions_filters_by_mask():
    mask = np.ones(6, dtype=bool)
    mask[FORWARD] = False
    legal = legal_escape_actions(mask)
    assert FORWARD not in legal
    assert STAY in legal


def test_no_force_when_not_in_danger():
    b = _belief((0, 0), 0)
    assert must_force_escape(b, DangerMap({}, b), _all_legal()) is False


def test_no_force_when_stay_then_move_survives():
    b = _belief((2, 0), 0)                              # facing RIGHT
    danger = DangerMap({(0, 0): 2}, b)
    assert danger.is_dangerous((2, 0), within=5) is True
    assert danger.is_lethal_at((2, 0), 2) is True
    assert danger.is_lethal_at((3, 0), 2) is False     # 1-move-away safe cell
    assert must_force_escape(b, danger, _all_legal()) is False


def test_force_when_stay_is_trapped_but_a_move_escapes():
    b = _belief((1, 0), 0)                              # facing RIGHT
    danger = DangerMap({(0, 0): 2}, b)
    assert danger.is_dangerous((1, 0), within=5) is True
    assert danger.is_lethal_at((2, 0), 2) is True
    assert danger.is_lethal_at((3, 0), 2) is False
    assert must_force_escape(b, danger, _all_legal()) is True
    action = escape_selector(b, danger, None, _all_legal())
    assert action == FORWARD                            # the surviving escape
    assert action != PLACE_BOMB                         # floor is bomb-free


def test_escape_selector_never_returns_place_bomb_even_when_all_doomed():
    b = _belief((0, 0), 0, grid_size=1)
    danger = DangerMap({(0, 0): 1}, b)
    action = escape_selector(b, danger, None, _all_legal())
    assert action in ESCAPE_ACTIONS
    assert action != PLACE_BOMB


def test_fires_and_picks_a_move_when_stay_is_masked_illegal():
    # Same trapped geometry as the fire test, but STAY is illegal in the mask.
    # The sentinel for the missing STAY score is below any surviving move, so the
    # floor still fires and the selector still picks the surviving FORWARD.
    b = _belief((1, 0), 0)                              # facing RIGHT
    danger = DangerMap({(0, 0): 2}, b)
    mask = _all_legal()
    mask[STAY] = False
    assert must_force_escape(b, danger, mask) is True
    assert escape_selector(b, danger, None, mask) == FORWARD
