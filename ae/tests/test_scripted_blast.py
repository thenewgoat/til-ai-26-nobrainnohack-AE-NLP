"""Tests for the ally-bomb blast / wall-destruction predictor."""
import numpy as np
from til_environment.helpers import supercover_line as env_supercover

from scripted.belief import Belief
from scripted.blast import (
    _los_to_tile, bomb_reaches, supercover_line, walls_destroyed_by,
)


class _Prior:
    """Hand-built prior: small grid; wall_between supplied per test."""

    def __init__(self, grid_size, wall_between):
        self.grid_size = grid_size
        self.wall_between = wall_between
        self.collectibles = {}


def _belief(grid_size, wall_between):
    b = Belief()
    b.prior = _Prior(grid_size, wall_between)
    b.destroyed_walls = set()
    return b


def test_supercover_matches_env():
    """Our vendored supercover_line must match til_environment's exactly."""
    for (sx, sy), (ex, ey) in [((0, 0), (4, 2)), ((1, 1), (1, 5)),
                               ((3, 3), (0, 0)), ((2, 0), (5, 5))]:
        ours = supercover_line((sx, sy), (ex, ey))
        env = env_supercover(np.array([sx, sy]), np.array([ex, ey]))
        assert ours == [tuple(int(c) for c in t) for t in env]


def test_los_clear_then_blocked():
    b = _belief(5, {})
    assert _los_to_tile(0, 0, 0, 2, b) is True            # clear line
    b.prior.wall_between = {frozenset({(0, 1), (0, 2)}): False}
    assert _los_to_tile(0, 0, 0, 2, b) is False           # wall blocks the line


def test_wall_on_bomb_cell_is_destroyed():
    w = frozenset({(2, 2), (3, 2)})
    b = _belief(5, {w: True})                             # destructible
    assert w in walls_destroyed_by((2, 2), b)


def test_wall_outside_blast_radius_survives():
    w = frozenset({(8, 8), (7, 8)})
    b = _belief(9, {w: True})
    assert w not in walls_destroyed_by((0, 0), b)


def test_indestructible_wall_never_destroyed():
    w = frozenset({(2, 2), (3, 2)})
    b = _belief(5, {w: False})                            # indestructible
    assert w not in walls_destroyed_by((2, 2), b)


def test_los_blocked_wall_survives():
    """A destructible wall the blast has no line-of-sight to (bomb sealed
    into its own cell by indestructible walls) is NOT destroyed."""
    walls = {frozenset({(0, 0), (1, 0)}): False,   # indestructible — seal bomb in
             frozenset({(0, 0), (0, 1)}): False,
             frozenset({(2, 2), (2, 3)}): True}    # destructible, but unreachable
    b = _belief(5, walls)
    assert frozenset({(2, 2), (2, 3)}) not in walls_destroyed_by((0, 0), b)


def test_bomb_reaches_within_radius_and_los():
    b = _belief(7, {})
    assert bomb_reaches((3, 3), (3, 3), b) is True        # same tile
    assert bomb_reaches((3, 3), (3, 5), b) is True        # Chebyshev 2, clear


def test_bomb_reaches_false_out_of_radius():
    b = _belief(9, {})
    assert bomb_reaches((0, 0), (0, 3), b) is False       # Chebyshev 3


def test_bomb_reaches_false_when_los_blocked():
    b = _belief(7, {frozenset({(3, 4), (3, 5)}): False})  # wall on the line
    assert bomb_reaches((3, 3), (3, 5), b) is False
