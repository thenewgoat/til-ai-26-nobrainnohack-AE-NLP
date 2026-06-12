"""Tests for the line-of-sight bomb DangerMap."""
from scripted.belief import Belief
from scripted.danger import DangerMap


class _Prior:
    """Minimal prior: a grid and a wall map (frozenset pair -> destructible)."""

    def __init__(self, grid_size=16, wall_between=None):
        self.grid_size = grid_size
        self.wall_between = dict(wall_between or {})


def _belief(grid_size=16, walls=None):
    """A Belief with just enough state for DangerMap's LOS check."""
    b = Belief()
    b.prior = _Prior(grid_size, walls)
    return b


def test_no_bombs_is_all_safe():
    d = DangerMap({}, _belief(16))
    assert d.ticks_to_danger((5, 5)) == 999
    assert not d.is_dangerous((5, 5))
    assert d.overlap((5, 5)) == 0


def test_bomb_threatens_cells_with_clear_line_of_sight():
    d = DangerMap({(5, 5): 3}, _belief(16))
    assert d.is_dangerous((5, 5))                    # the bomb's own cell
    assert d.is_dangerous((5, 7))                    # Chebyshev 2, clear LOS
    assert d.is_dangerous((7, 7))                    # diagonal, clear LOS
    assert d.ticks_to_danger((5, 7)) == 3


def test_blast_does_not_reach_beyond_chebyshev_2():
    d = DangerMap({(5, 5): 3}, _belief(16))
    assert not d.is_dangerous((5, 8))                # Chebyshev 3 — out of range


def test_a_destructible_wall_blocks_the_blast():
    # A destructible wall between the bomb (3,3) and (3,4) shadows below it.
    walls = {frozenset({(3, 3), (3, 4)}): True}
    d = DangerMap({(3, 3): 3}, _belief(7, walls))
    assert d.is_dangerous((3, 3))                    # bomb cell still lethal
    assert not d.is_dangerous((3, 4))                # wall blocks the blast
    assert not d.is_dangerous((3, 5))                # shadowed behind the wall


def test_an_indestructible_wall_also_blocks_the_blast():
    walls = {frozenset({(3, 3), (4, 3)}): False}     # False == indestructible
    d = DangerMap({(3, 3): 3}, _belief(7, walls))
    assert not d.is_dangerous((4, 3))


def test_a_confirmed_destroyed_wall_no_longer_blocks():
    walls = {frozenset({(3, 3), (3, 4)}): True}
    b = _belief(7, walls)
    b.destroyed_walls.add(frozenset({(3, 3), (3, 4)}))
    d = DangerMap({(3, 3): 3}, b)
    assert d.is_dangerous((3, 4))                    # wall gone -> blast passes


def test_ticks_to_danger_is_the_soonest_detonation():
    d = DangerMap({(5, 5): 4, (6, 5): 1}, _belief(16))
    assert d.ticks_to_danger((6, 5)) == 1            # the nearer-fused bomb wins


def test_overlap_counts_bombs_covering_a_cell():
    d = DangerMap({(5, 5): 3, (6, 5): 2}, _belief(16))
    assert d.overlap((5, 5)) == 2                    # both bombs reach it
    assert d.overlap((6, 5)) == 2


def test_is_lethal_at_only_on_the_detonation_tick():
    d = DangerMap({(5, 5): 3}, _belief(16))
    assert d.is_lethal_at((5, 5), 3)                 # blast lands at tick 3
    assert not d.is_lethal_at((5, 5), 2)
    assert not d.is_lethal_at((5, 5), 4)
