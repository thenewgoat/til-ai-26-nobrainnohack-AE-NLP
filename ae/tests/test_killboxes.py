"""Tests for the offline killbox solver.

NOTE: The plan's original hand-built unit tests assumed Chebyshev-2 blast
without LOS occlusion AND a grid larger than the blast diameter. On a 3x3
grid every cell is within Chebyshev-2 of every other cell, so any bomb's
blast region covers the entire grid — the enemy never has a non-blast cell
to escape to, and ALL (A, E) pairs trivially become killboxes (not 0 as the
draft test expected). Additionally, the env's `_los_to_tile` blocks blast
LOS through walls — so a wall that pens an enemy in a corner ALSO blocks
the bomb from reaching them, making the draft corner test impossible by
construction.

The tests below test the same three behaviors the plan intended, but use a
7-wide grid (well over the blast diameter) and a 1-cell-wide corridor with
an indestructible wall at one end — a configuration in which (a) the bomb
clearly reaches the enemy via cardinal LOS, (b) the enemy is trapped from
escaping the blast in 4 moves, and (c) destructible walls vs indestructible
both pin the enemy.
"""
from scripted.map_prior import MapPrior
from tools.build_killboxes import _compute_killboxes


class _Prior:
    """Minimal stand-in for MapPrior with hand-built walls."""

    def __init__(self, grid_size, walls=None):
        self.grid_size = grid_size
        self.wall_between = dict(walls or {})
        self.collectibles = {}
        self.enemy_bases = []
        self.our_base = (0, 0)


def _corridor_walls(grid_size, y, destructible=False):
    """Walls fencing off row `y` from rows y-1 and y+1 across [0, grid_size).
    Leaves the row itself open so bombs/enemies can traverse it left-right.
    The `destructible` flag controls all walls uniformly."""
    walls = {}
    for x in range(grid_size):
        if y - 1 >= 0:
            walls[frozenset({(x, y), (x, y - 1)})] = destructible
        if y + 1 < grid_size:
            walls[frozenset({(x, y), (x, y + 1)})] = destructible
    return walls


def test_killbox_open_grid_interior_enemies_have_no_killboxes():
    # 11x11 open grid. For an enemy at the interior (5, 5), no bomb that
    # reaches them can prevent escape — they have all 4 cardinal moves open
    # and step out of any Chebyshev-2 blast in one move. (Corner cells in
    # an open grid intrinsically have killboxes — the corner enemy needs 5
    # moves to escape some blast regions; that's tested elsewhere.)
    prior = _Prior(11)
    killboxes = _compute_killboxes(prior)
    interior_pairs = [((bx, by), (5, 5))
                      for bx in range(3, 8) for by in range(3, 8)
                      if max(abs(bx - 5), abs(by - 5)) <= 2]
    for pair in interior_pairs:
        assert pair not in killboxes, f"interior pair {pair} unexpectedly killbox"


def test_killbox_corridor_dead_end_fires():
    # 7x7 grid. Corridor at y=3 of length 5 (cells (1..5, 3)) walled at BOTH
    # ends ((0,3)-(1,3) and (5,3)-(6,3)). Bomb at (3, 3) blasts the entire
    # corridor (Chebyshev 2). Enemy at (1, 3) is trapped — every reachable
    # tile is in the blast.
    walls = _corridor_walls(7, 3, destructible=False)
    walls[frozenset({(0, 3), (1, 3)})] = False     # west pin
    walls[frozenset({(5, 3), (6, 3)})] = False     # east pin
    prior = _Prior(7, walls)
    killboxes = _compute_killboxes(prior)
    assert ((3, 3), (1, 3)) in killboxes
    # Without the east pin, the enemy walks east to (6, 3) in 5 moves —
    # Chebyshev 3 from the bomb, OUTSIDE blast → escape, no killbox.
    walls2 = _corridor_walls(7, 3, destructible=False)
    walls2[frozenset({(0, 3), (1, 3)})] = False     # only west pin
    prior2 = _Prior(7, walls2)
    killboxes2 = _compute_killboxes(prior2)
    assert ((3, 3), (1, 3)) not in killboxes2


def test_killbox_excludes_pairs_where_enemy_can_escape():
    # 7x7 open grid. Enemy at (3, 3) has all 4 cardinal escapes — moves
    # one step in any direction land outside ANY single bomb's Chebyshev-2
    # blast region centered nearby.
    prior = _Prior(7)
    killboxes = _compute_killboxes(prior)
    # Enemy at (3, 3) with bomb at (2, 3) — blast = (0..4, 3) ∪ ... but
    # (3, 4) is outside the blast (Chebyshev 2 from (2,3) to (3,4) is
    # max(1, 1) = 1; in blast). Actually all 4 cardinal neighbours of (3,3)
    # are within Chebyshev-2 of (2,3). Let's pick a bomb further away.
    # Bomb at (3, 6) — blast at most reaches (3, 4) (radius 2). Enemy at
    # (3, 3) can move to (3, 2) — outside blast. Not a killbox.
    assert ((3, 6), (3, 3)) not in killboxes
    # Bomb at (0, 0) — blast doesn't include (3, 3) at all (Chebyshev 3).
    # So the pair is never even considered.
    assert ((0, 0), (3, 3)) not in killboxes


def test_killbox_destructible_walls_count_as_blockers():
    # Same sealed corridor, but every wall is destructible. The solver still
    # treats them as blockers for the escape BFS (conservative — a
    # destructible wall may stay intact through the fuse). The same killbox
    # should fire.
    walls = _corridor_walls(7, 3, destructible=True)
    walls[frozenset({(0, 3), (1, 3)})] = True      # destructible west pin
    walls[frozenset({(5, 3), (6, 3)})] = True      # destructible east pin
    prior = _Prior(7, walls)
    killboxes = _compute_killboxes(prior)
    assert ((3, 3), (1, 3)) in killboxes


def test_killbox_pair_count_is_finite_and_reasonable():
    # Sanity: on the real shipped MapPrior, the killbox set should be
    # non-empty and well under the grid_size^4 upper bound.
    prior = MapPrior.load()
    killboxes = _compute_killboxes(prior)
    assert isinstance(killboxes, set)
    upper_bound = prior.grid_size ** 4
    assert 0 < len(killboxes) < upper_bound
