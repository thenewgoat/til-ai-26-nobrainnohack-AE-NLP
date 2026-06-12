from scripted.belief import Belief
from scripted.danger import DangerMap
from scripted.geometry import BACKWARD, FORWARD, PLACE_BOMB
from scripted.pathfind import BOMB_TIMER, INF, build_planner


class _Prior:
    """Minimal hand-built prior: open grid, no collectibles, no walls."""

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


def test_open_grid_distances():
    b = _belief((0, 0), 0)                       # facing RIGHT, open 5x5
    p = build_planner(b, DangerMap({}, b))
    assert p.dist_to((0, 0)) == 0
    assert p.dist_to((1, 0)) == 1                # one FORWARD
    assert p.dist_to((4, 0)) == 4


def test_first_action_forward_and_backward():
    fwd_b = _belief((0, 0), 0)
    fwd = build_planner(fwd_b, DangerMap({}, fwd_b))
    assert fwd.first_action((2, 0)) == FORWARD
    back_b = _belief((2, 0), 0)
    back = build_planner(back_b, DangerMap({}, back_b))
    assert back.first_action((1, 0)) == BACKWARD  # cheaper than two turns


def test_steps_to_equals_dist_to():
    b = _belief((0, 0), 0)
    p = build_planner(b, DangerMap({}, b))
    assert p.steps_to((3, 0)) == p.dist_to((3, 0)) == 3


def test_distances_stay_exact_past_the_horizon():
    # 4 FORWARD + 1 turn + 4 FORWARD = 9 ticks, well past T_MAX — the phase
    # collapse must not corrupt the true distance.
    b = _belief((0, 0), 0)
    p = build_planner(b, DangerMap({}, b))
    assert p.dist_to((4, 4)) == 9


def test_indestructible_wall_forces_a_detour():
    b = _belief((0, 0), 0)
    b.prior.wall_between = {frozenset({(0, 0), (1, 0)}): False}
    p = build_planner(b, DangerMap({}, b))
    # Detour (0,0)->(0,1)->(1,1)->(1,0): 3 moves + 3 turns from facing RIGHT.
    assert p.dist_to((1, 0)) == 6


def test_frozen_enemy_tile_is_impassable():
    b = _belief((0, 0), 0)
    b.frozen_enemies = {(1, 0)}
    p = build_planner(b, DangerMap({}, b))
    assert p.dist_to((1, 0)) == INF


def test_destructible_wall_impassable_without_a_bomb():
    # (1,0) is reachable only through the destructible wall from (0,0).
    b = _belief((0, 0), 0)
    b.prior.wall_between = {
        frozenset({(0, 0), (1, 0)}): True,        # the only door — destructible
        frozenset({(1, 0), (2, 0)}): False,
        frozenset({(1, 0), (1, 1)}): False,
    }
    p = build_planner(b, DangerMap({}, b))
    assert p.dist_to((1, 0)) == INF               # no bomb -> never opens


def test_destructible_wall_opens_at_bomb_detonation_tick():
    b = _belief((0, 0), 0)
    b.prior.wall_between = {
        frozenset({(0, 0), (1, 0)}): True,
        frozenset({(1, 0), (2, 0)}): False,
        frozenset({(1, 0), (1, 1)}): False,
    }
    b.ally_bombs = {(0, 0): 3}                    # blast opens the wall at tick 3
    p = build_planner(b, DangerMap({}, b))
    # Wall opens during DETONATE at tick 3, which runs AFTER the MOVE at tick
    # 3 — so the earliest the agent can cross is tick 4.
    assert p.dist_to((1, 0)) == 4


def test_enemy_blast_forbids_a_cell_only_on_its_detonation_tick():
    b = _belief((0, 3), 0, grid_size=7)           # facing RIGHT, open 7x7
    # Enemy bomb at (3,3) detonates at tick 1; its blast covers (1,3) but not
    # the agent's (0,3) (Chebyshev 3 away).
    blocked = build_planner(b, DangerMap({(3, 3): 1}, b))
    assert blocked.dist_to((1, 3)) == 2           # STAY past the blast, then FORWARD
    clear = build_planner(b, DangerMap({}, b))
    assert clear.dist_to((1, 3)) == 1             # no bomb -> straight FORWARD


def test_place_bomb_first_forces_place_then_opens_loc_walls():
    b = _belief((0, 0), 0)
    b.prior.wall_between = {
        frozenset({(0, 0), (1, 0)}): True,
        frozenset({(1, 0), (2, 0)}): False,
        frozenset({(1, 0), (1, 1)}): False,
    }
    p = build_planner(b, DangerMap({}, b), place_bomb_first=True)
    assert p.first_action((1, 0)) == PLACE_BOMB   # forced opening move
    # The hypothetical bomb opens (0,0)|(1,0) at tick `1 + BOMB_TIMER`; the
    # wall opens AFTER MOVE at that tick, so the earliest cross is one tick
    # later.
    assert p.dist_to((1, 0)) == 2 + BOMB_TIMER
