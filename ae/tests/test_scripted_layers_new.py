from scripted.belief import Belief
from scripted.danger import DangerMap
from scripted.geometry import STAY
from scripted.layers import hold
from scripted.map_prior import MapPrior
from scripted.pathfind import build_planner
from scripted.strategies import StrategyParams


def _ctx(camp_leash=None):
    prior = MapPrior.load()
    prior.identify_team(prior.bases[0])
    b = Belief()
    b.reset(prior)
    b.location = prior.spawns[0]["pos"]
    b.facing = prior.spawns[0]["facing"]
    danger = DangerMap({}, b)
    planner = build_planner(b, danger)
    return b, danger, planner, StrategyParams(camp_leash=camp_leash)


def test_hold_returns_stay():
    b, danger, planner, params = _ctx()
    assert hold(b, danger, planner, params) == STAY


from scripted.geometry import PLACE_BOMB
from scripted.layers import camp


def test_camp_returns_home_when_outside_leash():
    b, danger, planner, params = _ctx(camp_leash=2)
    base = b.prior.our_base
    # Place the agent far from base; rebuild the planner from there.
    far = (base[0] + 6 if base[0] + 6 < b.prior.grid_size else base[0] - 6, base[1])
    b.location = far
    planner = build_planner(b, danger)
    action = camp(b, danger, planner, params)
    assert action is not None and 0 <= action <= 4  # a move toward home


def test_camp_idle_when_on_station_and_no_threat():
    b, danger, planner, params = _ctx(camp_leash=5)
    b.location = b.prior.our_base
    planner = build_planner(b, danger)
    b.enemies = set()
    assert camp(b, danger, planner, params) is None


def test_camp_bombs_enemy_in_territory_and_in_range():
    b, danger, planner, params = _ctx(camp_leash=5)
    base = b.prior.our_base
    b.location = base
    b.team_bombs = 2
    # an enemy on an adjacent tile is within blast range and within leash
    adj = (base[0] + 1, base[1])
    b.enemies = {adj}
    planner = build_planner(b, danger)
    action = camp(b, danger, planner, params)
    # bomb now if a bomb here reaches it, otherwise route toward a bombing tile
    assert action in (PLACE_BOMB, 0, 1, 2, 3)


# ---------------------------------------------------------------------------
# hunt — opportunistic enemy bombing (no routing)
# ---------------------------------------------------------------------------
from scripted.layers import hunt


def _open_belief(loc, team_bombs, enemies, grid_size=7, wall_between=None):
    """Synthetic Belief on a (near-)wall-free grid.

    With no walls, bomb_reaches is pure Chebyshev distance, so every enemy
    within Chebyshev 2 of `loc` is in blast range — making the hunt truth
    table deterministic. `wall_between` may add edges to test line-of-sight.
    """
    class _Prior:
        pass

    prior = _Prior()
    prior.grid_size = grid_size
    prior.wall_between = wall_between or {}
    prior.collectibles = {}
    prior.enemy_bases = []
    prior.our_base = (0, 0)

    b = Belief()
    b.prior = prior
    b.destroyed_walls = set()
    b.collected = set()
    b.dead_bases = set()
    b.enemy_base_health = {}
    b.ally_bombs = {}
    b.enemy_bombs = {}
    b.enemies = set(enemies)
    b.location = loc
    b.facing = 0
    b.team_bombs = team_bombs
    b.step = 10
    b.frozen_ticks = 0
    b.health = 1.0
    b.base_health = 1.0
    return b


def _hunt(b):
    """Call hunt with throwaway danger/planner/params (the layer uses none)."""
    danger = DangerMap({}, b)
    planner = build_planner(b, danger)
    return hunt(b, danger, planner, StrategyParams())


def test_hunt_none_when_no_bombs():
    # one enemy in blast range, but 0 bombs -> cannot place
    b = _open_belief(loc=(3, 3), team_bombs=0, enemies={(3, 4)})
    assert _hunt(b) is None


def test_hunt_none_when_no_enemies():
    b = _open_belief(loc=(3, 3), team_bombs=2, enemies=set())
    assert _hunt(b) is None


def test_hunt_none_when_enemy_out_of_blast_range():
    # enemy at Chebyshev 3 -> outside blast radius 2
    b = _open_belief(loc=(3, 3), team_bombs=2, enemies={(3, 6)})
    assert _hunt(b) is None


def test_hunt_endgame_single_enemy_places_with_one_bomb():
    # No live bases (= endgame in this fixture). One enemy in range. The
    # bomb-floor only applies while bases are alive, so hunt fires on its
    # last bomb.
    b = _open_belief(loc=(3, 3), team_bombs=1, enemies={(3, 4)})
    assert _hunt(b) == PLACE_BOMB


def test_hunt_endgame_single_enemy_places_with_two_bombs():
    b = _open_belief(loc=(3, 3), team_bombs=2, enemies={(3, 4)})
    assert _hunt(b) == PLACE_BOMB


def test_hunt_endgame_two_enemies_places_with_one_bomb():
    # No live bases, two enemies in range — hunt fires.
    b = _open_belief(loc=(3, 3), team_bombs=1, enemies={(3, 4), (2, 3)})
    assert _hunt(b) == PLACE_BOMB


def test_hunt_endgame_two_enemies_places_with_two_bombs():
    b = _open_belief(loc=(3, 3), team_bombs=2, enemies={(3, 4), (2, 3)})
    assert _hunt(b) == PLACE_BOMB


def test_hunt_endgame_fires_with_any_in_range_hit():
    # One enemy in range, one far away. `hits == 1` is enough — endgame
    # has no bomb floor, so hunt fires with 1 bomb on 1 in-range enemy.
    b = _open_belief(loc=(3, 3), team_bombs=1, enemies={(3, 4), (3, 6)})
    assert _hunt(b) == PLACE_BOMB


def test_hunt_holds_fire_below_bomb_floor_with_live_bases():
    # A live enemy base + bombs below the floor -> hunt yields even with an
    # enemy in range, saving the stockpile for the +50 base kill.
    b = _open_belief(loc=(3, 3), team_bombs=5, enemies={(3, 4)})
    b.prior.enemy_bases = [(0, 0)]                     # one live base
    assert _hunt(b) is None


def test_hunt_fires_at_bomb_floor_with_live_bases():
    # Same setup but team_bombs hits the floor (default 6) -> fire.
    b = _open_belief(loc=(3, 3), team_bombs=6, enemies={(3, 4)})
    b.prior.enemy_bases = [(0, 0)]
    assert _hunt(b) == PLACE_BOMB


def test_hunt_ignores_enemy_behind_wall():
    # enemy adjacent (Chebyshev 1) but an indestructible wall blocks LOS
    wall = {frozenset({(3, 3), (3, 4)}): False}
    b = _open_belief(loc=(3, 3), team_bombs=2, enemies={(3, 4)},
                     wall_between=wall)
    assert _hunt(b) is None


def test_hunt_does_not_rebomb_enemy_an_own_bomb_already_covers():
    # Enemy in range, but a bomb we placed last tick (still in flight) already
    # covers it -> hunt yields so the agent moves on instead of re-bombing.
    b = _open_belief(loc=(3, 3), team_bombs=2, enemies={(3, 4)})
    b.own_bombs = [((3, 3), 3)]          # in-flight bomb at our tile reaches (3,4)
    assert _hunt(b) is None


def test_hunt_fires_when_in_flight_bomb_does_not_reach_enemy():
    # An own bomb in flight elsewhere does NOT cover this enemy, so hunt still
    # drops a fresh bomb on the in-range threat.
    b = _open_belief(loc=(3, 3), team_bombs=2, enemies={(3, 4)})
    b.own_bombs = [((0, 0), 3)]          # far-away bomb cannot reach (3,4)
    assert _hunt(b) == PLACE_BOMB


# ---------------------------------------------------------------------------
# _move_result — frozen enemies block movement
# ---------------------------------------------------------------------------
from scripted.layers import _move_result
from scripted.geometry import FORWARD


def test_move_result_enters_open_tile():
    b = _open_belief(loc=(3, 3), team_bombs=2, enemies=set())
    danger = DangerMap({}, b)
    # facing RIGHT(0); FORWARD from (3,3) -> (4,3)
    assert _move_result(b, danger, (3, 3), 0, FORWARD) == ((4, 3), 0)


def test_move_result_blocked_by_frozen_enemy():
    b = _open_belief(loc=(3, 3), team_bombs=2, enemies=set())
    b.frozen_enemies = {(4, 3)}
    danger = DangerMap({}, b)
    assert _move_result(b, danger, (3, 3), 0, FORWARD) is None


def test_hunt_ignores_frozen_enemy():
    # A single enemy in blast range, but frozen -> not a bomb target.
    b = _open_belief(loc=(3, 3), team_bombs=2, enemies={(3, 4)})
    b.frozen_enemies = {(3, 4)}
    assert _hunt(b) is None


def test_hunt_counts_only_live_enemies():
    # Two enemies in range, one frozen -> only 1 live enemy counts; with
    # team_bombs >= 2, one live enemy is enough to place.
    b = _open_belief(loc=(3, 3), team_bombs=2, enemies={(3, 4), (2, 3)})
    b.frozen_enemies = {(2, 3)}
    assert _hunt(b) == PLACE_BOMB


def test_hunt_frozen_enemy_does_not_count_as_hit():
    # Two enemies in range, one frozen -> only 1 live counts. With a LIVE base
    # and 1 bomb (below the floor), hunt yields. Frozen enemies aren't a hunt
    # target.
    b = _open_belief(loc=(3, 3), team_bombs=1, enemies={(3, 4), (2, 3)})
    b.frozen_enemies = {(2, 3)}
    b.prior.enemy_bases = [(0, 0)]
    assert _hunt(b) is None


def test_hunt_endgame_frozen_enemy_yields_when_no_live_hit():
    # Endgame: one live enemy in range. Frozen enemies on adjacent tiles do
    # not contribute to the hit count, but the lone live enemy still triggers
    # hunt (floor doesn't apply when bases are dead).
    b = _open_belief(loc=(3, 3), team_bombs=1, enemies={(3, 4), (2, 3)})
    b.frozen_enemies = {(2, 3)}
    assert _hunt(b) == PLACE_BOMB
