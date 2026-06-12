"""Tests for striking through destructible walls (self-breach LOS).

A tile behind a destructible wall is a valid strike tile: bombs placed there
on consecutive ticks detonate in order, the first opens the wall (no damage
to the base — the env computes blast cells before destroying walls), the rest
gain line-of-sight and damage it. Walls that in-flight bombs (own or ally)
will open count as already gone, and strike prefers a short walk to a
direct-LOS tile over wasting opener bombs.
"""
from scripted.belief import Belief
from scripted.blast import breach_bombs_needed, replay_blasts
from scripted.danger import DangerMap
from scripted.geometry import PLACE_BOMB
from scripted.layers import strike
from scripted.pathfind import build_planner
from scripted.strategies import StrategyParams


def _base_prior(grid_size=7, wall_between=None, enemy_bases=((3, 3),)):
    class _Prior:
        pass
    p = _Prior()
    p.grid_size = grid_size
    p.wall_between = dict(wall_between or {})
    p.collectibles = {}
    p.enemy_bases = list(enemy_bases)
    p.our_base = (0, 0)
    return p


def _strike_belief(prior, loc, team_bombs=3):
    b = Belief()
    b.prior = prior
    b.destroyed_walls = set()
    b.dead_bases = set()
    b.enemy_base_health = {}
    b.ally_bombs = {}
    b.location = loc
    b.facing = 0
    b.team_bombs = team_bombs
    b.step = 10
    return b


_ONE_WALL = {frozenset({(3, 4), (3, 3)}): True}        # destructible
_TWO_WALLS = {frozenset({(3, 5), (3, 4)}): True,
              frozenset({(3, 4), (3, 3)}): True}
_HARD_WALL = {frozenset({(3, 4), (3, 3)}): False}      # indestructible
# Base sealed on three sides by indestructible walls; the destructible south
# wall is the only way a blast ever reaches it — no direct-LOS tile exists.
_SEALED = {frozenset({(3, 3), (3, 4)}): True,
           frozenset({(3, 3), (2, 3)}): False,
           frozenset({(3, 3), (4, 3)}): False,
           frozenset({(3, 3), (3, 2)}): False}


# --- breach_bombs_needed ------------------------------------------------- #

def test_direct_los_needs_no_openers():
    b = _strike_belief(_base_prior(), (3, 5))
    assert breach_bombs_needed((3, 5), (3, 3), b, 2) == 0


def test_one_destructible_wall_needs_one_opener():
    b = _strike_belief(_base_prior(wall_between=_ONE_WALL), (3, 5))
    assert breach_bombs_needed((3, 5), (3, 3), b, 2) == 1


def test_stacked_destructible_walls_need_two_openers():
    b = _strike_belief(_base_prior(wall_between=_TWO_WALLS), (3, 5))
    assert breach_bombs_needed((3, 5), (3, 3), b, 2) == 2
    assert breach_bombs_needed((3, 5), (3, 3), b, 1) is None   # over the cap


def test_indestructible_wall_never_opens():
    b = _strike_belief(_base_prior(wall_between=_HARD_WALL), (3, 5))
    assert breach_bombs_needed((3, 5), (3, 3), b, 2) is None


def test_out_of_blast_radius_is_none():
    b = _strike_belief(_base_prior(), (3, 6))                  # Chebyshev 3
    assert breach_bombs_needed((3, 6), (3, 3), b, 2) is None


# --- replay_blasts -------------------------------------------------------- #

def test_replay_counts_bombs_behind_the_opener():
    b = _strike_belief(_base_prior(wall_between=_ONE_WALL), (3, 5))
    bombs = [((3, 5), True), ((3, 5), True), ((3, 5), True)]
    # Bomb 1 opens the wall (no damage); bombs 2 and 3 hit.
    hits, opened = replay_blasts(bombs, (3, 3), b)
    assert hits == 2
    assert opened == {frozenset({(3, 4), (3, 3)})}


def test_replay_ally_openers_are_not_counted_as_hits():
    b = _strike_belief(_base_prior(wall_between=_ONE_WALL), (3, 5))
    bombs = [((3, 5), False), ((3, 5), True)]    # ally opener, then ours
    hits, _ = replay_blasts(bombs, (3, 3), b)
    assert hits == 1


def test_replay_matches_direct_count_without_walls():
    b = _strike_belief(_base_prior(), (3, 5))
    bombs = [((3, 5), True), ((0, 0), True)]     # second is far away
    hits, opened = replay_blasts(bombs, (3, 3), b)
    assert hits == 1
    assert opened == set()


# --- strike through a destructible wall ----------------------------------- #

def _run_strike(b, params=None):
    danger = DangerMap({}, b)
    planner = build_planner(b, danger)
    return strike(b, danger, planner, params or StrategyParams())


def test_strike_dumps_on_a_tile_behind_a_destructible_wall():
    # Sealed base: no direct-LOS tile exists anywhere, so the walk-over-dump
    # preference never fires — openers are the only way in.
    b = _strike_belief(_base_prior(wall_between=_SEALED), (3, 5), team_bombs=5)
    assert _run_strike(b) == PLACE_BOMB


def test_strike_keeps_dumping_while_the_opener_cooks():
    b = _strike_belief(_base_prior(wall_between=_SEALED), (3, 5), team_bombs=5)
    b.own_bombs = [((3, 5), 4)]                  # the opener, still in flight
    assert _run_strike(b) == PLACE_BOMB


def test_strike_stops_when_post_breach_bombs_finish_the_base():
    b = _strike_belief(_base_prior(wall_between=_SEALED), (3, 5), team_bombs=5)
    b.enemy_base_health = {(3, 3): 0.4}          # 40 HP left
    b.own_bombs = [((3, 5), 3), ((3, 5), 4), ((3, 5), 5)]   # opener + 2 hits
    assert _run_strike(b) is None


def test_strike_bombs_behind_a_wall_an_ally_bomb_opens():
    # The ally bomb detonates before anything we place now, so our tile is a
    # zero-opener strike tile despite the wall still standing in the belief.
    b = _strike_belief(_base_prior(wall_between=_SEALED), (3, 5), team_bombs=2)
    b.ally_bombs = {(3, 4): 2}
    assert _run_strike(b) == PLACE_BOMB


def test_strike_wont_start_a_dump_with_a_single_bomb():
    # One bomb can only open the wall, never damage — navigate instead.
    b = _strike_belief(_base_prior(wall_between=_ONE_WALL), (3, 5), team_bombs=1)
    action = _run_strike(b)
    assert action is not None
    assert action != PLACE_BOMB


def test_los_breach_zero_restores_direct_only_strikes():
    b = _strike_belief(_base_prior(wall_between=_ONE_WALL), (3, 5), team_bombs=5)
    action = _run_strike(b, StrategyParams(los_breach_max=0))
    assert action != PLACE_BOMB


def test_strike_still_navigates_around_an_indestructible_wall():
    b = _strike_belief(_base_prior(wall_between=_HARD_WALL), (3, 5), team_bombs=5)
    action = _run_strike(b)
    assert action is not None
    assert action != PLACE_BOMB


# --- walk to a direct tile instead of dumping ------------------------------ #

def test_strike_walks_to_a_nearby_direct_tile_instead_of_dumping():
    # One destructible wall: direct-LOS tiles flank the base a short walk
    # away, so strike moves there rather than spending an opener bomb.
    b = _strike_belief(_base_prior(wall_between=_ONE_WALL), (3, 5), team_bombs=5)
    action = _run_strike(b)
    assert action is not None
    assert action != PLACE_BOMB


def test_strike_dumps_when_enemies_block_every_direct_tile():
    b = _strike_belief(_base_prior(wall_between=_ONE_WALL), (3, 5), team_bombs=5)
    # Visible enemies camp every direct-LOS tile around the base (the route
    # check includes the destination tile itself).
    b.enemies = {(2, 2), (2, 3), (2, 4), (3, 2), (4, 2), (4, 3), (4, 4),
                 (1, 3), (3, 1), (1, 1), (5, 1), (1, 5), (5, 5), (5, 3),
                 (4, 5), (2, 5), (1, 2), (1, 4), (5, 2), (5, 4), (2, 1),
                 (4, 1), (3, 4)}
    assert _run_strike(b) == PLACE_BOMB


def test_strike_dumps_when_direct_tiles_are_too_far():
    b = _strike_belief(_base_prior(wall_between=_ONE_WALL), (3, 5), team_bombs=5)
    action = _run_strike(b, StrategyParams(direct_walk_max=1.0))
    assert action == PLACE_BOMB
