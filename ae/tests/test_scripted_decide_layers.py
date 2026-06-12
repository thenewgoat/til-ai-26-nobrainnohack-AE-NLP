from scripted.belief import Belief
from scripted.danger import DangerMap
from scripted.layers import _openness, default, forage, hunt, strike, survive, sweep
from scripted.strategies import StrategyParams
from scripted.map_prior import MapPrior
from scripted.pathfind import build_planner


def _belief(loc, facing=0, team_bombs=3, bombs=None):
    m = MapPrior.load()
    m.identify_team((13, 9))                 # team 0
    b = Belief()
    b.reset(m)
    b.location = loc
    b.facing = facing
    b.team_bombs = team_bombs
    b.enemy_bombs = bombs or {}
    b.step = 10
    return b


def test_survive_returns_none_when_safe():
    b = _belief((5, 5))
    danger = DangerMap({}, b)
    p = build_planner(b, danger)
    assert survive(b, danger, p, StrategyParams()) is None


def test_survive_moves_off_a_dangerous_cell():
    # Bomb at (5,5) with timer=2 covers Chebyshev-2 radius at the exact
    # detonation tick. Every cell reachable in 1 step is within the blast;
    # every path out requires passing through the blast at tick 2. The
    # time-aware planner hard-blocks lethal (cell, tick) pairs, so no
    # cell outside the blast is reachable. Tier-2 selects (5,5) itself
    # (dist 0) as the "least-bad" candidate; survive correctly yields None
    # so objective layers can act (the agent will take the hit regardless).
    b = _belief((5, 5))
    danger = DangerMap({(5, 5): 2}, b)      # standing in a blast zone
    p = build_planner(b, danger)
    action = survive(b, danger, p, StrategyParams())
    assert action is None                    # inescapable blast — Tier 2 yields


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


def test_strike_places_bomb_in_range_of_enemy_base():
    """Strike bombs from within blast range + line-of-sight of a base, without
    having to stand on it."""
    b = _strike_belief(_base_prior(), (3, 5))   # 2 tiles from base (3,3), clear
    danger = DangerMap({}, b)
    p = build_planner(b, danger)
    assert strike(b, danger, p, StrategyParams()) == 5            # PLACE_BOMB


def test_strike_does_not_bomb_base_through_a_wall():
    """A base within blast range but behind a wall is not bomb-reachable —
    Strike navigates instead of wasting a bomb."""
    prior = _base_prior(wall_between={frozenset({(3, 5), (3, 4)}): False})
    b = _strike_belief(prior, (3, 5))
    danger = DangerMap({}, b)
    p = build_planner(b, danger)
    assert strike(b, danger, p, StrategyParams()) != 5            # cannot hit the base from here


def test_strike_skips_a_base_believed_destroyed():
    """A base believed destroyed is not targeted — with no live base, Strike
    yields to the next cascade layer."""
    b = _strike_belief(_base_prior(), (3, 5))
    b.dead_bases = {(3, 3)}                      # the only base is destroyed
    danger = DangerMap({}, b)
    p = build_planner(b, danger)
    assert strike(b, danger, p, StrategyParams()) is None


def test_strike_gives_up_after_three_dead_enemy_bases():
    """Three bases observed dead -> strike yields even with a live base in
    bomb range and bombs in hand."""
    prior = _base_prior(enemy_bases=((3, 3), (5, 5), (1, 1), (5, 1)))
    b = _strike_belief(prior, (3, 5))            # in range of live base (3,3)
    b.base_health = 100.0                        # our base alive
    b.dead_bases = {(5, 5), (1, 1), (5, 1)}      # 3 enemy bases down
    danger = DangerMap({}, b)
    p = build_planner(b, danger)
    assert strike(b, danger, p, StrategyParams()) is None


def test_strike_counts_our_own_dead_base_toward_the_cap():
    prior = _base_prior(enemy_bases=((3, 3), (5, 5), (1, 1)))
    b = _strike_belief(prior, (3, 5))
    b.base_health = 0.0                          # our base destroyed
    b.dead_bases = {(5, 5), (1, 1)}              # + 2 enemy bases = 3 dead
    danger = DangerMap({}, b)
    p = build_planner(b, danger)
    assert strike(b, danger, p, StrategyParams()) is None


def test_strike_keeps_attacking_below_the_dead_base_cap():
    prior = _base_prior(enemy_bases=((3, 3), (5, 5), (1, 1)))
    b = _strike_belief(prior, (3, 5))
    b.base_health = 100.0
    b.dead_bases = {(5, 5)}                      # only 2 dead incl. nobody else
    danger = DangerMap({}, b)
    p = build_planner(b, danger)
    assert strike(b, danger, p, StrategyParams()) == 5        # PLACE_BOMB


def test_strike_dead_base_cap_zero_disables_the_give_up():
    prior = _base_prior(enemy_bases=((3, 3), (5, 5), (1, 1), (5, 1)))
    b = _strike_belief(prior, (3, 5))
    b.base_health = 0.0
    b.dead_bases = {(5, 5), (1, 1), (5, 1)}      # 4 dead incl. our own
    danger = DangerMap({}, b)
    p = build_planner(b, danger)
    params = StrategyParams(strike_dead_bases_cap=0)
    assert strike(b, danger, p, params) == 5                  # PLACE_BOMB


def test_strike_stops_bombing_a_doomed_base():
    """When the agent's own bombs already in flight will finish a base, Strike
    does not waste another bomb on it."""
    b = _strike_belief(_base_prior(), (3, 3))    # standing on the base
    b.enemy_base_health = {(3, 3): 0.4}          # 40 HP left
    b.own_bombs = [((3, 3), 3), ((3, 3), 3)]     # 2 own bombs reach it -> 40 dmg
    danger = DangerMap({}, b)
    p = build_planner(b, danger)
    assert strike(b, danger, p, StrategyParams()) is None          # base doomed


def _forage_belief(collectibles, loc, facing=0, enemy_bases=(), dead_bases=()):
    prior = _base_prior(grid_size=7, enemy_bases=list(enemy_bases))
    prior.collectibles = dict(collectibles)
    b = Belief()
    b.prior = prior
    b.destroyed_walls = set()
    b.dead_bases = set(dead_bases)
    b.collected = set()
    b.enemy_base_health = {}
    b.ally_bombs = {}
    b.location = loc
    b.facing = facing
    b.step = 10
    return b


def test_forage_inactive_while_a_base_lives():
    b = _forage_belief({(3, 3): 5.0}, (2, 2), enemy_bases=[(5, 5)])
    danger = DangerMap({}, b)
    p = build_planner(b, danger)
    assert forage(b, danger, p, StrategyParams()) is None          # a base still stands


def test_forage_steps_onto_a_collectible():
    # All bases gone; a collectible one tile ahead (facing RIGHT).
    b = _forage_belief({(3, 2): 5.0}, (2, 2), facing=0)
    danger = DangerMap({}, b)
    p = build_planner(b, danger)
    assert forage(b, danger, p, StrategyParams()) == 0             # FORWARD onto the loot


def test_forage_prefers_the_richer_two_move_path():
    # Facing RIGHT at (2,2): one low-value tile ahead, two high-value behind.
    collectibles = {(3, 2): 1.0, (1, 2): 5.0, (0, 2): 5.0}
    b = _forage_belief(collectibles, (2, 2), facing=0)
    danger = DangerMap({}, b)
    p = build_planner(b, danger)
    assert forage(b, danger, p, StrategyParams()) == 1             # BACKWARD — 10 value vs 1


def test_forage_yields_when_no_collectible_in_reach():
    b = _forage_belief({}, (2, 2))               # endgame, nothing to collect
    danger = DangerMap({}, b)
    p = build_planner(b, danger)
    assert forage(b, danger, p, StrategyParams()) is None          # falls through to sweep


def test_strike_holds_fire_with_no_bombs():
    b = _belief((9, 11), team_bombs=0)
    danger = DangerMap({}, b)
    p = build_planner(b, danger)
    assert strike(b, danger, p, StrategyParams()) != 5


def test_strike_places_bomb_without_escape_route():
    """Our own bomb is harmless to us, so Strike places a bomb whenever in
    range of an enemy base — no escape route required."""
    class _Prior:
        grid_size = 5            # on a 5x5 grid no tile is Chebyshev > 2 from
        wall_between = {}        # the centre, so an escape check could never pass
        collectibles = {}
        enemy_bases = [(2, 2)]
        our_base = (0, 0)

    b = Belief()
    b.prior = _Prior()
    b.destroyed_walls = set()
    b.location = (2, 2)          # standing on the enemy base
    b.facing = 0
    b.team_bombs = 3
    b.step = 10
    danger = DangerMap({}, b)
    p = build_planner(b, danger)
    assert strike(b, danger, p, StrategyParams()) == 5     # PLACE_BOMB — no escape needed


def test_openness_counts_open_area():
    # Fully open 5x5: BFS radius 4 from the centre reaches every cell.
    b = _strike_belief(_base_prior(grid_size=5, wall_between={}), (2, 2))
    danger = DangerMap({}, b)
    assert _openness(b, danger, (2, 2), radius=4) == 25


def test_openness_isolated_pocket_scores_one():
    # Wall (0,0) off from both its neighbours -> a one-cell pocket.
    prior = _base_prior(grid_size=5, wall_between={
        frozenset({(0, 0), (1, 0)}): False,
        frozenset({(0, 0), (0, 1)}): False,
    })
    b = _strike_belief(prior, (0, 0))
    danger = DangerMap({}, b)
    assert _openness(b, danger, (0, 0), radius=4) == 1        # walled-in pocket
    assert _openness(b, danger, (2, 2), radius=4) == 24       # everything but the pocket


# --- survive Tier 1: full escape, openness bias, opportunistic bomb drop ---

def test_survive_tier1_routes_to_safety():
    """A live base in the prior + 1 bomb (< bomb_drop_min=6) — survive
    hoards the bomb and routes toward safety with a move. Agent at (3, 2)
    is Chebyshev 3 from the base at (0, 4), so neither Case B nor Case A
    of the strike caveat triggers (no safe hit-tile is reachable from
    inside the blast at (1, 2))."""
    b = _strike_belief(_base_prior(grid_size=5, enemy_bases=((0, 4),)),
                       (3, 2), team_bombs=1)
    danger = DangerMap({(1, 2): 4}, b)            # blast covers x0..3, agent at (3,2)
    p = build_planner(b, danger)
    assert survive(b, danger, p, StrategyParams()) == 0     # FORWARD toward x=4 safety


def test_survive_tier1_drops_surplus_bomb_in_endgame():
    """No live base + 3 bombs in hand + chosen safe cell 1 step away ->
    survive's bomb_drop floor collapses to 1 in the endgame, so the agent
    drops the bomb first (1 + 1 place + 1 buffer <= 4 ticks)."""
    b = _strike_belief(_base_prior(grid_size=5, enemy_bases=()), (3, 2),
                       team_bombs=3)
    danger = DangerMap({(1, 2): 4}, b)
    p = build_planner(b, danger)
    assert survive(b, danger, p, StrategyParams()) == 5     # PLACE_BOMB


def test_survive_tier1_holds_fire_below_bomb_drop_min_while_bases_live():
    """With a live enemy base AND 3 bombs in hand (below the default
    `bomb_drop_min=6`), survive does NOT spend a bomb while fleeing — bombs
    are hoarded for the base offense. Agent placed at (0, 2) so it is
    Chebyshev > 2 from the base at (0, 4) for hit-tile purposes (avoiding
    the strike caveat)."""
    b = _strike_belief(_base_prior(grid_size=5, enemy_bases=((0, 4),)),
                       (0, 2), team_bombs=3)
    danger = DangerMap({(1, 2): 4}, b)
    p = build_planner(b, danger)
    action = survive(b, danger, p, StrategyParams())
    # Bombs (3) < default bomb_drop_min (6) AND base alive → no bomb drop.
    assert action != 5                          # not PLACE_BOMB


def test_survive_tier1_no_bomb_without_spare_time():
    """3 bombs in hand but the agent is at the blast centre -> the nearest safe
    cell is 3 steps away, 3 + 1 + 1 > 4, so no bomb: just flee. No live base."""
    b = _strike_belief(_base_prior(grid_size=5, enemy_bases=()), (1, 2),
                       team_bombs=3)
    danger = DangerMap({(1, 2): 4}, b)
    p = build_planner(b, danger)
    assert survive(b, danger, p, StrategyParams()) == 0     # FORWARD, no PLACE_BOMB


def test_survive_avoids_dead_end_pocket():
    """Four equidistant safe cells; (0,3) is a walled-off pocket. survive's
    openness bias must pick an open cell, never the pocket (BACKWARD heads
    to the pocket from (3,3) facing RIGHT). No live base, so strike caveat
    doesn't preempt the openness logic."""
    prior = _base_prior(grid_size=7, enemy_bases=(), wall_between={
        frozenset({(0, 3), (0, 2)}): False,
        frozenset({(0, 3), (0, 4)}): False,
    })
    b = _strike_belief(prior, (3, 3), team_bombs=3)
    danger = DangerMap({(3, 3): 4}, b)
    p = build_planner(b, danger)
    action = survive(b, danger, p, StrategyParams())
    assert action in (0, 2, 3)        # an open cell; never BACKWARD(1) into the pocket


# --- survive Tier 2: least-bad fallback, no STAY ---

def test_survive_tier2_moves_from_double_to_single():
    """Whole 5x5 grid is dangerous; row y=0 is single-damage, the rest double.
    From (2,1) facing UP, survive steps FORWARD onto the single-damage (2,0)."""
    b = _strike_belief(_base_prior(grid_size=5), (2, 1), team_bombs=3)
    b.facing = 3                                  # facing UP
    danger = DangerMap({(2, 2): 2, (2, 3): 2}, b)
    p = build_planner(b, danger)
    assert survive(b, danger, p, StrategyParams()) == 0     # FORWARD to lower overlap


def test_survive_tier2_yields_when_least_bad():
    """Whole grid uniformly single-damage; the agent is already on a least-bad
    cell -> survive yields (None) so the objective layers drive. Never STAY."""
    b = _strike_belief(_base_prior(grid_size=5), (0, 0), team_bombs=3)
    danger = DangerMap({(2, 2): 2}, b)            # one central bomb covers all 5x5
    p = build_planner(b, danger)
    assert survive(b, danger, p, StrategyParams()) is None


# --- survive strike caveat (Case B place-and-escape, Case A safe-tile) -----

def test_survive_strike_caveat_case_b_places_at_loc():
    """Agent at a hit-tile of a live base AND in danger AND a safe escape
    exists in deadline-1 phases — Case B fires, return PLACE_BOMB."""
    b = _strike_belief(_base_prior(grid_size=7, enemy_bases=((3, 3),)),
                       (3, 5), team_bombs=1)
    b.enemy_bombs = {(3, 5): 4}                   # timer = lethal phase 4 (post-shift)
    danger = DangerMap(b.enemy_bombs, b)
    p = build_planner(b, danger)
    # Loc (3, 5) reaches base (3, 3) at Chebyshev 2 — a hit-tile. Safe cell
    # at (6, 5) reachable in 3 moves (< deadline - 1 = 3 phases). Case B fires.
    assert survive(b, danger, p, StrategyParams()) == 5     # PLACE_BOMB


def test_survive_strike_caveat_case_a_routes_to_safe_hit_tile():
    """Agent in danger, loc is NOT a hit-tile, but a SAFE hit-tile is reachable
    by the deadline — Case A fires, route toward it."""
    b = _strike_belief(_base_prior(grid_size=9, enemy_bases=((6, 4),)),
                       (2, 4), team_bombs=1)
    b.enemy_bombs = {(2, 4): 5}                   # lethal phase 5
    danger = DangerMap(b.enemy_bombs, b)
    p = build_planner(b, danger)
    # Loc (2, 4) is Chebyshev 4 from base (6, 4) — NOT a hit-tile. A safe
    # hit-tile (7, 4) sits 5 cardinal moves away; planner.dist == 5 == deadline.
    # Case A picks it; first action is FORWARD (= 0).
    assert survive(b, danger, p, StrategyParams()) == 0


def test_survive_strike_caveat_yields_when_no_live_bases():
    """No live enemy base — caveat skips, normal tier-1 runs. With no live
    bases the `bomb_drop_min` floor collapses to 1, so a single bomb in
    hand triggers the opportunistic drop; assert the action is a strike or
    flee, just NOT a STAY (= 4)."""
    b = _strike_belief(_base_prior(grid_size=5, enemy_bases=()),
                       (3, 2), team_bombs=1)
    b.enemy_bombs = {(1, 2): 4}
    danger = DangerMap(b.enemy_bombs, b)
    p = build_planner(b, danger)
    action = survive(b, danger, p, StrategyParams())
    assert action is not None and action != 4


def test_survive_strike_caveat_yields_when_own_bombs_already_finish():
    """Loc IS a hit-tile but our in-flight bombs already cover the base's HP —
    no point bombing again. Caveat skips, normal tier-1 flee runs."""
    b = _strike_belief(_base_prior(grid_size=7, enemy_bases=((3, 3),)),
                       (3, 5), team_bombs=1)
    b.enemy_base_health = {(3, 3): 0.2}            # 20 HP observed
    b.own_bombs = [((3, 3), 3)]                    # 1 in-flight hit = 20 damage
    b.enemy_bombs = {(3, 5): 4}
    danger = DangerMap(b.enemy_bombs, b)
    p = build_planner(b, danger)
    # 20 damage >= 20 HP — base will already die. Caveat yields → tier-1 flee.
    action = survive(b, danger, p, StrategyParams())
    assert action != 5                              # NOT PLACE_BOMB


# --- strike: two-scenario wall breach -------------------------------------

def _walled_base_prior():
    """7x7 prior: enemy base at (3,3) sealed behind one destructible wall at
    (2,3)|(3,3); its other three sides are indestructible. The only way to a
    tile that can hit the base is to bomb that destructible wall open."""
    return _base_prior(grid_size=7, enemy_bases=((3, 3),), wall_between={
        frozenset({(3, 3), (2, 3)}): True,    # destructible — the only breach point
        frozenset({(3, 3), (4, 3)}): False,
        frozenset({(3, 3), (3, 2)}): False,
        frozenset({(3, 3), (3, 4)}): False,
    })


def test_strike_breaches_a_wall_to_reach_a_base():
    """With enough bombs to clear `breach_min_bombs=6` and a base reachable
    only by breaching, strike drops a bomb now — scenario B reaches a
    hit-tile, scenario A never does."""
    b = _strike_belief(_walled_base_prior(), (1, 3), team_bombs=6)
    danger = DangerMap({}, b)
    p = build_planner(b, danger)
    assert strike(b, danger, p, StrategyParams()) == 5            # PLACE_BOMB


def test_strike_does_not_breach_with_only_one_bomb():
    """team_bombs below breach_min_bombs (2) -> strike never spends the bomb on
    a breach; with no other option it yields."""
    b = _strike_belief(_walled_base_prior(), (1, 3), team_bombs=1)
    danger = DangerMap({}, b)
    p = build_planner(b, danger)
    # Only one bomb: no breach, and the walled base is otherwise unreachable,
    # so strike yields to the next cascade layer.
    assert strike(b, danger, p, StrategyParams()) is None


def test_strike_bombs_through_a_wall_an_ally_bomb_is_opening():
    """An ally bomb already opening the breach wall turns the agent's tile
    into a free strike tile: every bomb in the air detonates before one placed
    now, so the wall is gone by the time our bomb blows — it damages the base
    directly, no opener wasted."""
    from scripted.geometry import PLACE_BOMB as _PLACE_BOMB_ACT
    b = _strike_belief(_walled_base_prior(), (1, 3), team_bombs=3)
    b.ally_bombs = {(2, 3): 2}                 # blast opens (2,3)|(3,3) at tick 2
    danger = DangerMap({}, b)
    p = build_planner(b, danger)
    action = strike(b, danger, p, StrategyParams())
    assert action == _PLACE_BOMB_ACT


# --- effective HP ----------------------------------------------------------

def test_effective_hp_subtracts_in_flight_own_bombs():
    from scripted.layers import _effective_hp
    b = _strike_belief(_base_prior(), (3, 3))
    b.enemy_base_health = {(3, 3): 0.8}          # 80 HP observed
    b.own_bombs = [((3, 3), 3)]                  # one own bomb reaches it
    assert _effective_hp(b, (3, 3)) == 60.0      # 80 - 20


def test_effective_hp_counts_stacked_bombs():
    from scripted.layers import _effective_hp
    b = _strike_belief(_base_prior(), (3, 3))
    b.enemy_base_health = {(3, 3): 0.8}
    b.own_bombs = [((3, 3), 3), ((3, 3), 2)]     # two bombs stacked on one tile
    assert _effective_hp(b, (3, 3)) == 40.0      # 80 - 40 — both counted


def test_effective_hp_floors_at_zero():
    from scripted.layers import _effective_hp
    b = _strike_belief(_base_prior(), (3, 3))
    b.enemy_base_health = {(3, 3): 0.6}          # 60 HP
    b.own_bombs = [((3, 3), 3)] * 5              # 100 dmg in flight
    assert _effective_hp(b, (3, 3)) == 0.0       # floored, never negative


def test_base_doomed_true_exactly_when_effective_hp_zero():
    from scripted.layers import _base_doomed
    b = _strike_belief(_base_prior(), (3, 3))
    b.enemy_base_health = {(3, 3): 0.4}
    b.own_bombs = [((3, 3), 3)]                  # 20 dmg -> 20 HP left
    assert _base_doomed(b, (3, 3)) is False
    b.own_bombs = [((3, 3), 3), ((3, 3), 3)]     # 40 dmg -> 0 HP
    assert _base_doomed(b, (3, 3)) is True


# --- _target_base ----------------------------------------------------------

def test_target_base_prefers_the_more_damaged_base():
    from scripted.layers import _target_base
    prior = _base_prior(grid_size=7, enemy_bases=((1, 1), (5, 5)))
    b = _strike_belief(prior, (3, 3))
    b.enemy_base_health = {(1, 1): 1.0, (5, 5): 0.4}   # (5,5) crippled
    danger = DangerMap({}, b)
    p = build_planner(b, danger)
    base, eff_hp, bombs_needed = _target_base(b, p, StrategyParams())
    assert base == (5, 5)                # fewer bombs to kill wins
    assert eff_hp == 40.0
    assert bombs_needed == 2


def test_target_base_none_when_no_live_base():
    from scripted.layers import _target_base
    prior = _base_prior(grid_size=7, enemy_bases=((1, 1),))
    b = _strike_belief(prior, (3, 3))
    b.dead_bases = {(1, 1)}
    danger = DangerMap({}, b)
    p = build_planner(b, danger)
    assert _target_base(b, p, StrategyParams()) is None


def test_target_base_blend_breaks_ties_by_distance():
    from scripted.layers import _target_base
    # two equal-HP bases; the nearer one wins on the travel term.
    prior = _base_prior(grid_size=7, enemy_bases=((1, 3), (6, 3)))
    b = _strike_belief(prior, (2, 3))                  # next to (1,3)
    b.enemy_base_health = {(1, 3): 1.0, (6, 3): 1.0}
    danger = DangerMap({}, b)
    p = build_planner(b, danger)
    base, _, _ = _target_base(b, p, StrategyParams())
    assert base == (1, 3)


def test_target_base_skips_a_doomed_base():
    from scripted.layers import _target_base
    prior = _base_prior(grid_size=7, enemy_bases=((1, 1), (5, 5)))
    b = _strike_belief(prior, (3, 3))
    b.enemy_base_health = {(1, 1): 0.2, (5, 5): 0.5}
    b.own_bombs = [((1, 1), 3)]            # one bomb finishes (1,1) -> doomed
    danger = DangerMap({}, b)
    p = build_planner(b, danger)
    base, _, _ = _target_base(b, p, StrategyParams())
    assert base == (5, 5)                  # the doomed base is skipped


# --- strike: bomb-while-our-damage-still-lands ------------------------------

def test_strike_bombs_when_own_damage_still_lands():
    """A near-full base, single bomb in hand, no in-flight bombs of ours —
    own_hits * 20 = 0 < 100 → bomb."""
    b = _strike_belief(_base_prior(), (3, 5), team_bombs=1)
    danger = DangerMap({}, b)
    p = build_planner(b, danger)
    assert strike(b, danger, p, StrategyParams()) == 5


def test_strike_bombs_a_partly_softened_base_while_own_damage_still_lands():
    """HP=60, no in-flight own bombs (someone else softened) — 0 < 60 → bomb."""
    b = _strike_belief(_base_prior(), (3, 5), team_bombs=3)
    b.enemy_base_health = {(3, 3): 0.6}
    danger = DangerMap({}, b)
    p = build_planner(b, danger)
    assert strike(b, danger, p, StrategyParams()) == 5


def test_strike_yields_when_own_in_flight_already_finishes():
    """HP=40 but two own in-flight bombs hit this base — 40 ≤ 40 → yield."""
    b = _strike_belief(_base_prior(), (3, 5), team_bombs=3)
    b.enemy_base_health = {(3, 3): 0.4}            # 40 HP observed
    # Two of our in-flight bombs reach (3, 3) — 40 in-flight damage covers it.
    b.own_bombs = [((3, 3), 4), ((3, 3), 3)]
    danger = DangerMap({}, b)
    p = build_planner(b, danger)
    assert strike(b, danger, p, StrategyParams()) is None


def test_strike_keeps_bombing_when_own_damage_only_partially_covers():
    """HP=60, one own in-flight bomb (= 20 damage). 20 < 60 → still bomb."""
    b = _strike_belief(_base_prior(), (3, 5), team_bombs=2)
    b.enemy_base_health = {(3, 3): 0.6}
    b.own_bombs = [((3, 3), 3)]                    # one own in-flight hits
    danger = DangerMap({}, b)
    p = build_planner(b, danger)
    assert strike(b, danger, p, StrategyParams()) == 5


def test_strike_bombs_a_low_hp_base_with_a_single_bomb():
    """HP=20, 1 bomb, no in-flight — 0 < 20 → bomb."""
    b = _strike_belief(_base_prior(), (3, 5), team_bombs=1)
    b.enemy_base_health = {(3, 3): 0.2}
    danger = DangerMap({}, b)
    p = build_planner(b, danger)
    assert strike(b, danger, p, StrategyParams()) == 5


# --- sweep leash & default target ------------------------------------------

def test_sweep_leashes_collection_to_a_crippled_target_base():
    """With a Phase-B target base, sweep ignores collectibles outside the
    HP-scaled leash (bombs_needed + 1) of that base."""
    prior = _base_prior(grid_size=16, enemy_bases=((8, 8),))
    prior.collectibles = {(8, 9): 5.0, (1, 1): 5.0}   # near base; far corner
    b = _strike_belief(prior, (8, 11))                # within reach of (8,9)
    b.enemy_base_health = {(8, 8): 0.4}               # 40 HP -> bombs_needed 2 -> leash 3
    danger = DangerMap({}, b)
    p = build_planner(b, danger)
    action = sweep(b, danger, p, StrategyParams())
    # (1,1) is Chebyshev 7+ from the base -> outside leash 3 -> only (8,9) is
    # eligible, so sweep heads toward it (a real move, not None).
    assert action in (0, 1, 2, 3)
    # Confirm the far collectible alone yields nothing (it is leashed out).
    prior.collectibles = {(1, 1): 5.0}
    assert sweep(b, danger, p, StrategyParams()) is None


def test_sweep_unleashed_for_a_full_hp_target_base():
    """A Phase-A (full-HP) target imposes no leash — a far collectible is
    still pursued."""
    prior = _base_prior(grid_size=16, enemy_bases=((8, 8),))
    prior.collectibles = {(1, 1): 5.0}                # far from the base
    b = _strike_belief(prior, (2, 2))
    # no enemy_base_health -> full HP -> Phase A -> no leash
    danger = DangerMap({}, b)
    p = build_planner(b, danger)
    assert sweep(b, danger, p, StrategyParams()) in (0, 1, 2, 3)


def test_default_advances_toward_the_target_base():
    prior = _base_prior(grid_size=7, enemy_bases=((1, 1), (5, 5)))
    b = _strike_belief(prior, (3, 3))
    b.enemy_base_health = {(1, 1): 1.0, (5, 5): 0.4}  # (5,5) is the target
    danger = DangerMap({}, b)
    p = build_planner(b, danger)
    assert default(b, danger, p, StrategyParams()) in (0, 1, 2, 3)


# --- hunt & survive bomb discipline ----------------------------------------

def test_hunt_reserves_a_bomb_for_base_offense():
    """One enemy in range, 2 bombs: while a live base remains hunt holds fire
    (needs 3); with no live base it bombs (needs 2)."""
    from scripted.layers import hunt
    prior = _base_prior(grid_size=7, enemy_bases=((6, 6),))
    b = _strike_belief(prior, (3, 3), team_bombs=2)
    b.enemies = {(4, 3)}                          # one enemy, one tile ahead
    danger = DangerMap({}, b)
    p = build_planner(b, danger)
    assert hunt(b, danger, p, StrategyParams()) != 5      # base lives -> hold
    b.dead_bases = {(6, 6)}                               # endgame
    assert hunt(b, danger, p, StrategyParams()) == 5      # now bombs


def test_survive_skips_a_valueless_flee_drop_while_bases_live():
    """survive's opportunistic flee-drop is suppressed while a live base
    remains if the dropped bomb would reach neither a base nor an enemy."""
    prior = _base_prior(grid_size=5, enemy_bases=((4, 4),))
    b = _strike_belief(prior, (0, 0), team_bombs=3)       # corner, base far
    danger = DangerMap({(2, 0): 4}, b)                    # blast covers x0..3
    p = build_planner(b, danger)
    action = survive(b, danger, p, StrategyParams())
    assert action != 5                                    # no zero-value drop


from scripted.layers import forage_chain


def _chain_belief(collectibles, loc, facing=0, enemy_bases=(), dead_bases=(),
                  grid_size=8, wall_between=None):
    """Belief tailored for forage_chain tests. Reuses _base_prior with
    explicit grid_size + wall_between control."""
    prior = _base_prior(grid_size=grid_size, wall_between=wall_between,
                        enemy_bases=list(enemy_bases))
    prior.collectibles = dict(collectibles)
    b = Belief()
    b.prior = prior
    b.destroyed_walls = set()
    b.dead_bases = set(dead_bases)
    b.collected = set()
    b.enemy_base_health = {}
    b.ally_bombs = {}
    b.location = loc
    b.facing = facing
    b.step = 10
    return b


def test_forage_chain_inactive_while_a_base_lives():
    b = _chain_belief({(3, 3): 5.0}, (2, 2), enemy_bases=[(5, 5)])
    danger = DangerMap({}, b)
    p = build_planner(b, danger)
    assert forage_chain(b, danger, p, StrategyParams()) is None


def test_forage_chain_respects_camp_leash():
    """A high-value cell outside camp_leash radius is filtered out;
    if the only in-leash cell is the agent's own (cost 0 -> unreachable
    semantics for chain), the layer yields."""
    # our_base = (0, 0); camp_leash = 1 limits chebyshev <= 1.
    b = _chain_belief({(1, 1): 1.0, (4, 4): 100.0}, (1, 1), facing=0)
    danger = DangerMap({}, b)
    p = build_planner(b, danger)
    params = StrategyParams(camp_leash=1, forage_requires_endgame=False)
    assert forage_chain(b, danger, p, params) is None


def test_forage_chain_walks_toward_single_reachable_collectible():
    """Single collectible one cell forward of agent (facing RIGHT)."""
    b = _chain_belief({(3, 2): 5.0}, (2, 2), facing=0)
    danger = DangerMap({}, b)
    p = build_planner(b, danger)
    assert forage_chain(b, danger, p, StrategyParams()) == 0  # FORWARD


def test_forage_chain_prefers_forward_over_lateral_when_value_equal():
    """Equal-value cells: one FORWARD (cost 1, no turn), one to the side
    (cost 2 — turn + forward). Forward wins on rate (5/1 > 5/2)."""
    # Facing RIGHT (=0) at (2, 2): FORWARD lands at (3,2); RIGHT-then-FORWARD
    # lands at (2, 3); LEFT-then-FORWARD lands at (2, 1).
    b = _chain_belief({(3, 2): 5.0, (2, 3): 5.0}, (2, 2), facing=0)
    danger = DangerMap({}, b)
    p = build_planner(b, danger)
    assert forage_chain(b, danger, p, StrategyParams()) == 0  # FORWARD


def test_forage_chain_stops_when_marginal_rate_below_running_average():
    """Cells at (3,2) v=5 and (4,2) v=1 with agent at (2,2) facing RIGHT.
    First chain step: (3,2) wins (rate 5 vs (4,2) rate 0.5).
    Second step gate: marginal (4,2) rate from (3,2) = 1/1 = 1.0, current
    chain avg = 5.0. 1.0 < 5.0 -> gate blocks, chain stops at length 1.
    Returned first_action: FORWARD (toward (3,2))."""
    b = _chain_belief({(3, 2): 5.0, (4, 2): 1.0}, (2, 2), facing=0)
    danger = DangerMap({}, b)
    p = build_planner(b, danger)
    assert forage_chain(b, danger, p, StrategyParams()) == 0  # FORWARD


def test_forage_chain_multi_cell_when_marginal_rate_beats_average():
    """Cells (3,2) v=5 and (4,2) v=6, agent at (2,2) facing RIGHT.
    First step: (3,2) rate 5 vs (4,2) rate 3 -> pick (3,2).
    chain_value=5 chain_cost=1, avg=5.
    Second step from (3,2): (4,2) cost 1, marginal 6/1 = 6 > 5 -> accept.
    Both in chain. First action: FORWARD (toward closer cell (3,2))."""
    b = _chain_belief({(3, 2): 5.0, (4, 2): 6.0}, (2, 2), facing=0)
    danger = DangerMap({}, b)
    p = build_planner(b, danger)
    assert forage_chain(b, danger, p, StrategyParams()) == 0  # FORWARD


def test_forage_chain_yields_when_no_collectibles_remain():
    b = _chain_belief({}, (2, 2), facing=0)
    danger = DangerMap({}, b)
    p = build_planner(b, danger)
    assert forage_chain(b, danger, p, StrategyParams()) is None


def test_forage_chain_yields_when_all_collectibles_unreachable():
    """A single collectible at (3, 2) walled off on all four sides is
    unreachable. forage_chain yields and falls through to sweep."""
    target = (3, 2)
    walls = {
        frozenset({target, (4, 2)}): False,
        frozenset({target, (2, 2)}): False,
        frozenset({target, (3, 3)}): False,
        frozenset({target, (3, 1)}): False,
    }
    b = _chain_belief({target: 5.0}, (2, 2), facing=0, wall_between=walls)
    danger = DangerMap({}, b)
    p = build_planner(b, danger)
    assert forage_chain(b, danger, p, StrategyParams()) is None


def test_centre_prox_geometry():
    """`_centre_prox` returns 1.0 at the geometric centre of a 16x16 board
    and 0.0 at any corner, with the appropriate symmetry across both axes."""
    from scripted.layers import _centre_prox

    m = MapPrior.load()
    m.identify_team((13, 9))
    b = Belief()
    b.reset(m)

    # Corners: distance from ((gs-1)/2, (gs-1)/2) is maximal -> centre_prox == 0.
    for corner in [(0, 0), (15, 0), (0, 15), (15, 15)]:
        assert abs(_centre_prox(b, corner)) < 1e-9

    # The four cells nearest the geometric centre (7.5, 7.5) on a 16x16 board
    # all sit at Euclidean distance sqrt(0.5) -> the same centre_prox value,
    # and that value is the maximum any cell can achieve (~0.933).
    centres = [(7, 7), (7, 8), (8, 7), (8, 8)]
    vals = [_centre_prox(b, c) for c in centres]
    assert all(abs(v - vals[0]) < 1e-9 for v in vals)   # symmetry
    assert vals[0] > 0.9                                  # near the max

    # Mid-edge cell is between corner and centre.
    assert 0.0 < _centre_prox(b, (0, 7)) < vals[0]


def _sweep_belief(collectibles, loc, facing=0, dead_bases=(), grid_size=16,
                  our_base=(0, 0)):
    """Minimal belief for sweep/forage_chain tests on an arbitrary square
    grid. No enemy bases (so `_target_base` returns None — sweep's leash
    and base-gradient terms stay inert), no resource gating, no walls."""
    class _Prior:
        pass
    prior = _Prior()
    prior.grid_size = grid_size
    prior.wall_between = {}
    prior.collectibles = dict(collectibles)
    prior.enemy_bases = []
    prior.our_base = our_base
    prior.resource_cells = None
    b = Belief()
    b.prior = prior
    b.dead_bases = set(dead_bases)
    b.location = loc
    b.facing = facing
    b.step = 10
    return b


def test_sweep_prefers_centre_when_boost_overrides_distance():
    """Sweep wiring: at weight=0 the peripheral tile wins on raw
    v/(1+d); at weight=1.0 the multiplicative boost flips the choice."""
    # Agent at (4, 7) facing DIR_RIGHT (=0). Equal value v=5.0 at both
    # collectibles; the central one is one tile farther:
    #
    #   peripheral (2, 7): 2 BACKWARDs at weighted cost 1.4 each = d_p = 2.8
    #     centre_prox(2, 7) = 1 - sqrt(5.5^2 + 0.5^2) / sqrt(2 * 7.5^2)
    #                       ~ 1 - 5.523 / 10.607 ~ 0.479
    #     raw  v/(1+d) = 5 / 3.8 ~ 1.316
    #     boost (W=1.0) = 1 + 1*0.479 = 1.479
    #     boosted score = 5 * 1.479 / 3.8 ~ 1.946
    #
    #   central    (7, 7): 3 FORWARDs at weighted cost 1.0 each = d_c = 3.0
    #     centre_prox(7, 7) = 1 - sqrt(0.5^2 + 0.5^2) / 10.607
    #                       ~ 1 - 0.707 / 10.607 ~ 0.933
    #     raw  v/(1+d) = 5 / 4.0 = 1.250
    #     boost (W=1.0) = 1 + 1*0.933 = 1.933
    #     boosted score = 5 * 1.933 / 4.0 ~ 2.416
    #
    # At weight=0: peripheral 1.316 > central 1.250 -> peripheral wins.
    # At weight=1.0: peripheral 1.946 < central 2.416 -> central wins.
    collectibles = {(2, 7): 5.0, (7, 7): 5.0}
    b = _sweep_belief(collectibles, (4, 7))
    danger = DangerMap({}, b)
    p = build_planner(b, danger)

    action_disabled = sweep(b, danger, p, StrategyParams(centre_value_weight=0.0))
    action_enabled = sweep(b, danger, p, StrategyParams(centre_value_weight=1.0))

    assert action_disabled is not None and action_enabled is not None
    assert action_disabled != action_enabled, (
        "centre bias did not change sweep's choice — wiring regression")
    # Cheapest first action from (4, 7, DIR_RIGHT) to (2, 7) is BACKWARD
    # (=1): 2 BACKWARDs at weighted 1.4 each (2.8) beats LEFT-LEFT-FORWARD-
    # FORWARD (1+1+1+1=4.0). Cheapest first action to (7, 7) is FORWARD (=0).
    assert action_disabled == 1     # BACKWARD toward (2, 7)
    assert action_enabled == 0      # FORWARD toward (7, 7)


def test_forage_chain_prefers_centre_when_boost_overrides_rate():
    """forage_chain wiring: at weight=0 the higher-rate peripheral tile
    wins; at weight=1.0 the boost flips the choice to the central tile."""
    # Agent at (4, 7) facing DIR_RIGHT (0). forage_chain's BFS (`_bfs_facing`)
    # counts every action (FORWARD/BACKWARD/LEFT/RIGHT) at cost 1 — unlike
    # the planner, which weights BACKWARD at 1.4. Equal value v=5.0 at both:
    #
    #   peripheral (1, 7): 3 BACKWARDs at cost 1 each -> min_cost = 3
    #     (LEFT-LEFT-FORWARD*3 = 5; 3 BACKWARDs is cheaper)
    #     centre_prox(1, 7) ~ 0.385
    #     raw rate v / min_cost = 5/3 ~ 1.667
    #     boost (W=1) = 1.385 -> boosted rate = 5 * 1.385 / 3 ~ 2.308
    #
    #   central    (8, 7): 4 FORWARDs at cost 1 each -> min_cost = 4
    #     centre_prox(8, 7) ~ 0.933
    #     raw rate = 5/4 = 1.250
    #     boost (W=1) = 1.933 -> boosted rate = 5 * 1.933 / 4 ~ 2.416
    #
    # At weight=0: peripheral 1.667 > central 1.250 -> peripheral wins.
    # At weight=1.0: peripheral 2.308 < central 2.416 -> central wins.
    collectibles = {(1, 7): 5.0, (8, 7): 5.0}
    b = _sweep_belief(collectibles, (4, 7), facing=0)
    # forage_chain requires endgame by default. `_sweep_belief` sets
    # prior.enemy_bases = [], so live_enemy_bases() returns [], and the
    # endgame gate passes.
    danger = DangerMap({}, b)
    p = build_planner(b, danger)

    action_disabled = forage_chain(
        b, danger, p, StrategyParams(centre_value_weight=0.0))
    action_enabled = forage_chain(
        b, danger, p, StrategyParams(centre_value_weight=1.0))

    assert action_disabled is not None and action_enabled is not None
    assert action_disabled != action_enabled, (
        "centre bias did not change forage_chain's choice — wiring regression")
    # First action from (4, 7, DIR_RIGHT) to (1, 7) via 3 BACKWARDs is
    # BACKWARD (=1). To (8, 7) via 4 FORWARDs is FORWARD (=0).
    assert action_disabled == 1     # BACKWARD toward (1, 7)
    assert action_enabled == 0      # FORWARD toward (8, 7)


def test_enemy_distances_per_enemy_bfs():
    """`_enemy_distances` returns one BFS dict per visible non-frozen
    enemy. Sources have cost 0 in their own dict; cells unreachable from
    a given enemy don't appear in that enemy's dict. Walls block. Frozen
    enemies are excluded entirely (no dict for them) and block their
    cell as an obstacle. Empty `live_enemies()` returns an empty list."""
    from scripted.layers import _enemy_distances

    # Empty case: no enemies -> empty list.
    b_empty = _sweep_belief({}, (4, 7))
    assert _enemy_distances(b_empty) == []

    # Two live enemies plus one frozen. The frozen enemy contributes no
    # dict AND blocks its own cell so no other enemy's BFS can enter it.
    b = _sweep_belief({}, (4, 7))
    b.enemies = {(3, 3), (8, 8), (10, 10)}
    b.frozen_enemies = {(10, 10)}
    dists = _enemy_distances(b)
    assert len(dists) == 2                       # one dict per live enemy
    by_source = {min(d.keys(), key=lambda c: d[c]): d for d in dists}
    # Each live enemy is a source with cost 0 in its own dict.
    d_from_3_3 = next(d for d in dists if d.get((3, 3)) == 0)
    d_from_8_8 = next(d for d in dists if d.get((8, 8)) == 0)
    # From (3, 3) the cell (5, 5) is 4 steps away (2 + 2).
    assert d_from_3_3[(5, 5)] == 4
    # From (8, 8) the same cell is 6 steps away (3 + 3).
    assert d_from_8_8[(5, 5)] == 6
    # Frozen-blocked cell (10, 10) is in NO live enemy's reachable set.
    assert (10, 10) not in d_from_3_3
    assert (10, 10) not in d_from_8_8

    # Wall blocks direct adjacency. Enemy at (5, 5) with a wall between
    # (5, 5) and (5, 6) — BFS to (5, 6) must reroute via (4, 5) -> (4, 6)
    # -> (5, 6), giving cost 3 instead of 1.
    b_wall = _sweep_belief({}, (4, 7))
    b_wall.prior.wall_between = {frozenset({(5, 5), (5, 6)}): False}
    b_wall.enemies = {(5, 5)}
    dists_wall = _enemy_distances(b_wall)
    assert len(dists_wall) == 1
    assert dists_wall[0][(5, 6)] == 3


def test_sweep_deflates_contested_collectible():
    """Sweep wiring: at contested_value_factor=1.0 (off), sweep picks the
    contested tile (it's on the agent's facing axis and at equal weighted
    distance). At contested_value_factor=0.1, the deflation flips the
    choice to the uncontested tile."""
    # centre_value_weight is forced to 0 to isolate the deflation signal.
    # Agent at (4, 7) facing DIR_RIGHT (=0). Equal value v=5.0 at both:
    #
    #   contested   (6, 7): 2 FORWARDs from us -> planner d = 2.0
    #     enemy at (6, 6) -> BFS cost to (6, 7) = 1 step
    #     enemy_cost + 0.5 = 1.5 < 2.0 -> deflate when factor < 1.0
    #
    #   uncontested (4, 8): RIGHT turn + 1 FORWARD -> planner d = 2.0
    #     enemy BFS cost to (4, 8) is 4 (path via (5,6)-(5,7)-(4,7)-(4,8))
    #     enemy_cost + 0.5 = 4.5 > 2.0 -> never deflated
    #
    # Raw v/(1+d) ties at 5/3 ~ 1.667 for both. The dict literal inserts
    # contested first; sweep's `if score > best` is strict, so at
    # factor=1.0 the tie goes to the first-inserted (contested).
    # At factor=0.1, contested boosted = 5 * 0.1 / 3 ~ 0.167 < 1.667;
    # uncontested wins.
    collectibles = {(6, 7): 5.0, (4, 8): 5.0}
    b = _sweep_belief(collectibles, (4, 7))
    b.enemies = {(6, 6)}
    danger = DangerMap({}, b)
    p = build_planner(b, danger)

    action_off = sweep(b, danger, p, StrategyParams(
        centre_value_weight=0.0, contested_value_factor=1.0))
    action_on = sweep(b, danger, p, StrategyParams(
        centre_value_weight=0.0, contested_value_factor=0.1))

    assert action_off is not None and action_on is not None
    assert action_off != action_on, (
        "contested deflation did not change sweep's choice — wiring regression")
    # Cheapest first action from (4, 7, DIR_RIGHT) to (6, 7) is FORWARD
    # (=0): 2 FORWARDs at cost 1.0 each. To (4, 8) the cheapest is
    # RIGHT (=3): turn to DIR_DOWN then 1 FORWARD = cost 2.0.
    assert action_off == 0     # FORWARD toward contested (6, 7)
    assert action_on == 3      # RIGHT toward uncontested (4, 8)


def test_forage_chain_deflates_contested_collectible():
    """forage_chain wiring: at contested_value_factor=1.0 (off), the chain
    picks the contested tile (tied raw rate, first-inserted in the dict).
    At contested_value_factor=0.1, the deflation flips the choice to the
    uncontested tile."""
    from scripted.layers import forage_chain

    # centre_value_weight forced to 0 to isolate the deflation signal.
    # Agent at (4, 7) facing DIR_RIGHT (=0). forage_chain's BFS counts
    # every action at cost 1 — no BACKWARD penalty, turns cost 1.
    #
    #   contested   (6, 7): 2 FORWARDs in (cell, facing) BFS -> min_cost 2
    #     enemy at (6, 6) -> BFS cost to (6, 7) = 1
    #     enemy_cost + 0.5 = 1.5 < 2.0 -> deflate when factor < 1.0
    #
    #   uncontested (4, 8): RIGHT turn + 1 FORWARD -> min_cost 2 at
    #     facing=DIR_DOWN. Enemy BFS cost to (4, 8) = 4.
    #     enemy_cost + 0.5 = 4.5 > 2.0 -> never deflated
    #
    # Raw rates: both v/min_cost = 5/2 = 2.5 -> tie. Dict literal inserts
    # contested first; forage_chain's `if rate > best_rate` is strict, so
    # at factor=1.0 the tie goes to first-inserted (contested).
    # At factor=0.1, contested rate = 5 * 0.1 / 2 = 0.25; uncontested rate
    # stays 2.5 -> uncontested wins.
    collectibles = {(6, 7): 5.0, (4, 8): 5.0}
    b = _sweep_belief(collectibles, (4, 7), facing=0)
    b.enemies = {(6, 6)}
    # forage_chain requires endgame by default; _sweep_belief sets
    # prior.enemy_bases = [] so live_enemy_bases() returns [].
    danger = DangerMap({}, b)
    p = build_planner(b, danger)

    action_off = forage_chain(b, danger, p, StrategyParams(
        centre_value_weight=0.0, contested_value_factor=1.0))
    action_on = forage_chain(b, danger, p, StrategyParams(
        centre_value_weight=0.0, contested_value_factor=0.1))

    assert action_off is not None and action_on is not None
    assert action_off != action_on, (
        "contested deflation did not change forage_chain's choice — wiring regression")
    # First action from (4, 7, DIR_RIGHT) to (6, 7) via 2 FORWARDs is
    # FORWARD (=0). To (4, 8) via RIGHT-turn-then-FORWARD it is RIGHT (=3).
    assert action_off == 0     # FORWARD toward contested (6, 7)
    assert action_on == 3      # RIGHT toward uncontested (4, 8)


def test_sweep_deflation_compounds_with_overlapping_enemies():
    """When TWO enemies both beat us to the same tile, the deflation
    multiplier is `factor ** 2`, not `factor`. The single-enemy case is
    structurally unchanged (k=1 still gives factor^1 = factor) — this
    test exercises only the compounding path."""
    # centre_value_weight forced to 0 to isolate the deflation signal.
    # Agent at (4, 7) facing DIR_RIGHT (=0). Two equal-value collectibles.
    #
    #   single-contest  (6, 7): 2 FORWARDs -> our weighted d = 2.0
    #     enemy_A at (6, 6) reaches in 1 step; enemy_cost+0.5 = 1.5 < 2.0
    #     enemy_B at (3, 0) reaches in (3 left + 7 down) = 10 steps;
    #                                  enemy_cost+0.5 = 10.5 > 2.0 — does
    #                                  NOT beat us. So k = 1 for this tile.
    #     deflation: factor^1 = 0.5 at our test factor.
    #
    #   double-contest  (4, 8): RIGHT-turn + 1 FORWARD -> our weighted d = 2.0
    #     enemy_A at (6, 6) reaches via (5,6)-(5,7)-(4,7)-(4,8) = 4;
    #                                  enemy_cost+0.5 = 4.5 > 2.0 — does
    #                                  NOT beat us via A. Need a closer enemy.
    #
    # Rethink positions so the double-contest tile actually has k=2:
    #
    # Put enemy_A at (4, 9) and enemy_B at (5, 8). Both are 1 step from
    # (4, 8). enemy_cost+0.5 = 1.5 < our d=2.0 -> both beat us -> k=2.
    # Distances from these enemies to (6, 7):
    #   enemy_A (4, 9): 4 steps (down-left chain to (6, 7) needs +2 x, -2 y => 4)
    #   enemy_B (5, 8): 2 steps ((5, 8) -> (5, 7) -> (6, 7) = 2);
    #                            enemy_cost+0.5 = 2.5 > our d=2.0 -> doesn't
    #                            beat us. enemy_A: 4+0.5=4.5 > 2.0 either.
    # So (6, 7) has k=0 from these two enemies. We need a third for the
    # single-contest tile. Add enemy_C at (6, 6) -> beats us to (6, 7)
    # with cost 1+0.5=1.5 < 2.0 (k=1 for (6, 7)). enemy_C's distance to
    # (4, 8) is 4 -> doesn't beat us there. So (4, 8) still has k=2.
    #
    # Final positions:
    #   enemy_A = (4, 9), enemy_B = (5, 8), enemy_C = (6, 6)
    #   (4, 8) contested by A and B (k=2) — factor^2 deflation
    #   (6, 7) contested by C only (k=1) — factor deflation
    #
    # Use factor = 0.5 so the test math is easy to verify:
    #   With centre_value_weight=0 and v=5.0 at both tiles, d=2.0 at both:
    #     k=1 (single, (6, 7)): boost = 0.5 -> score = 5*0.5/3 ~ 0.833
    #     k=2 (double, (4, 8)): boost = 0.25 -> score = 5*0.25/3 ~ 0.417
    #   Single-contest tile (6, 7) wins. First action toward (6, 7) is
    #   FORWARD (=0).
    #
    # For the comparison, also run with factor=1.0 (deflation off):
    #   Both raw scores 5/3 ~ 1.667 (TIE). Dict literal inserts (4, 8)
    #   first; sweep's `>` is strict so (4, 8) wins the tie at factor=1.0.
    #   First action toward (4, 8) is RIGHT (=3).
    collectibles = {(4, 8): 5.0, (6, 7): 5.0}
    b = _sweep_belief(collectibles, (4, 7))
    b.enemies = {(4, 9), (5, 8), (6, 6)}
    danger = DangerMap({}, b)
    p = build_planner(b, danger)

    action_off = sweep(b, danger, p, StrategyParams(
        centre_value_weight=0.0, contested_value_factor=1.0))
    action_on = sweep(b, danger, p, StrategyParams(
        centre_value_weight=0.0, contested_value_factor=0.5))

    assert action_off is not None and action_on is not None
    # Off: (4, 8) wins by insertion order on tied raw score.
    assert action_off == 3      # RIGHT toward (4, 8)
    # On: (4, 8) is deflated factor^2 (k=2); (6, 7) is deflated factor^1
    # (k=1). (6, 7)'s deflated score wins -> FORWARD toward (6, 7).
    assert action_on == 0       # FORWARD toward (6, 7)
