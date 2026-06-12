"""Adaptive cascade layers and shared loop data.

Plan 2 provides the `forage_loop` layer; Plans 3-4 add `defend_intercept`,
`hunt_pursuit`, `rush_roi`, and `trap` here. Every layer follows the cascade
contract used by `decide.act()`:

    layer(belief, danger, planner, params) -> action_int_or_None

and routes via the `planner` argument (a `build_planner` result), never a new
pathfinder.
"""
import json
import math
from pathlib import Path

from scripted.belief import _trace_decision
from scripted.blast import bomb_reaches
from scripted.layers import BASE_MAX_HEALTH, BOMB_ATTACK, _effective_hp
from scripted.geometry import PLACE_BOMB, chebyshev
from scripted.pathfind import BOMB_TIMER, build_planner

_LOOPS_PATH = Path(__file__).resolve().parent.parent / "forage_loops.json"

INF = float("inf")
# Passive bomb economy: base_resource_rate 0.1/step, bomb_cost 1.5 -> ~1 bomb
# per 15 steps. (Resource-tile pickups add more; this is the floor.)
BOMB_REGEN_TICKS = 15


def _load_forage_loops(path=_LOOPS_PATH):
    """Load the shipped forage-loop artifact. Returns (loops, teams)."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return data["loops"], data["teams"]


# Loaded once at import. `LOOPS` is the global loop list; `TEAMS` maps a team
# id (str) to {"home_loop": idx, "order": [idx, ...]}. Built by the offline
# tools/build_forage_loops.py.
LOOPS, TEAMS = _load_forage_loops()


def _best_loop(yield_key):
    """Index of the loop with the highest `yield_key`; None if no loops."""
    if not LOOPS:
        return None
    return max(range(len(LOOPS)), key=lambda i: LOOPS[i][yield_key])


def _maybe_switch(belief, params, active, yield_key, state):
    """Return the loop index to forage this tick — `active`, or a better loop.

    Switches only after `loop_commit_ticks` have elapsed since the last switch
    (hysteresis), and only when the active loop's realised yield has fallen
    below `switch_factor` of its estimate and another loop's estimate beats the
    realised rate.

    On a switch this both (a) returns the new index for the caller to use this
    tick and (b) commits the switch to `state` for subsequent ticks — it writes
    `forage_active_loop`, `forage_waypoint_index`, and `forage_switch_step`. The
    caller must use the returned value (do not drop it): the state write and the
    return value are two halves of the same switch and must stay in sync.
    """
    if belief.step - state["forage_switch_step"] < params.loop_commit_ticks:
        return active
    realised = belief.realised_yield(params.forage_yield_window)
    if realised >= params.switch_factor * LOOPS[active][yield_key]:
        return active                                  # loop still paying
    best = _best_loop(yield_key)
    if best != active and LOOPS[best][yield_key] > realised:
        state["forage_active_loop"] = best
        state["forage_waypoint_index"] = 0
        state["forage_switch_step"] = belief.step
        return best
    return active


def _follow(belief, planner, loop, state):
    """First action toward the loop's next waypoint.

    Advances `state["forage_waypoint_index"]` when the agent is already on the
    current waypoint, and skips waypoints that are unreachable from here.
    Returns None only when no waypoint on the loop is reachable.
    """
    waypoints = [tuple(w) for w in loop["waypoints"]]
    n = len(waypoints)
    idx = state["forage_waypoint_index"] % n
    if waypoints[idx] == belief.location:
        idx = (idx + 1) % n
    for _ in range(n):
        target = waypoints[idx]
        if target != belief.location:
            action = planner.first_action(target)
            if action is not None:
                state["forage_waypoint_index"] = idx
                return action
        idx = (idx + 1) % n
    state["forage_waypoint_index"] = idx
    return None


def forage_loop(belief, danger, planner, params):
    """Patrol the highest-yield endgame forage loop.

    Active only once every enemy base is destroyed: picks the loop with the
    best `yield_endgame` score and walks its waypoints via the planner,
    advancing on arrival. Yields (None) while any base is alive, when no loop
    exists, or when none is reachable.
    """
    live_bases = belief.live_enemy_bases()
    _trace_decision(belief, "forage_loop", "live_bases_count", len(live_bases))
    if live_bases:
        _trace_decision(belief, "forage_loop", "yield_not_endgame", True)
        return None
    if not LOOPS:
        _trace_decision(belief, "forage_loop", "yield_no_loops", True)
        return None
    state = belief.adaptive_state
    yield_key = "yield_endgame"

    active = state.get("forage_active_loop")
    if active is None:
        active = _best_loop(yield_key)
        state["forage_active_loop"] = active
        state["forage_waypoint_index"] = 0
        state["forage_switch_step"] = belief.step
    else:
        active = _maybe_switch(belief, params, active, yield_key, state)
    _trace_decision(belief, "forage_loop", "active_loop", active)

    action = _follow(belief, planner, LOOPS[active], state)
    _trace_decision(belief, "forage_loop", "follow_action", action)
    return action


def _hit_tiles(belief, base):
    """Every tile from which a bomb's blast would reach `base`."""
    gs = belief.prior.grid_size
    return [(x, y) for x in range(gs) for y in range(gs)
            if bomb_reaches((x, y), base, belief)]


def _projected_hp(belief, base, at_tick):
    """Believed HP of `base` after the bomb horizon, projected `at_tick` ticks.

    Subtracts BOMB_ATTACK for every in-flight bomb — the agent's own
    (`belief.own_bombs`, a list of (cell, timer)) and the enemy's
    (`belief.enemy_bombs`, a dict cell -> timer) — whose timer is `<= at_tick`
    (so it will have detonated by then) and whose blast reaches `base`. Floored
    at 0. An unobserved base is assumed full HP (conservative — never skipped).
    """
    observed = belief.enemy_base_health.get(base, 1.0) * BASE_MAX_HEALTH
    hits = sum(1 for cell, timer in belief.own_bombs
               if timer <= at_tick and bomb_reaches(cell, base, belief))
    hits += sum(1 for cell, timer in belief.enemy_bombs.items()
                if timer <= at_tick and bomb_reaches(cell, base, belief))
    return max(0.0, observed - BOMB_ATTACK * hits)


def _forage_rate(belief, params):
    """The forage opportunity cost: the better of the realised forage yield and
    the best shipped loop's attack-phase estimate (reward-per-tick)."""
    realised = belief.realised_yield(params.forage_yield_window)
    best = _best_loop("yield_attack")
    estimate = LOOPS[best]["yield_attack"] if best is not None else 0.0
    return max(realised, estimate)


def _base_roi(belief, planner, base, params):
    """Score attacking `base`. Returns (roi, bombs_needed, eff_hp, nearest_tile).

    HP is projected (`_projected_hp`) to the tick the agent would arrive at a
    bombing tile: a base the in-flight bomb horizon will have already destroyed
    by then scores 0 (do not chase a corpse), and a base being softened scores
    cheaper (vulture it). roi = 50 / kill_cost, boosted by `vulture_hp_boost`
    as projected HP falls; kill_cost is the larger of the arrival ticks and the
    ticks to regenerate the bombs still needed, floored at 1 so it is never a
    zero divisor. roi is 0 / nearest_tile None when no bombing tile is
    reachable.

    The projection uses the single arrival tick for both the skip check and
    `bombs_needed`; a bomb detonating just after arrival is not counted, so
    `bombs_needed` can be marginally high for an already-adjacent base — a
    mild, deliberate conservatism.
    """
    arrival, tile = INF, None
    for t in _hit_tiles(belief, base):
        d = planner.dist_to(t)
        if d < arrival:
            arrival, tile = d, t
    if arrival == INF:
        return (0.0, 1, _projected_hp(belief, base, 0), None)
    eff_hp = _projected_hp(belief, base, arrival)
    if eff_hp <= 0.0:
        return (0.0, 0, 0.0, tile)        # the bomb horizon kills it before arrival
    bombs_needed = max(1, math.ceil(eff_hp / BOMB_ATTACK))
    bombs_short = max(0, bombs_needed - belief.team_bombs)
    kill_cost = max(arrival, bombs_short * BOMB_REGEN_TICKS, 1)
    roi = 50.0 / kill_cost
    roi *= 1.0 + params.vulture_hp_boost * (1.0 - eff_hp / BASE_MAX_HEALTH)
    return (roi, bombs_needed, eff_hp, tile)


def _attack(belief, danger, planner, base):
    """Action that presses the attack on `base`.

    In blast range with a bomb in hand -> PLACE_BOMB (unconditional; bomb-
    stacking is intended). Otherwise route to the nearest bombing tile via the
    planner. If no bombing tile is reachable, breach a wall by bombing now when
    that opens a route. None when nothing productive is possible — including
    the case where the agent is already on a hit-tile but holds no bomb.
    """
    loc = belief.location
    if belief.team_bombs > 0 and bomb_reaches(loc, base, belief):
        return PLACE_BOMB
    hit = _hit_tiles(belief, base)
    best, tile = INF, None
    for t in hit:
        d = planner.dist_to(t)
        if d < best:
            best, tile = d, t
    if tile is not None and best != INF:
        action = planner.first_action(tile)
        if action is not None:
            return action
    # No bombing tile reachable on the open routes — breach a wall toward one.
    if belief.team_bombs > 0:
        breach = build_planner(belief, danger, place_bomb_first=True)
        if any(breach.dist_to(t) != INF for t in hit):
            return PLACE_BOMB
    return None


def rush_roi(belief, danger, planner, params):
    """ROI-gated enemy-base attacker.

    Scores every reachable, non-dead enemy base by `_base_roi` and commits to
    the best — sticky within `roi_gate_margin` so the target does not thrash.
    Attacks that base only while its ROI beats the forage opportunity cost
    (`_forage_rate`, also widened by `roi_gate_margin`); otherwise yields so the
    cascade forages, which itself regenerates bombs. The committed base is
    recorded in `belief.adaptive_state["rush_target"]`.
    """
    live = [b for b in belief.live_enemy_bases()
            if _effective_hp(belief, b) > 0.0]
    _trace_decision(belief, "rush_roi", "live_count", len(live))
    state = belief.adaptive_state
    if not live:
        _trace_decision(belief, "rush_roi", "yield_no_live", True)
        state["rush_target"] = None
        return None

    roi = {b: _base_roi(belief, planner, b, params)[0] for b in live}
    best = max(live, key=lambda b: roi[b])
    _trace_decision(belief, "rush_roi", "best_base", best)
    _trace_decision(belief, "rush_roi", "best_roi", roi[best])

    # Sticky target: keep the committed base unless another beats it by margin.
    prev = state.get("rush_target")
    if (prev in roi and roi[prev] > 0.0
            and roi[best] <= roi[prev] * (1.0 + params.roi_gate_margin)):
        best = prev

    if roi[best] <= 0.0:                               # no base is reachable
        _trace_decision(belief, "rush_roi", "yield_unreachable", True)
        state["rush_target"] = None
        return None
    # ROI gate: attack only when it out-earns foraging. The committed target is
    # kept (not cleared) so the sticky guard still holds on re-entry after a
    # forage interruption — only the immediate action yields.
    forage_rate = _forage_rate(belief, params)
    _trace_decision(belief, "rush_roi", "forage_rate", forage_rate)
    if roi[best] <= forage_rate * (1.0 + params.roi_gate_margin):
        _trace_decision(belief, "rush_roi", "yield_forage_wins", True)
        state["rush_target"] = best
        return None

    state["rush_target"] = best
    action = _attack(belief, danger, planner, best)
    _trace_decision(belief, "rush_roi", "attack_action", action)
    return action


def defend_intercept(belief, danger, planner, params):
    """Intercept an enemy threatening our base — when it can be done in time
    and it out-earns foraging.

    Triggers on a live enemy within `defend_radius` (Chebyshev) of our base.
    Two gates: (1) feasibility — the agent must reach a tile that bombs the
    attacker before the base could be destroyed, else yield (a late defense is
    pure tempo loss); (2) ROI — the +50 swing of saving the base, over the
    round trip, must beat the forage opportunity cost. Gated in: bomb the
    attacker in range (our bomb is friendly-fire-safe — no escape needed), else
    route to the nearest interception tile.
    """
    base = belief.prior.our_base
    if base is None or belief.base_health <= 0.0 or belief.team_bombs <= 0:
        return None
    threats = [e for e in belief.live_enemies()
               if chebyshev(e, base) <= params.defend_radius]
    if not threats:
        return None
    attacker = min(threats, key=lambda e: chebyshev(e, base))

    # Nearest tile from which a bomb reaches the attacker.
    arrival, tile = INF, None
    for t in _hit_tiles(belief, attacker):
        d = planner.dist_to(t)
        if d < arrival:
            arrival, tile = d, t
    if arrival == INF:
        return None

    # Feasibility: arrive before the base could be bombed down.
    attacker_ttk = max(1, math.ceil(belief.base_health / BOMB_ATTACK)) * BOMB_TIMER
    if arrival >= attacker_ttk:
        return None

    # ROI: the +50 swing of saving the base, over the round trip, vs foraging.
    defense_roi = 50.0 / max(1, 2 * arrival)
    if defense_roi <= _forage_rate(belief, params) * (1.0 + params.roi_gate_margin):
        return None

    # Engage: bomb the attacker in range, else route to an interception tile.
    if bomb_reaches(belief.location, attacker, belief):
        return PLACE_BOMB
    return planner.first_action(tile)


_KILLBOXES_PATH = Path(__file__).resolve().parent.parent / "killboxes.json"


def _load_killboxes(path=_KILLBOXES_PATH):
    """Load the shipped killbox-pair artifact. Returns a frozenset of
    `(agent_tile, enemy_tile)` (x, y)-tuple pairs."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return frozenset((tuple(a), tuple(e)) for a, e in data["killboxes"])


KILLBOXES = _load_killboxes()


def trap(belief, danger, planner, params):
    """Deterministic dead-enemy bomb.

    If any visible live enemy `e` satisfies `(belief.location, e) ∈ KILLBOXES`
    — an offline-proved configuration where the enemy cannot escape a bomb
    placed here within the 4-tick fuse — and the runtime safety gates hold,
    `PLACE_BOMB`. Otherwise yield.

    Safety gates: `trap_enabled` knob must be True; the agent must hold at
    least one bomb; no enemy OR ally bombs may be observed on the board
    (since either could open a destructible wall and invalidate the offline
    escape calculation).
    """
    _trace_decision(belief, "trap", "team_bombs", belief.team_bombs)
    if not params.trap_enabled or belief.team_bombs <= 0:
        _trace_decision(belief, "trap", "yield_disabled_or_no_bombs", True)
        return None
    bombs_on_board = bool(belief.enemy_bombs) or bool(belief.ally_bombs)
    _trace_decision(belief, "trap", "bombs_on_board", bombs_on_board)
    if bombs_on_board:
        _trace_decision(belief, "trap", "yield_bomb_safety", True)
        return None
    loc = belief.location
    enemies = list(belief.live_enemies())
    _trace_decision(belief, "trap", "live_enemies", enemies)
    for e in enemies:
        if (loc, e) in KILLBOXES:
            _trace_decision(belief, "trap", "killbox_match", (loc, e))
            return PLACE_BOMB
    _trace_decision(belief, "trap", "yield_no_killbox_match", True)
    return None
