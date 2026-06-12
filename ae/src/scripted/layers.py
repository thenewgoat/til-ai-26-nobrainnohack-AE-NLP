"""The menu of cascade layers. Each public layer is a pure function
(belief, danger, planner, params) -> action_or_None. Composed into named
strategies by strategies.py and run by decide.act().
"""
import math
from collections import deque

from scripted.belief import _trace_decision
from scripted.blast import (
    BLAST_RADIUS, _WallsOpenedView, bomb_reaches, breach_bombs_needed,
    replay_blasts,
)
from scripted.geometry import (
    BACKWARD, FORWARD, LEFT, MOVE, PLACE_BOMB, RIGHT, STAY, chebyshev,
)
from scripted.pathfind import BOMB_TIMER, build_planner

INF = float("inf")
ESCAPE_HORIZON = BOMB_TIMER          # steps to vacate a blast zone (= bomb fuse)
BASE_MAX_HEALTH = 100.0              # enemy base HP at full (til_environment default)
BOMB_ATTACK = 20.0                   # damage one bomb deals
OPENNESS_WINDOW = 12     # closest safe cells scored for openness (bounds per-tick cost)


def _openness(belief, danger, cell, radius):
    """Count safe cells reachable from `cell` within `radius` BFS steps (cell
    itself included). A dead-end pocket scores low; an open area scores high.
    Walls block traversal; danger cells are neither counted nor traversed."""
    gs = belief.prior.grid_size
    start = tuple(cell)
    seen = {start}
    q = deque([(start, 0)])
    while q:
        t, d = q.popleft()
        if d >= radius:
            continue
        for mdir in range(4):
            dx, dy = MOVE[mdir]
            nb = (t[0] + dx, t[1] + dy)
            if (nb in seen or not (0 <= nb[0] < gs and 0 <= nb[1] < gs)
                    or belief.is_wall(t, nb) or danger.is_dangerous(nb)):
                continue
            seen.add(nb)
            q.append((nb, d + 1))
    return len(seen)


def _centre_prox(belief, cell):
    """Approaches 1.0 at the geometric map centre, 0.0 at any corner.
    Euclidean distance from ((gs-1)/2, (gs-1)/2) — the centre lies between
    cells on an even grid, so no integer cell reaches 1.0 exactly; the four
    cells nearest centre on a 16x16 board top out around 0.933. Matches the
    env's radial respawn gradient in
    `til_environment.arena.ArenaGenerator.generate_respawn_map`. Used as a
    coarse proxy for centre tiles respawning ~4x faster than corners; the
    env's Perlin component makes the proxy approximate but the underlying
    gradient is purely geometric, so walls do not enter the calculation."""
    gs = belief.prior.grid_size
    cx = cy = (gs - 1) / 2.0
    dx, dy = cell[0] - cx, cell[1] - cy
    dist = (dx * dx + dy * dy) ** 0.5
    max_dist = (2.0 * cx * cx) ** 0.5
    return 1.0 - dist / max_dist


def _enemy_distances(belief):
    """Per-enemy 4-connected BFS distances from each visible (non-frozen)
    enemy. Returns a list of dicts: one `dict[cell, int]` per enemy, where
    each dict maps cells reachable from that specific enemy to tile-distance.
    Empty list if no enemies are visible.

    Facing-blind: every step is 1 tick, no turn cost. The env does not
    expose enemy facing in the viewcone observation (see
    `til_environment/observation.py` ViewChannel — ENEMY_AGENT presence
    and ENEMY_AGENT_HEALTH only). Callers add `+0.5` for the expected
    turn-cost adjustment when comparing against our own cost.

    Per-enemy structure (vs. a single merged-min dict) lets callers count
    how many enemies could individually beat us to a tile, so the
    deflation can compound (`factor ** k`) for tiles contested by `k`
    enemies — single-enemy contested tiles still get `factor`, but
    overlapping tiles get progressively more aggressive deflation.

    Walls and frozen enemies block. Danger cells do NOT block — enemies
    are not deterred by their own teammates' bombs the way we model our
    own avoidance. Same wall belief as our planner (live `wall_between`
    minus `destroyed_walls`, via `belief.is_wall`)."""
    sources = belief.live_enemies()
    if not sources:
        return []
    gs = belief.prior.grid_size
    blocked = set(belief.frozen_enemies)
    result = []
    for src in sources:
        s = tuple(src)
        if s in blocked:
            continue
        cost = {s: 0}
        q = deque([s])
        while q:
            cell = q.popleft()
            c = cost[cell]
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nb = (cell[0] + dx, cell[1] + dy)
                if not (0 <= nb[0] < gs and 0 <= nb[1] < gs):
                    continue
                if nb in cost or nb in blocked:
                    continue
                if belief.is_wall(cell, nb):
                    continue
                cost[nb] = c + 1
                q.append(nb)
        result.append(cost)
    return result


def _enemy_threat_penalty(belief, cell, factor):
    """Multiplicative devaluation for a collectible inside a VISIBLE enemy's
    bomb-blast footprint — the "agent bias" that steers foraging away from
    enemies.

    A cell is threatened by an enemy if a bomb at the enemy's tile would hit it
    (`bomb_reaches`: LOS + Chebyshev `BLAST_RADIUS`). The penalty is
    `factor ** (BLAST_RADIUS + 1 - cheb)`, so an adjacent (cheb 1) collectible
    is hit hardest (factor**2 at radius 2) and a range-2 one less (factor**1);
    penalties compound across enemies that each threaten the cell. Returns 1.0
    when `factor >= 1.0` (disabled), the cell is out of every enemy's footprint
    (walls/LOS cut it off), or only frozen enemies are near (live_enemies only).

    Uses only enemies visible THIS tick — there is no decaying memory of
    out-of-view enemies; the next observation re-derives the footprint."""
    if factor >= 1.0:
        return 1.0
    penalty = 1.0
    for e in belief.live_enemies():
        if bomb_reaches(e, cell, belief):
            penalty *= factor ** (BLAST_RADIUS + 1 - chebyshev(e, cell))
    return penalty


def _visibility_penalty(belief, cell, factor):
    """Multiplicative devaluation for a collectible the agent cannot currently
    see — shrinks the forager's effective search space to loot it has eyes on
    rather than chasing stale belief-prior tiles across the map.

    `factor` (1.0 disables) is applied once when `cell` is not in
    `belief.last_visible_cells`, the LOS-occluded set of cells the viewcone
    folded this tick. A visible cell is unpenalised."""
    if factor >= 1.0:
        return 1.0
    return factor if tuple(cell) not in belief.last_visible_cells else 1.0


def _strike_caveat(belief, danger, planner, loc, deadline, gs):
    """When the agent is in danger, see if a strike can still be fulfilled
    before fleeing. Returns the action to take (PLACE_BOMB or a movement) or
    None if no opportunity exists.

    Tries Case B first (place here and escape) then Case A (route to a safer
    hit-tile). Case B is prioritised because scoring opportunities can vanish
    in a single tick.

    A base is "bombable" only when our in-flight damage doesn't already
    finish it — adding more bombs past that point is wasted.
    """
    if belief.team_bombs <= 0:
        return None
    live_bases = belief.live_enemy_bases()
    if not live_bases:
        return None

    bombable = []
    for base in live_bases:
        own_hits = sum(1 for cell, _ in belief.own_bombs
                       if bomb_reaches(cell, base, belief))
        observed_hp = belief.enemy_base_health.get(base, 1.0) * BASE_MAX_HEALTH
        if BOMB_ATTACK * own_hits < observed_hp:
            bombable.append(base)
    if not bombable:
        return None

    # Case B — place here, escape after. Loc must hit a bombable base AND a
    # non-dangerous cell must be reachable in `deadline - 1` phases (one phase
    # used for the place action).
    loc_hits = any(bomb_reaches(loc, base, belief) for base in bombable)
    if loc_hits:
        escape_budget = deadline - 1
        for x in range(gs):
            for y in range(gs):
                cell = (x, y)
                if cell == loc or danger.is_dangerous(cell):
                    continue
                if planner.steps_to(cell) <= escape_budget:
                    _trace_decision(belief, "survive", "strike_caveat_case_b",
                                    cell)
                    return PLACE_BOMB

    # Case A — route to a safer hit-tile.
    best_tile, best_d = None, INF
    for base in bombable:
        for x in range(gs):
            for y in range(gs):
                tile = (x, y)
                if tile == loc:
                    continue
                if danger.is_dangerous(tile):
                    continue
                if not bomb_reaches(tile, base, belief):
                    continue
                d = planner.dist_to(tile)
                if d == INF or d > deadline:
                    continue
                if d < best_d:
                    best_tile, best_d = tile, d
    if best_tile is not None:
        _trace_decision(belief, "survive", "strike_caveat_case_a", best_tile)
        return planner.first_action(best_tile)
    return None


def survive(belief, danger, planner, params):
    """Layer 1 — escape bomb danger.

    Tier 1: when a fully-safe cell is reachable before our cell detonates,
    route there, biasing toward open cells over dead-end pockets; first drop a
    surplus bomb if the escape still completes afterwards. Tier 2: when no
    fully-safe cell is reachable in time, move to the least-bad reachable cell
    (fewest overlapping bombs, then latest detonation, then nearest), or yield
    to the objective layers when already on it or when nothing is reachable.
    Never returns STAY.
    """
    loc = belief.location
    loc_danger = danger.is_dangerous(loc, within=ESCAPE_HORIZON)
    _trace_decision(belief, "survive", "loc_is_dangerous", loc_danger)
    if not loc_danger:
        _trace_decision(belief, "survive", "yield_safe_loc", True)
        return None
    gs = belief.prior.grid_size
    deadline = danger.ticks_to_danger(loc)
    _trace_decision(belief, "survive", "deadline", deadline)

    # Strike caveat: don't immediately flee — every bomb that damages a base
    # scores, so if a scoring opportunity is still available we'd rather take
    # it than waste the tick fleeing. Case B (place at loc, escape after) is
    # tried first because scoring windows are short-lived; Case A (route to a
    # safer hit-tile) is the cleaner-but-slower follow-up.
    strike_action = _strike_caveat(belief, danger, planner, loc, deadline, gs)
    if strike_action is not None:
        return strike_action

    # Tier 1 — fully-safe cells reachable before loc detonates.
    safe = []
    for x in range(gs):
        for y in range(gs):
            cell = (x, y)
            if danger.is_dangerous(cell):
                continue
            # The agent must reach the safe cell BY the detonation tick. MOVE
            # runs before DETONATE in the env, so a phase-`deadline` arrival
            # lands the agent at the safe cell just before the blast.
            if planner.steps_to(cell) <= deadline:
                safe.append(cell)
    _trace_decision(belief, "survive", "tier1_safe_count", len(safe))
    if safe:
        # Closest first; score openness only for the closest few (bounds cost).
        safe.sort(key=planner.dist_to)
        chosen, best_score = None, INF
        for cell in safe[:OPENNESS_WINDOW]:
            score = (planner.dist_to(cell)
                     - params.openness_weight
                     * _openness(belief, danger, cell, params.openness_radius))
            if score < best_score:
                best_score, chosen = score, cell
        # Opportunistic bomb drop: a surplus bomb, the escape still completes
        # after the place tick — and, while live bases remain, the dropped
        # bomb must actually reach a base or a live enemy (no zero-value drop).
        # The `bomb_drop_min` floor exists to hoard bombs for base offense; once
        # every enemy base is down, the pool has no other use, so the floor
        # drops to 1 (= any bomb in hand qualifies).
        live_bases = belief.live_enemy_bases()
        drop_hits = (not live_bases
                     or any(bomb_reaches(loc, bs, belief) for bs in live_bases)
                     or any(bomb_reaches(loc, e, belief)
                            for e in belief.live_enemies()))
        bombs_floor = params.bomb_drop_min if live_bases else 1
        bomb_drop_ok = (belief.team_bombs >= bombs_floor
                        and planner.steps_to(chosen) + 1 + params.bomb_drop_buffer
                        <= deadline
                        and drop_hits)
        _trace_decision(belief, "survive", "tier1_chosen", chosen)
        _trace_decision(belief, "survive", "tier1_drop_bomb", bomb_drop_ok)
        if bomb_drop_ok:
            return PLACE_BOMB
        # `chosen` is a non-dangerous cell while `loc` is dangerous, so
        # `chosen != loc` and `first_action` is always a real move here.
        return planner.first_action(chosen)

    # Tier 2 — no fully-safe escape in time; move to the least-bad cell.
    best_key, best_tile = None, None
    for x in range(gs):
        for y in range(gs):
            cell = (x, y)
            d = planner.dist_to(cell)
            if d == INF:
                continue
            key = (danger.overlap(cell), -danger.ticks_to_danger(cell), d)
            if best_key is None or key < best_key:
                best_key, best_tile = key, cell
    _trace_decision(belief, "survive", "tier2_best_tile", best_tile)
    if best_tile is None or best_tile == loc:
        _trace_decision(belief, "survive", "yield_tier2_stuck", True)
        return None
    return planner.first_action(best_tile)


def _bomb_sequence(belief):
    """Every in-flight bomb (own + ally) as (cell, counts_for_damage), sorted
    by detonation order.

    Both `own_bombs` entries and `ally_bombs` values carry the same remaining
    timer (phases until detonation from the current obs), so the timer IS the
    detonation order. Only own bombs count for damage (ally damage stays
    unaccounted, conservative — same as before), but EVERY blast opens walls
    for the bombs behind it."""
    seq = [(timer, tuple(cell), True) for cell, timer in belief.own_bombs]
    seq += [(timer, tuple(cell), False)
            for cell, timer in belief.ally_bombs.items()]
    seq.sort(key=lambda entry: entry[0])
    return [(cell, counted) for _, cell, counted in seq]


def _effective_hp(belief, base):
    """Believed HP of `base` after the agent's own in-flight bombs land.

    Observed HP (last-seen ratio x BASE_MAX_HEALTH) minus BOMB_ATTACK per own
    bomb whose blast reaches the base — replayed in detonation order, so an
    own bomb behind a destructible wall counts once an earlier blast (ours or
    an ally's) opens it. Floored at 0. This is the true remaining work —
    observed HP does not drop until a bomb's fuse expires. A base never
    observed is assumed full (conservative — never skipped).
    """
    observed = belief.enemy_base_health.get(base, 1.0) * BASE_MAX_HEALTH
    in_flight, _ = replay_blasts(_bomb_sequence(belief), base, belief)
    return max(0.0, observed - BOMB_ATTACK * in_flight)


def _base_doomed(belief, base):
    """True if the agent's own in-flight bombs already finish `base`, so
    another bomb on it would be wasted."""
    return _effective_hp(belief, base) <= 0.0


def _strike_tiles(belief, base, params):
    """tile -> wall-opening bombs needed (0 = direct LOS) for every tile from
    which a bomb sequence can damage `base`.

    Extends the classic hit-tile set with tiles whose line-of-sight is blocked
    only by destructible walls: up to `params.los_breach_max` openers placed
    there blow the walls, and the bombs behind them damage the base. A tile
    needing k openers is offered only while `team_bombs >= k + 1`, so at least
    one damaging bomb can follow. Walls that in-flight bombs (own or ally)
    will open count as already gone — every bomb in the air detonates before
    one placed this tick. Only the Chebyshev blast window around the base can
    ever qualify, so the scan is 5x5, not grid-wide."""
    _, opened = replay_blasts(_bomb_sequence(belief), base, belief)
    view = _WallsOpenedView(belief, opened) if opened else belief
    gs = belief.prior.grid_size
    bx, by = base
    out = {}
    for x in range(max(0, bx - BLAST_RADIUS), min(gs, bx + BLAST_RADIUS + 1)):
        for y in range(max(0, by - BLAST_RADIUS), min(gs, by + BLAST_RADIUS + 1)):
            k = breach_bombs_needed((x, y), base, view, params.los_breach_max)
            if k is not None and belief.team_bombs >= k + 1:
                out[(x, y)] = k
    return out


def _walkable_direct_tile(belief, planner, tiles, max_walk):
    """Nearest direct-LOS (zero-opener) strike tile worth walking to instead
    of dumping opener bombs where we stand.

    Qualifies when its planner arrival cost is strictly under `max_walk` AND
    no visible live enemy stands on the route (an enemy body-blocking the
    corridor would stall the walk — keep dumping instead; frozen enemies are
    already planner-impassable). Returns the tile, or None."""
    enemies = {tuple(e) for e in belief.live_enemies()}
    best, best_tile = max_walk, None
    for tile, k in tiles.items():
        if k:
            continue
        d = planner.dist_to(tile)
        if d >= best or d == 0:
            continue
        if enemies and enemies.intersection(planner.route_to(tile)):
            continue
        best, best_tile = d, tile
    return best_tile


def _target_base(belief, planner, params):
    """Pick the single enemy base the agent commits to.

    Returns (base, effective_hp, bombs_needed) for the live, non-doomed base
    with the lowest blended score `bombs_needed + target_travel_weight *
    arrival` (arrival = earliest planner tick to a tile that can bomb it).
    Ties, and the all-unreachable case, fall to the lowest bombs_needed. A
    damaged base keeps the lowest bombs_needed, so the target is sticky.
    Returns None when no live, non-doomed base exists.
    """
    best_key, best = None, None
    for base in belief.live_enemy_bases():
        if _base_doomed(belief, base):
            continue
        eff = _effective_hp(belief, base)
        bombs_needed = math.ceil(eff / BOMB_ATTACK)
        arrival = INF
        for tile, k in _strike_tiles(belief, base, params).items():
            d = planner.dist_to(tile) + k
            if d < arrival:
                arrival = d
        score = bombs_needed + params.target_travel_weight * arrival
        key = (score, bombs_needed, arrival)
        if best_key is None or key < best_key:
            best_key, best = key, (base, eff, bombs_needed)
    return best


def strike(belief, danger, planner, params):
    """Bomb the chosen enemy base while our own damage still lands.

    Every bomb we place that DAMAGES the base scores (`attack_damage` is
    `scale_by_return`), so the meta is to dump bombs until our in-flight
    damage would already finish the base — beyond that, additional bombs
    are wasted on a dead target. Concretely: yield iff
    `BOMB_ATTACK * own_hits >= observed_hp` where `own_hits` counts our
    in-flight bombs whose blast reaches the base.

    Once committed: bomb from the best strike tile, else breach a wall toward
    the base when that is strictly faster, else navigate to a strike tile. A
    strike tile either has direct line-of-sight within Chebyshev 2, or gains
    it after up to `los_breach_max` of its own bombs open the destructible
    walls in between (`_strike_tiles`) — the openers do no damage, the bombs
    behind them do. Tile costs blend travel ticks with opener bombs (one
    opener = one tick spent placing without damaging), so a k-opener tile
    underfoot beats a direct tile k+1 ticks away.

    Gives up entirely once `strike_dead_bases_cap` bases are observed dead
    (our own base counts) — past that point the cascade forages/hunts.
    """
    _trace_decision(belief, "strike", "team_bombs", belief.team_bombs)
    if belief.team_bombs <= 0:
        _trace_decision(belief, "strike", "yield_no_bombs", True)
        return None
    # Endgame give-up: once enough bases are observed dead (ours included),
    # razing another one no longer moves the result — yield every tick so the
    # cascade spends the remaining bombs and ticks on hunt/forage instead.
    dead = len(belief.dead_bases) + (1 if belief.base_health <= 0 else 0)
    if params.strike_dead_bases_cap and dead >= params.strike_dead_bases_cap:
        _trace_decision(belief, "strike", "yield_dead_bases_cap", dead)
        return None
    target = _target_base(belief, planner, params)
    _trace_decision(belief, "strike", "target", target)
    if target is None:
        _trace_decision(belief, "strike", "yield_no_target", True)
        return None
    base, _, _ = target

    own_hits, _ = replay_blasts(_bomb_sequence(belief), base, belief)
    observed_hp = belief.enemy_base_health.get(base, 1.0) * BASE_MAX_HEALTH
    in_flight_damage = BOMB_ATTACK * own_hits
    _trace_decision(belief, "strike", "own_hits", own_hits)
    _trace_decision(belief, "strike", "observed_hp", observed_hp)
    if in_flight_damage >= observed_hp:
        _trace_decision(belief, "strike", "yield_own_bombs_finish", True)
        return None

    loc = belief.location
    tiles = _strike_tiles(belief, base, params)

    def hit_cost(plan):
        """Cheapest (arrival + openers, tile) over the strike tiles."""
        best, best_tile = INF, None
        for tile, k in tiles.items():
            c = plan.dist_to(tile) + k
            if c < best:
                best, best_tile = c, tile
        return best, best_tile

    # 1. Bomb from where we stand — a direct hit (k=0), or the start of an
    #    opener dump when no other tile is worth the walk.
    t_a, best_tile = hit_cost(planner)
    _trace_decision(belief, "strike", "hit_cost_no_breach", t_a)
    k_here = tiles.get(loc)
    _trace_decision(belief, "strike", "direct_hit", k_here == 0)
    if k_here is not None and k_here <= t_a:
        if k_here:
            # Bombs are a team resource: before dumping openers, prefer a
            # short walk to a no-breach tile — one < `direct_walk_max` ticks
            # away with no visible enemy standing on the route.
            direct = _walkable_direct_tile(belief, planner, tiles,
                                           params.direct_walk_max)
            if direct is not None:
                _trace_decision(belief, "strike", "walk_over_breach", direct)
                return planner.first_action(direct)
            _trace_decision(belief, "strike", "los_breach_openers", k_here)
        return PLACE_BOMB

    # 2. Breach: dropping a bomb now opens a wall and reaches a strike tile
    #    strictly sooner.
    if belief.team_bombs >= params.breach_min_bombs:
        bomb_planner = build_planner(belief, danger, place_bomb_first=True)
        t_b, _ = hit_cost(bomb_planner)
        _trace_decision(belief, "strike", "hit_cost_with_breach", t_b)
        if t_b < t_a:
            _trace_decision(belief, "strike", "breach_now", True)
            return PLACE_BOMB

    # 3. Navigate toward the cheapest strike tile (no breach).
    _trace_decision(belief, "strike", "nav_best_tile", best_tile)
    if best_tile is None or t_a == INF:
        _trace_decision(belief, "strike", "yield_no_nav", True)
        return None
    return planner.first_action(best_tile)


FORAGE_MOVES = (FORWARD, BACKWARD, LEFT, RIGHT)


def _move_result(belief, danger, tile, facing, action):
    """Apply one movement action. Returns (new_tile, new_facing), or None if
    the move is blocked, leaves the grid, or steps into a danger cell."""
    if action == LEFT:
        return (tile, (facing + 3) % 4)
    if action == RIGHT:
        return (tile, (facing + 1) % 4)
    mdir = facing if action == FORWARD else (facing + 2) % 4
    dx, dy = MOVE[mdir]
    nb = (tile[0] + dx, tile[1] + dy)
    gs = belief.prior.grid_size
    if not (0 <= nb[0] < gs and 0 <= nb[1] < gs):
        return None
    if (belief.is_wall(tile, nb) or danger.is_dangerous(nb)
            or nb in belief.frozen_enemies):
        return None
    return (nb, facing)


def forage(belief, danger, planner, params):
    """Endgame greedy collector — active once every enemy base is destroyed.

    Looks two moves ahead: for each first move, adds the collectible value of
    the tile it lands on to the best value reachable by one further move, then
    returns the first move of the best pair. Yields (returns None) when no
    collectible is within two moves, so the cascade falls through to sweep.
    """
    if params.forage_requires_endgame and belief.live_enemy_bases():
        return None                       # bases remain — not the endgame yet
    remaining = belief.remaining_collectibles()
    if params.camp_leash is not None:
        base = belief.prior.our_base
        remaining = {c: v for c, v in remaining.items()
                     if chebyshev(c, base) <= params.camp_leash}
    start_tile, facing = belief.location, belief.facing
    best_score, best_action = 0.0, None
    for a1 in FORAGE_MOVES:
        s1 = _move_result(belief, danger, start_tile, facing, a1)
        if s1 is None:
            continue
        v1 = remaining.get(s1[0], 0.0) if s1[0] != start_tile else 0.0
        best_after = 0.0
        for a2 in FORAGE_MOVES:
            s2 = _move_result(belief, danger, s1[0], s1[1], a2)
            if s2 is None:
                continue
            is_new = s2[0] not in (start_tile, s1[0])
            v2 = remaining.get(s2[0], 0.0) if is_new else 0.0
            best_after = max(best_after, v2)
        score = v1 + best_after
        if score > best_score:
            best_score, best_action = score, a1
    return best_action if best_score > 0.0 else None


def _bfs_facing(belief, danger, start):
    """BFS in (cell, facing) state space, edges = 4 actions cost 1 each.

    Returns (cost, parent). cost[(cell, facing)] = min ticks to reach that
    state from `start`. parent[(cell, facing)] = (prev_state, action) so the
    path to any state can be reconstructed by walking parent pointers back to
    `start`. Blocked moves (walls / danger / off-grid / frozen enemies) follow
    `_move_result`'s gating exactly. Turn actions (LEFT/RIGHT) traverse to a
    new state (same cell, different facing) at cost 1 — so a plan that turns
    twice before moving costs 2 more ticks than one that moves straight."""
    cost = {start: 0}
    parent = {}
    q = deque([start])
    while q:
        s = q.popleft()
        tile, facing = s
        for action in FORAGE_MOVES:
            result = _move_result(belief, danger, tile, facing, action)
            if result is None:
                continue
            ns = result
            if ns in cost:
                continue
            cost[ns] = cost[s] + 1
            parent[ns] = (s, action)
            q.append(ns)
    return cost, parent


def _trace_first_action(parent, start, target):
    """Walk back through `parent` from `target` until the predecessor is
    `start`; return the action on that first edge. Returns None when target
    == start (the path is empty)."""
    s = target
    while s in parent:
        prev, action = parent[s]
        if prev == start:
            return action
        s = prev
    return None


def forage_chain(belief, danger, planner, params):
    """Rate-maximizing facing-aware collector.

    Same gating as `forage` (yields while a base lives unless
    `params.forage_requires_endgame` is False; respects `camp_leash`).

    Searches in (cell, facing) state space — turn actions cost a tick, so
    plans that waste ticks turning have a higher path cost than direct
    forward sweeps. Greedily chains collectibles: each iteration, BFS from
    the chain's tail, then accept the candidate `c` with the highest
    marginal rate `v_c / min_cost_to_c` provided it strictly beats the
    chain's running average rate (first iteration is unconditional). Stops
    when no candidate beats the average. Returns the first action of the
    path to the first chained collectible, or None if no positive-rate
    candidate exists (cascade falls through to sweep)."""
    if params.forage_requires_endgame and belief.live_enemy_bases():
        return None
    remaining = belief.remaining_collectibles()
    if params.camp_leash is not None:
        base = belief.prior.our_base
        remaining = {c: v for c, v in remaining.items()
                     if chebyshev(c, base) <= params.camp_leash}
    if not remaining:
        return None

    enemy_distances = _enemy_distances(belief)
    start_state = (belief.location, belief.facing)
    chained = set()
    chain_value = 0.0
    chain_cost = 0
    first_action = None
    current_state = start_state

    while True:
        cost, parent = _bfs_facing(belief, danger, current_state)

        best_c = None
        best_cost = 0
        best_facing = 0
        best_rate = 0.0
        for c, v in remaining.items():
            if c in chained:
                continue
            min_cost = None
            min_facing = None
            for f in range(4):
                state = (c, f)
                if state in cost and (min_cost is None or cost[state] < min_cost):
                    min_cost = cost[state]
                    min_facing = f
            if min_cost is None or min_cost == 0:
                continue
            boost = 1.0 + params.centre_value_weight * _centre_prox(belief, c)
            k = sum(1 for ed in enemy_distances if (c in ed) and ed[c] + 0.5 < min_cost)
            if k > 0:
                boost *= params.contested_value_factor ** k
            boost *= _enemy_threat_penalty(belief, c, params.enemy_avoid_factor)
            boost *= _visibility_penalty(belief, c, params.unseen_value_factor)
            rate = (v * boost) / min_cost
            if rate > best_rate:
                best_rate = rate
                best_c = c
                best_cost = min_cost
                best_facing = min_facing

        if best_c is None:
            break

        if chain_cost > 0:
            avg_rate = chain_value / chain_cost
            if best_rate <= avg_rate:
                break

        if first_action is None:
            first_action = _trace_first_action(parent, start_state,
                                               (best_c, best_facing))

        chained.add(best_c)
        chain_value += remaining[best_c]
        chain_cost += best_cost
        current_state = (best_c, best_facing)

    return first_action


def sweep(belief, danger, planner, params):
    """Head for the best-value reachable collectible.

    Leash precedence: the camper's our-base leash (`camp_leash`) wins; else a
    Phase-B target base (effective_hp <= soften_floor) leashes collection to
    `bombs_needed + 1` Chebyshev of that base, so the agent accumulates bombs
    near the kill; else no leash. The drift gradient points at the target base.

    score = value * centre_boost / (1 + dist) + a small gradient toward the
    target base. `centre_boost = 1 + centre_value_weight * _centre_prox(cell)`
    biases scoring toward central tiles (which respawn ~4x faster).
    """
    gs = belief.prior.grid_size
    target = _target_base(belief, planner, params)
    _trace_decision(belief, "sweep", "target", target)

    # Resolve the leash (centre, radius), by precedence.
    leash_centre, leash_radius = None, None
    if params.camp_leash is not None:
        leash_centre, leash_radius = belief.prior.our_base, params.camp_leash
    elif target is not None:
        base, eff_hp, bombs_needed = target
        if eff_hp <= params.soften_floor:
            leash_centre, leash_radius = base, bombs_needed + 1

    # Filter to resource-kind tiles when the prior has kind info; otherwise
    # fall back to all collectibles (hand-rolled test priors leave it None).
    # Open the gate when every enemy base is dead (endgame — no need to hoard
    # bombs) or when no resource remains in our belief (the resource pool is
    # exhausted; mission/recon still pay reward).
    resource_cells = getattr(belief.prior, "resource_cells", None)
    remaining = belief.remaining_collectibles()
    allow_all = (
        resource_cells is None
        or not belief.live_enemy_bases()
        or not any(c in resource_cells for c in remaining))
    _trace_decision(belief, "sweep", "leash", (leash_centre, leash_radius))
    _trace_decision(belief, "sweep", "remaining_count", len(remaining))
    _trace_decision(belief, "sweep", "allow_all", allow_all)
    enemy_distances = _enemy_distances(belief)
    best, best_tile = -INF, None
    for cell, value in remaining.items():
        if not allow_all and cell not in resource_cells:
            continue
        if (leash_centre is not None
                and chebyshev(cell, leash_centre) > leash_radius):
            continue
        d = planner.dist_to(cell)
        # d == 0: already on the tile (first_action would be None); d == INF: unreachable.
        if d == INF or d == 0:
            continue
        boost = 1.0 + params.centre_value_weight * _centre_prox(belief, cell)
        k = sum(1 for ed in enemy_distances if (cell in ed) and ed[cell] + 0.5 < d)
        if k > 0:
            boost *= params.contested_value_factor ** k
        boost *= _enemy_threat_penalty(belief, cell, params.enemy_avoid_factor)
        boost *= _visibility_penalty(belief, cell, params.unseen_value_factor)
        score = (value * boost) / (1.0 + d)
        if target is not None:
            near = chebyshev(cell, target[0])
            score += params.sweep_base_gradient * (1.0 - near / gs)
        if score > best:
            best, best_tile = score, cell
    _trace_decision(belief, "sweep", "best_tile", best_tile)
    if best_tile is None:
        _trace_decision(belief, "sweep", "yield_no_collectible", True)
        return None
    return planner.first_action(best_tile)


def default(belief, danger, planner, params):
    """Advance toward the chosen enemy base. Yields (None) when no live,
    non-doomed base exists or the chosen base is unreachable from here."""
    target = _target_base(belief, planner, params)
    _trace_decision(belief, "default", "target", target)
    if target is None:
        _trace_decision(belief, "default", "yield_no_target", True)
        return None
    action = planner.first_action(target[0])
    _trace_decision(belief, "default", "first_action", action)
    return action


def camp(belief, danger, planner, params):
    """Defensive homebody. Territory is `camp_leash` Chebyshev of our base.

    Bomb enemies inside the territory; return home when outside the leash;
    otherwise None, letting forage/sweep collect within the leash.

    Precondition: ``params.camp_leash is not None`` — this layer is only used
    by the ``camper`` strategy, which sets ``camp_leash=4``.
    """
    leash = params.camp_leash
    base = belief.prior.our_base
    loc = belief.location

    # Defend: an enemy inside our territory.
    if belief.team_bombs > 0:
        threats = [e for e in belief.enemies if chebyshev(e, base) <= leash]
        if threats:
            if any(bomb_reaches(loc, e, belief) for e in threats):
                return PLACE_BOMB
            target = min(threats, key=lambda e: planner.dist_to(e))
            gs = belief.prior.grid_size
            best, best_tile = INF, None
            for x in range(gs):
                for y in range(gs):
                    if not bomb_reaches((x, y), target, belief):
                        continue
                    d = planner.dist_to((x, y))
                    if d < best:
                        best, best_tile = d, (x, y)
            if best_tile is not None and best != INF:
                action = planner.first_action(best_tile)
                if action is not None:
                    return action

    # Return home if outside the leash.
    if chebyshev(loc, base) > leash:
        return planner.first_action(base)

    return None


def hunt(belief, danger, planner, params):
    """Drop one bomb on enemy agents in blast range, then move on.

    Bomb-conservation rule: while any enemy base is still alive, hunt holds
    fire unless `team_bombs >= params.hunt_bomb_floor`. A bomb spent on an
    enemy agent is at most ~+15 (freeze) + a few damage points, vs +50 for a
    base kill — so we hoard the stockpile for base offense. Once all enemy
    bases are dead the bomb pool has no other use, so hunt fires on any
    visible live enemy in blast range (subject to bombs in hand).

    One bomb is enough: an enemy a bomb we ALREADY placed (still in flight)
    would hit is skipped, so hunt does not re-bomb the same stationary enemy
    tick after tick — the cascade falls through to movement once the bomb is
    down. Counts only LIVE enemies the bomb would actually hit (LOS + Chebyshev
    2); frozen enemies are excluded. Our bomb is friendly-fire safe, so no
    escape route is needed.

    `danger`, `planner` are unused — kept for cascade-uniform layer typing.
    """
    _trace_decision(belief, "hunt", "team_bombs", belief.team_bombs)
    if belief.team_bombs <= 0:
        _trace_decision(belief, "hunt", "yield_no_bombs", True)
        return None
    loc = belief.location
    hits = sum(1 for e in belief.live_enemies()
               if bomb_reaches(loc, e, belief)
               and not any(bomb_reaches(cell, e, belief)
                           for cell, _ in belief.own_bombs))
    _trace_decision(belief, "hunt", "hits", hits)
    if hits == 0:
        _trace_decision(belief, "hunt", "yield_no_hits", True)
        return None
    bases_alive = bool(belief.live_enemy_bases())
    _trace_decision(belief, "hunt", "bases_alive", bases_alive)
    if bases_alive and belief.team_bombs < params.hunt_bomb_floor:
        _trace_decision(belief, "hunt", "yield_save_for_bases", True)
        return None
    _trace_decision(belief, "hunt", "place_bomb", True)
    return PLACE_BOMB


def hold(belief, danger, planner, params):
    """Final fallback for the camper — stay on station."""
    return STAY
