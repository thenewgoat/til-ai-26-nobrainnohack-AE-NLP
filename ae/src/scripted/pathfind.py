"""Time-aware planner over (tile, facing, phase) states.

Every action — turn LEFT/RIGHT, move FORWARD/BACKWARD, or STAY — costs one
game tick. A destructible wall is traversable only once a bomb has opened it
(at that bomb's detonation tick); an indestructible wall always blocks. A
state whose (cell, tick) an enemy blast covers is forbidden. Enemy bombs
detonate within BOMB_TIMER ticks and blasts are instantaneous, so past T_MAX
no hazard remains and no further wall opens — `phase` saturates at T_MAX and
the search collapses to an ordinary static Dijkstra tail.
"""
import heapq

from scripted.blast import walls_destroyed_by
from scripted.geometry import (
    BACKWARD, FORWARD, LEFT, MOVE, PLACE_BOMB, RIGHT, STAY,
)

BOMB_TIMER = 5                  # planner-phases from a bomb's placement step (= action 1) to the
                                # phase at which its blast damages — the env's bomb fuse is 4
                                # internal-timer ticks PLUS the post-init +1 the env adds to
                                # compensate for the placement-step upkeep, AND DETONATE runs
                                # AFTER MOVE in the detonation step, so the lethal phase is
                                # env_timer + 1 from the observer's POV.
T_MAX = BOMB_TIMER + 2          # past this tick no hazard remains; phase saturates
INF = float("inf")
BACKWARD_COST = 1.4             # KNOB: weight on a BACKWARD action — every other action costs 1.0.
                                # >1 makes the planner prefer routes that face their target;
                                # affects `dist_to`/`steps_to` (now floats — callers compare, not index).
BLACKLIST_COST = 1000.0          # KNOB: weighted-cost surcharge for traversing a
                                  # tile in belief.stuck_blacklist. Large enough that
                                  # the planner only uses these tiles when no alternative
                                  # exists; small enough that "no alternative" still
                                  # yields a finite cost (so cascade fallthrough can
                                  # decide based on relative magnitudes).
BLACKLIST_JITTER = 0.5           # ± slot-seeded jitter in ticks on the blacklist cost,
                                  # breaks symmetry between mirror-strategy agents.


class Planner:
    """Time-expanded search result: per-tile earliest weighted arrival cost + routes.

    Action costs are 1.0 for FORWARD/LEFT/RIGHT/STAY and `BACKWARD_COST` for
    BACKWARD, so a `dist_to`/`steps_to` value is a *weighted* arrival cost
    rather than a pure tick count — equal to the tick count whenever the route
    contains no BACKWARD steps. Callers compare these scalars (`<`, `>=`,
    `INF`); none index them or rely on integer arithmetic."""

    def __init__(self, start, cost, prev):
        self._start = start
        self._prev = prev                       # state -> (prev_state, action)
        # Collapse the (tile, facing, phase) cost map to per-tile bests.
        tile_best, tile_goal = {}, {}
        for state, c in cost.items():
            tile = state[0]
            if c < tile_best.get(tile, INF):
                tile_best[tile], tile_goal[tile] = c, state
        self._tile_best = tile_best
        self._tile_goal = tile_goal

        # Survival aggregates for the forced-escape floor (Plan 2). `phase`
        # saturates at T_MAX (the hazard-free tail), so a reachable state at
        # phase T_MAX means the agent stayed non-lethal until every known enemy
        # bomb had detonated. `_deepest_*` describe the deepest reachable phase
        # frontier (where a non-survivor ultimately dies).
        max_phase = 0
        reached_tail = False
        deepest_tiles = set()
        deepest_cost = INF
        for (t, _f, ph), cc in cost.items():
            if ph > max_phase:
                max_phase, deepest_tiles, deepest_cost = ph, {t}, cc
            elif ph == max_phase:
                deepest_tiles.add(t)
                if cc < deepest_cost:
                    deepest_cost = cc
            if ph == T_MAX:
                reached_tail = True
        self._max_phase = max_phase
        self._reached_tail = reached_tail
        self._deepest_tiles = deepest_tiles
        self._deepest_cost = deepest_cost

    def dist_to(self, tile):
        """Earliest weighted arrival cost at `tile` in any facing (INF if
        unreachable). FORWARD/LEFT/RIGHT/STAY weigh 1.0, BACKWARD weighs
        `BACKWARD_COST`."""
        return self._tile_best.get(tuple(tile), INF)

    def steps_to(self, tile):
        """Alias for `dist_to`. Kept as a separate name for callers (survive)
        that read it as a step count — value is identical to `dist_to` and
        coincides with the integer action count when no BACKWARD is used."""
        return self.dist_to(tile)

    def first_action(self, tile):
        """First action of the earliest route to `tile`; None if unreachable
        or already there."""
        goal = self._tile_goal.get(tuple(tile))
        if goal is None or goal == self._start:
            return None
        s, action = goal, None
        while s != self._start:
            s, action = self._prev[s]
        return action

    def route_to(self, tile):
        """Tiles along the earliest route to `tile`, start cell excluded
        (turn-in-place steps repeat a tile). Empty if unreachable or already
        there."""
        goal = self._tile_goal.get(tuple(tile))
        if goal is None or goal == self._start:
            return []
        cells, s = [], goal
        while s != self._start:
            cells.append(s[0])
            s, _ = self._prev[s]
        cells.reverse()
        return cells

    def has_safe_continuation(self):
        """True iff some reachable state reaches the hazard-free tail (phase
        T_MAX) — i.e. the agent can stay non-lethal until every known enemy bomb
        has detonated. Meaningful for a planner built with `forced_first_action`
        (the forced-escape floor)."""
        return self._reached_tail

    def survival_score(self, danger):
        """Lexicographic survival quality of this (forced-first-action) plan;
        higher is better:
            (survives, deepest_non_lethal_phase, -min_overlap_at_frontier,
             -weighted_route_cost_to_frontier)
        WHETHER to intervene is decided on the survival prefix `[:3]` only; the
        cost slot is for CHOOSING among actions once intervention is decided
        (all survivors tie on `[:3]`)."""
        # Survivors always reach phase T_MAX, which is past every hazard, so their
        # frontier tiles have overlap 0 — hence all survivors tie on slots 1-2 and
        # the floor's `[:3]`-prefix compare never separates them.
        min_overlap = min((danger.overlap(t) for t in self._deepest_tiles), default=0)
        survives = 1 if self._reached_tail else 0
        return (survives, self._max_phase, -min_overlap, -self._deepest_cost)


def _wall_open_ticks(belief, place_bomb_first):
    """Map each destructible wall pair to the earliest tick a known bomb opens
    it. Enemy, ally, and our own bombs all open their walls at their remaining-
    timer tick. With `place_bomb_first`, a hypothetical bomb dropped on the
    agent's tile by the forced opening PLACE_BOMB (tick 1) opens its walls at
    1 + BOMB_TIMER."""
    sources = (list(belief.enemy_bombs.items())
               + list(belief.ally_bombs.items())
               + [(cell, timer) for cell, timer in belief.own_bombs])
    if place_bomb_first:
        sources.append((tuple(belief.location), 1 + BOMB_TIMER))
    opens = {}
    for cell, detonation in sources:
        for pair in walls_destroyed_by(cell, belief):
            if detonation < opens.get(pair, INF):
                opens[pair] = detonation
    return opens


def _edge_passable(belief, a, b, arrival_tick, open_ticks):
    """True if the agent may cross from tile a to adjacent tile b, arriving at
    `arrival_tick`. A destructible wall is crossable only AFTER the tick a
    bomb opens it (the wall opens during DETONATE, which runs after the MOVE
    of that same step — so a cross at arrival_tick == open_tick still sees the
    wall intact); an indestructible wall never is."""
    pair = frozenset({a, b})
    if pair in belief.destroyed_walls or pair not in belief.prior.wall_between:
        return True
    if not belief.prior.wall_between[pair]:              # indestructible
        return False
    return arrival_tick > open_ticks.get(pair, INF)      # destructible


def build_planner(belief, danger, place_bomb_first=False, forced_first_action=None):
    """Dijkstra over (tile, facing, phase) from the agent's current state.

    `phase` = min(tick, T_MAX); states at phase T_MAX form the hazard-free
    static tail. With `place_bomb_first`, the only action out of the start
    state is PLACE_BOMB — used to price the 'breach now' scenario.

    Costs (and the ticks `Planner.dist_to` / `steps_to` return) count actions
    taken from now, not absolute game-step numbers.
    """
    gs = belief.prior.grid_size
    blacklist = {t for t, exp in belief.stuck_blacklist.items() if exp > belief.step}
    team = getattr(belief.prior, "team", 0)

    def blacklist_surcharge(tile):
        """0 if not blacklisted; BLACKLIST_COST + per-slot jitter otherwise."""
        if tile not in blacklist:
            return 0.0
        # Use hash for a deterministic per-(slot, tile) signed value.
        j = ((hash((team, tile[0], tile[1])) % 101) - 50) / 100.0 * BLACKLIST_JITTER * 2
        return BLACKLIST_COST + j

    open_ticks = _wall_open_ticks(belief, place_bomb_first)
    start = (tuple(belief.location), int(belief.facing), 0)
    cost = {start: 0.0}                         # state -> weighted cost
    prev = {}
    pq = [(0.0, start)]
    while pq:
        c, state = heapq.heappop(pq)
        if c > cost.get(state, INF):
            continue
        tile, facing, phase = state
        nphase = min(phase + 1, T_MAX)

        # `c` / `nphase` are final for this iteration and `relax` is called
        # synchronously below — no late-binding closure hazard. `nc` is now
        # action-dependent (BACKWARD costs more than 1), so it lives inside relax.
        def relax(ntile, nfacing, action, step_cost=1.0):
            nc = c + step_cost
            # Lethality is checked at the *arrival tick*, not the weighted cost
            # — every action advances one tick regardless of weight, so use
            # phase+1 (== nphase) for the danger check.
            if danger.is_lethal_at(ntile, nphase):
                return
            ns = (ntile, nfacing, nphase)
            if nc < cost.get(ns, INF):
                cost[ns] = nc
                prev[ns] = (state, action)
                heapq.heappush(pq, (nc, ns))

        if place_bomb_first and state == start:
            relax(tile, facing, PLACE_BOMB)         # forced opening move
            continue

        if forced_first_action is not None and state == start:
            # Force the FIRST action only; the planner then explores freely from
            # the resulting state. Movement actions still obey bounds / frozen
            # enemies / edge passability — a blocked forced move yields no
            # successor (the agent "dies in place" for survival-scoring). The
            # escape floor never forces PLACE_BOMB (the floor is bomb-free).
            a = forced_first_action
            if a == STAY:
                relax(tile, facing, STAY)
            elif a == LEFT:
                relax(tile, (facing + 3) % 4, LEFT)
            elif a == RIGHT:
                relax(tile, (facing + 1) % 4, RIGHT)
            elif a in (FORWARD, BACKWARD):
                mdir = facing if a == FORWARD else (facing + 2) % 4
                step = 1.0 if a == FORWARD else BACKWARD_COST
                dx, dy = MOVE[mdir]
                nb = (tile[0] + dx, tile[1] + dy)
                if (0 <= nb[0] < gs and 0 <= nb[1] < gs
                        and nb not in belief.frozen_enemies
                        and _edge_passable(belief, tile, nb, nphase, open_ticks)):
                    relax(nb, facing, a, step_cost=step + blacklist_surcharge(nb))
            continue

        # Turns and STAY keep the agent on `tile`.
        relax(tile, (facing + 3) % 4, LEFT)
        relax(tile, (facing + 1) % 4, RIGHT)
        relax(tile, facing, STAY)
        # Forward / backward moves. BACKWARD pays `BACKWARD_COST` instead of 1
        # so the planner prefers routes that face their target.
        for action, mdir, step in (
            (FORWARD, facing, 1.0),
            (BACKWARD, (facing + 2) % 4, BACKWARD_COST),
        ):
            dx, dy = MOVE[mdir]
            nb = (tile[0] + dx, tile[1] + dy)
            if not (0 <= nb[0] < gs and 0 <= nb[1] < gs):
                continue
            if nb in belief.frozen_enemies:
                continue
            # Edge passability is keyed by arrival *tick*, not weighted cost.
            if not _edge_passable(belief, tile, nb, nphase, open_ticks):
                continue
            relax(nb, facing, action, step_cost=step + blacklist_surcharge(nb))
    return Planner(start, cost, prev)
