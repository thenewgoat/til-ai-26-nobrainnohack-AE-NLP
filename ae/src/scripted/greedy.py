"""Greedy AE strategy: A* to nearest live enemy base, bomb when in range.

This module is a deliberate departure from the cascade in `decide.py`. It
imports no other scripted modules except `geometry` for the action and
direction integers — all perception (walls, dead bases, frozen ticks) is read
from the `Belief` instance the manager hands in.
"""
import heapq

from scripted.geometry import (
    BACKWARD, FORWARD, LEFT, MOVE, PLACE_BOMB, RIGHT, STAY, chebyshev,
)


def _h(a, b):
    """Manhattan heuristic — admissible for 4-connected unit-cost grids."""
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def _neighbours(cell, grid_size):
    """4-connected in-bounds neighbours of `cell`."""
    x, y = cell
    for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        nx, ny = x + dx, y + dy
        if 0 <= nx < grid_size and 0 <= ny < grid_size:
            yield (nx, ny)


def _reconstruct(came_from, end):
    path = [end]
    while end in came_from:
        end = came_from[end]
        path.append(end)
    path.reverse()
    return path


def _astar(belief, start, goal):
    """Shortest 4-connected path through `belief.is_wall`; `[]` if unreachable."""
    if start == goal:
        return [start]
    gs = belief.prior.grid_size
    open_heap = [(_h(start, goal), 0, start)]
    came_from = {}
    g_score = {start: 0}
    while open_heap:
        _, g, cur = heapq.heappop(open_heap)
        if cur == goal:
            return _reconstruct(came_from, cur)
        if g > g_score[cur]:
            continue
        for nb in _neighbours(cur, gs):
            if belief.is_wall(cur, nb):
                continue
            tentative = g + 1
            if tentative < g_score.get(nb, float("inf")):
                g_score[nb] = tentative
                came_from[nb] = cur
                f = tentative + _h(nb, goal)
                heapq.heappush(open_heap, (f, tentative, nb))
    return []


def _astar_cost(belief, start, goal):
    """Path length (int) or `float('inf')` if unreachable."""
    path = _astar(belief, start, goal)
    return len(path) - 1 if path else float("inf")


def _first_legal(mask, preference):
    """First action in `preference` legal under `mask`; else any legal action;
    else `STAY`. Mirrors decide._first_legal so greedy.py can stay independent
    of decide.py."""
    for a in preference:
        if 0 <= a < len(mask) and mask[a] == 1:
            return a
    for a in range(len(mask)):
        if mask[a] == 1:
            return a
    return STAY


def _direction_from(loc, next_cell):
    """Return the MOVE-direction integer d such that loc + MOVE[d] == next_cell,
    or None if next_cell is not 4-adjacent to loc."""
    dx, dy = next_cell[0] - loc[0], next_cell[1] - loc[1]
    for d, (mx, my) in MOVE.items():
        if (mx, my) == (dx, dy):
            return d
    return None


def _cell_to_action(next_cell, loc, facing, mask):
    """Pick an action that steps the agent into `next_cell` from `loc`.

    Preference: FORWARD when facing the step, BACKWARD when facing away (saves
    a turn), otherwise a single LEFT or RIGHT turn toward the step. Falls
    through to any legal action when the preferred one is masked.
    """
    d = _direction_from(loc, next_cell)
    if d is None:
        return _first_legal(mask, [FORWARD, RIGHT, LEFT, BACKWARD, STAY])
    if facing == d and mask[FORWARD] == 1:
        return FORWARD
    if facing == (d + 2) % 4 and mask[BACKWARD] == 1:
        return BACKWARD
    if (d - facing) % 4 == 1 and mask[RIGHT] == 1:
        return RIGHT
    if (facing - d) % 4 == 1 and mask[LEFT] == 1:
        return LEFT
    # 180-degree case where BACKWARD was masked, or any other miss.
    return _first_legal(mask, [FORWARD, RIGHT, LEFT, BACKWARD, STAY])


def act(belief, action_mask):
    """Greedy AE decision: A* to the nearest live enemy base, bomb when in range.

    Pure function of the current belief and action mask. `belief.update` must
    have been called for this tick before this function is called.
    """
    mask = list(action_mask)
    if belief.frozen_ticks > 0:
        return _first_legal(mask, [STAY])

    live = belief.live_enemy_bases()
    if not live:
        return _first_legal(mask, [FORWARD, RIGHT, LEFT, BACKWARD, STAY])

    loc = belief.location
    target = min(live, key=lambda b: _astar_cost(belief, loc, b))

    if (chebyshev(loc, target) <= 2 and belief.team_bombs > 0
            and mask[PLACE_BOMB] == 1):
        return PLACE_BOMB

    path = _astar(belief, loc, target)
    if len(path) < 2:
        return _first_legal(mask, [FORWARD, RIGHT, LEFT, BACKWARD, STAY])
    return _cell_to_action(path[1], loc, belief.facing, mask)
