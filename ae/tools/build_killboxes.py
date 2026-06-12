"""Offline killbox solver.

For every pair `(A, E)` of tiles on the map, decide whether a bomb at `A`
guarantees a kill on an enemy stationed at `E`. The criterion:

1. A bomb at `A` reaches `E` (LOS-occluded Chebyshev-radius-2 blast).
2. From `E`, the enemy cannot reach a tile outside the blast within
   `ESCAPE_BUDGET` cardinal moves, treating both indestructible and
   destructible walls as blockers.

`ESCAPE_BUDGET = BOMB_TIMER`: `BOMB_TIMER` (in `scripted.pathfind`) is the
number of planner-phases from a bomb's placement step to the phase at which
its blast lands — exactly the number of deliberate enemy moves available
before the blast lands. The env's Bomb.__post_init__ pads the configured
4-tick fuse by 1 to compensate for the upkeep tick at the placement step,
AND the enemy gets a MOVE in the detonation step before DETONATE — both
factors are baked into the BOMB_TIMER value (= 5).

The set ships as `ae/src/killboxes.json` and is loaded by the trap layer in
`scripted.adaptive_layers`.
"""
import json
import sys
from collections import deque
from pathlib import Path

# Make `scripted/` importable when this file is run as a script.
_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from scripted.belief import Belief                    # noqa: E402
from scripted.blast import bomb_reaches               # noqa: E402
from scripted.map_prior import MapPrior               # noqa: E402
from scripted.pathfind import BOMB_TIMER              # noqa: E402

_OUTPUT_PATH = (Path(__file__).resolve().parent.parent
                / "src" / "killboxes.json")

ESCAPE_BUDGET = BOMB_TIMER       # planner-phases from placement to lethal phase


def _belief_from_prior(prior):
    """Build a minimal Belief that `bomb_reaches` can read.

    Sets the prior and an empty destroyed-walls set so all walls are intact.
    """
    b = Belief()
    b.prior = prior
    b.destroyed_walls = set()
    return b


def _blast_region(belief, bomb_cell):
    """Set of tiles the blast covers when a bomb is placed at `bomb_cell`."""
    gs = belief.prior.grid_size
    return {(x, y) for x in range(gs) for y in range(gs)
            if bomb_reaches(bomb_cell, (x, y), belief)}


def _enemy_escape_reachable(belief, enemy_cell, blast):
    """True if the enemy can reach a non-blast tile from `enemy_cell` in at
    most `ESCAPE_BUDGET` cardinal moves, treating any wall (destructible or
    not) as a blocker."""
    gs = belief.prior.grid_size
    seen = {enemy_cell}
    frontier = deque([(enemy_cell, 0)])
    while frontier:
        cell, depth = frontier.popleft()
        if cell not in blast:
            return True
        if depth >= ESCAPE_BUDGET:
            continue
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nb = (cell[0] + dx, cell[1] + dy)
            if nb in seen:
                continue
            if not (0 <= nb[0] < gs and 0 <= nb[1] < gs):
                continue
            if belief.is_wall(cell, nb):
                continue
            seen.add(nb)
            frontier.append((nb, depth + 1))
    return False


def _compute_killboxes(prior):
    """Return `set[((ax, ay), (ex, ey))]` — every `(agent_tile, enemy_tile)`
    pair where bombing from `agent_tile` guarantees a kill on an enemy at
    `enemy_tile` before the bomb detonates (ESCAPE_BUDGET-move horizon)."""
    belief = _belief_from_prior(prior)
    gs = prior.grid_size
    killboxes = set()
    for ax in range(gs):
        for ay in range(gs):
            blast = _blast_region(belief, (ax, ay))
            for ex, ey in blast:
                if _enemy_escape_reachable(belief, (ex, ey), blast):
                    continue
                killboxes.add(((ax, ay), (ex, ey)))
    return killboxes


def main():
    prior = MapPrior.load()
    killboxes = _compute_killboxes(prior)
    payload = {"killboxes": [[list(a), list(e)] for a, e in sorted(killboxes)]}
    _OUTPUT_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"wrote {_OUTPUT_PATH} — {len(killboxes)} killbox pairs")


if __name__ == "__main__":
    main()
