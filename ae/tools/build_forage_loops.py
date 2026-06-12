"""Offline forage-loop solver for the AE novice map.

Reads ae/src/arena_map.json + ae/src/respawn_map.json, finds high-yield cyclic
foraging routes, and writes ae/src/forage_loops.json. Run once offline; the
output is shipped in the container and patrolled by the `forage_loop` layer.

Stdlib only — imports nothing from scripted/. Direction ints match
scripted.geometry: RIGHT=0, DOWN=1, LEFT=2, UP=3.
"""
import json
from collections import deque
from pathlib import Path

INF = float("inf")
_MOVE = {0: (1, 0), 1: (0, 1), 2: (-1, 0), 3: (0, -1)}
_VALUE = {"mission": 5.0, "recon": 1.0, "resource": 2.0}


def chebyshev(a, b):
    """Chebyshev (chessboard) distance between two (x, y) tiles."""
    return max(abs(a[0] - b[0]), abs(a[1] - b[1]))


def load_arena(path):
    """Parse arena_map.json into {grid_size, blocked, collectibles, bases}.

    `blocked` is a set of frozenset{tileA, tileB} edge pairs that cannot be
    crossed — every wall (destructible or not) blocks, since the forager never
    bombs. `collectibles` maps (x, y) -> kind; on a stacked tile the last
    listed kind wins (a minor simplification — stacks are rare).
    """
    data = json.loads(Path(path).read_text())
    grid_size = int(data["grid_size"])
    blocked = set()
    for ax, ay, direction, _destructible in data["walls"]:
        dx, dy = _MOVE[direction]
        blocked.add(frozenset({(ax, ay), (ax + dx, ay + dy)}))
    collectibles = {(x, y): kind for x, y, kind in data["collectibles"]}
    bases = {int(t): tuple(p) for t, p in data["bases"].items()}
    return {"grid_size": grid_size, "blocked": blocked,
            "collectibles": collectibles, "bases": bases}


def load_respawn(path):
    """Parse respawn_map.json (nested list indexed [x][y]) into {(x,y): delay}."""
    data = json.loads(Path(path).read_text())
    return {(x, y): int(data[x][y])
            for x in range(len(data)) for y in range(len(data[x]))}


# ---------------------------------------------------------------------------
# Tour optimisation and loop scoring.
# ---------------------------------------------------------------------------

def route_ticks(blocked, grid_size, start, start_facing, goal):
    """Min ticks from tile `start` facing `start_facing` to tile `goal`.

    BFS over (tile, facing) states — every action (FORWARD, BACKWARD, LEFT,
    RIGHT) costs one tick; all-unit edges make a deque BFS exact. Returns
    (ticks, facing_on_arrival); (0, start_facing) when start == goal;
    (INF, start_facing) when `goal` is unreachable.
    """
    start, goal = tuple(start), tuple(goal)
    if start == goal:
        return (0, start_facing)
    seen = {(start, start_facing)}
    q = deque([(start, start_facing, 0)])
    while q:
        tile, facing, cost = q.popleft()
        nc = cost + 1
        # Turns — same tile, new facing.
        for nf in ((facing + 1) % 4, (facing + 3) % 4):
            if (tile, nf) not in seen:
                seen.add((tile, nf))
                q.append((tile, nf, nc))
        # Forward / backward — new tile, facing unchanged.
        for mdir in (facing, (facing + 2) % 4):
            dx, dy = _MOVE[mdir]
            nb = (tile[0] + dx, tile[1] + dy)
            if not (0 <= nb[0] < grid_size and 0 <= nb[1] < grid_size):
                continue
            if frozenset({tile, nb}) in blocked:
                continue
            if nb == goal:
                return (nc, facing)
            if (nb, facing) not in seen:
                seen.add((nb, facing))
                q.append((nb, facing, nc))
    return (INF, start_facing)


def cluster_collectibles(collectibles, radius):
    """Greedy clustering of collectible tiles by proximity.

    `collectibles` maps (x, y) -> kind. Repeatedly seed a new cluster at the
    highest-value unassigned tile and absorb every unassigned tile within
    Chebyshev `radius` of the seed. Returns a list of clusters, each a sorted
    list of (x, y) tiles.
    """
    unassigned = dict(collectibles)
    clusters = []
    while unassigned:
        seed = max(unassigned, key=lambda t: (_VALUE[unassigned[t]], t))
        members = [t for t in unassigned if chebyshev(t, seed) <= radius]
        for t in members:
            del unassigned[t]
        clusters.append(sorted(members))
    return clusters


def nearest_neighbour_tour(tiles, distfn):
    """Greedy nearest-neighbour tour over `tiles`, starting at tiles[0].

    `distfn(a, b)` is the edge cost. Ties broken by tile coordinate for
    determinism. Returns an ordered list (a cycle is implied: last -> first).
    """
    if not tiles:
        return []
    tour = [tiles[0]]
    remaining = list(tiles[1:])
    while remaining:
        last = tour[-1]
        nxt = min(remaining, key=lambda t: (distfn(last, t), t))
        tour.append(nxt)
        remaining.remove(nxt)
    return tour


def _cycle_len(tour, distfn):
    """Total cost of the closed tour (last waypoint wraps to the first)."""
    n = len(tour)
    return sum(distfn(tour[i], tour[(i + 1) % n]) for i in range(n))


def two_opt(tour, distfn, max_passes=4):
    """2-opt improvement of a closed tour: reverse segments while that lowers
    the cycle length. Stops at a local optimum or after `max_passes`."""
    tour = list(tour)
    n = len(tour)
    if n < 4:
        return tour
    for _ in range(max_passes):
        improved = False
        for i in range(n - 1):
            for j in range(i + 2, n):
                if i == 0 and j == n - 1:
                    continue                      # adjacent on the cycle
                a, b = tour[i], tour[i + 1]
                c, d = tour[j], tour[(j + 1) % n]
                if distfn(a, c) + distfn(b, d) < distfn(a, b) + distfn(c, d):
                    tour[i + 1:j + 1] = reversed(tour[i + 1:j + 1])
                    improved = True
        if not improved:
            break
    return tour


def loop_period(tour, blocked, grid_size):
    """Ticks to traverse the closed loop `tour` once.

    Facing is threaded leg-to-leg (each leg starts in the facing the previous
    leg ended in), and the result is minimised over the 4 possible starting
    facings. Returns INF if any leg is unroutable; 0 for a degenerate tour.
    """
    if len(tour) < 2:
        return 0
    best = INF
    for start_facing in range(4):
        total = 0
        facing = start_facing
        ok = True
        for i in range(len(tour)):
            a = tour[i]
            b = tour[(i + 1) % len(tour)]
            ticks, facing = route_ticks(blocked, grid_size, a, facing, b)
            if ticks == INF:
                ok = False
                break
            total += ticks
        if ok and total < best:
            best = total
    return best


def v_eff(kind, bomb_value):
    """Effective value of a collectible tile.

    Direct reward plus, for `resource` tiles only, the bomb-fuel worth: a
    resource adds 0.5 to the pool and `bomb_cost` is 1.5, so it is (0.5 / 1.5)
    of a bomb, worth `bomb_value` apiece. Mission/recon give reward only.
    """
    base = _VALUE[kind]
    if kind == "resource":
        return base + (0.5 / 1.5) * bomb_value
    return base


def loop_yield_rate(tour, period, collectibles, respawn, bomb_value):
    """Steady-state reward-per-tick of a loop.

    Σ over waypoint tiles of `v_eff(t) * min(1, period / respawn_delay(t))`,
    divided by `period`. The `min` term models a tile being ready only a
    fraction of revisits when the loop is faster than the tile's refill.
    """
    if period <= 0:
        return 0.0
    total = 0.0
    for t in tour:
        total += (v_eff(collectibles[t], bomb_value)
                  * min(1.0, period / respawn[t]))
    return total / period


def build_loops(arena, respawn, cluster_radius,
                bomb_value_attack, bomb_value_endgame):
    """Build one scored loop per collectible cluster.

    Each loop dict has: `waypoints` (ordered list of [x, y]), `period` (ticks),
    `yield_attack` / `yield_endgame` (steady-state reward-per-tick at the two
    bomb-value regimes), and `resource_leaning` (True when >=1/3 of the
    waypoints are resource tiles). Clusters of fewer than 2 tiles, and clusters
    with no routable loop, are skipped.
    """
    blocked = arena["blocked"]
    grid_size = arena["grid_size"]
    collectibles = arena["collectibles"]

    def dist(a, b):
        return route_ticks(blocked, grid_size, a, 0, b)[0]

    loops = []
    for members in cluster_collectibles(collectibles, cluster_radius):
        if len(members) < 2:
            continue
        tour = two_opt(nearest_neighbour_tour(members, dist), dist)
        period = loop_period(tour, blocked, grid_size)
        if period == INF or period <= 0:
            continue
        n_resource = sum(1 for t in tour if collectibles[t] == "resource")
        loops.append({
            "waypoints": [list(t) for t in tour],
            "period": period,
            "yield_attack": loop_yield_rate(tour, period, collectibles,
                                            respawn, bomb_value_attack),
            "yield_endgame": loop_yield_rate(tour, period, collectibles,
                                             respawn, bomb_value_endgame),
            "resource_leaning": n_resource * 3 >= len(tour),
        })
    return loops


def assign_teams(loops, bases):
    """Per-team loop metadata: the `home_loop` (loop whose nearest waypoint is
    closest to that team's base) and `order` (loop indices sorted by
    `yield_attack`, descending)."""
    teams = {}
    order = sorted(range(len(loops)),
                   key=lambda i: -loops[i]["yield_attack"])
    for team, base in bases.items():
        home = min(
            range(len(loops)),
            key=lambda i: min(chebyshev(tuple(w), base)
                              for w in loops[i]["waypoints"]),
        )
        teams[str(team)] = {"home_loop": home, "order": list(order)}
    return teams


def main():
    """Build forage_loops.json from the shipped novice map."""
    src = Path(__file__).resolve().parent.parent / "src"
    arena = load_arena(src / "arena_map.json")
    respawn = load_respawn(src / "respawn_map.json")
    loops = build_loops(arena, respawn, cluster_radius=3,
                        bomb_value_attack=10.0, bomb_value_endgame=0.0)
    teams = assign_teams(loops, arena["bases"])
    out = {"loops": loops, "teams": teams}
    (src / "forage_loops.json").write_text(json.dumps(out, indent=1))
    print(f"wrote {len(loops)} loops for {len(teams)} teams "
          f"-> {src / 'forage_loops.json'}")


if __name__ == "__main__":
    main()
