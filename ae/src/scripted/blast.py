"""Predict which destructible walls an ally bomb's blast opens.

Faithful port of til_environment's bomb blast geometry (``supercover_line`` +
``_los_to_tile`` + ``_directional_blast`` Pass 2), reimplemented numpy-free
against the agent's own wall belief so it runs in the served container.

A bomb opens a destructible wall when the bomb has line-of-sight to at least
one of the two tiles that wall separates, within Chebyshev ``BLAST_RADIUS``.
"""
from scripted.geometry import MOVE

BLAST_RADIUS = 2


def supercover_line(start, end):
    """Tiles a straight line from `start` to `end` passes through.

    Vendored verbatim from til_environment.helpers.supercover_line.
    """
    x0, y0 = start
    x1, y1 = end
    dx, dy = x1 - x0, y1 - y0
    nx, ny = abs(dx), abs(dy)
    sign_x = 1 if dx > 0 else -1 if dx < 0 else 0
    sign_y = 1 if dy > 0 else -1 if dy < 0 else 0

    px, py = x0, y0
    tiles = [(px, py)]
    ix = iy = 0
    while ix < nx or iy < ny:
        if (1 + 2 * ix) * ny == (1 + 2 * iy) * nx:
            px += sign_x
            py += sign_y
            ix += 1
            iy += 1
        elif (1 + 2 * ix) * ny < (1 + 2 * iy) * nx:
            px += sign_x
            ix += 1
        else:
            py += sign_y
            iy += 1
        tiles.append((px, py))
    return tiles


def _wall(belief, tile, direction):
    """True if a wall (destructible or not) sits on `direction` side of `tile`."""
    dx, dy = MOVE[direction]
    return belief.is_wall(tile, (tile[0] + dx, tile[1] + dy))


def _los_to_tile(ox, oy, tx, ty, belief):
    """True if (tx, ty) has line-of-sight from (ox, oy) — no wall crossing.

    Vendored from til_environment.helpers._los_to_tile, with the env's raw
    wall-bit grid lookups replaced by `belief.is_wall`.
    """
    if tx == ox and ty == oy:
        return True
    path = supercover_line((ox, oy), (tx, ty))
    for i in range(len(path) - 1):
        cx, cy = path[i]
        nx, ny = path[i + 1]
        dx, dy = nx - cx, ny - cy
        if dx != 0 and dy != 0:
            h_dir = 0 if dx > 0 else 2          # RIGHT / LEFT
            v_dir = 1 if dy > 0 else 3          # DOWN / UP
            h_blocked = (_wall(belief, (cx, cy), h_dir)
                         or _wall(belief, (nx, cy), v_dir))
            v_blocked = (_wall(belief, (cx, cy), v_dir)
                         or _wall(belief, (cx, ny), h_dir))
            if h_blocked and v_blocked:
                return False
        else:
            d_val = 0 if dx == 1 else 1 if dy == 1 else 2 if dx == -1 else 3
            if _wall(belief, (cx, cy), d_val):
                return False
    return True


def walls_destroyed_by(bomb_cell, belief, blast_radius=BLAST_RADIUS):
    """Set of destructible wall pairs an ally bomb at `bomb_cell` opens.

    Pass 1 — collect blast cells: tiles within Chebyshev `blast_radius` that
    the bomb has line-of-sight to. Pass 2 — a destructible wall is opened if
    either tile it separates is a blast cell (and within radius). Mirrors
    til_environment `_directional_blast`.
    """
    ox, oy = bomb_cell
    gs = belief.prior.grid_size
    r = blast_radius

    reachable = set()
    for tx in range(max(0, ox - r), min(gs, ox + r + 1)):
        for ty in range(max(0, oy - r), min(gs, oy + r + 1)):
            if _los_to_tile(ox, oy, tx, ty, belief):
                reachable.add((tx, ty))

    destroyed = set()
    for pair, destructible in belief.prior.wall_between.items():
        if not destructible or pair in belief.destroyed_walls:
            continue
        a, b = tuple(pair)
        a_in = max(abs(a[0] - ox), abs(a[1] - oy)) <= r
        b_in = max(abs(b[0] - ox), abs(b[1] - oy)) <= r
        if not (a_in or b_in):
            continue
        if a in reachable or b in reachable:
            destroyed.add(pair)
    return destroyed


def bomb_reaches(bomb_cell, target, belief, blast_radius=BLAST_RADIUS):
    """True if a bomb placed at `bomb_cell` would hit `target`.

    Matches the env blast: the target must be within Chebyshev `blast_radius`
    AND have line-of-sight from the bomb (a wall in between stops the blast).
    """
    ox, oy = bomb_cell
    tx, ty = target
    if max(abs(tx - ox), abs(ty - oy)) > blast_radius:
        return False
    return _los_to_tile(ox, oy, tx, ty, belief)


class _WallsOpenedView:
    """Read-only belief view with `opened` walls treated as already destroyed.

    Exposes exactly the surface the blast geometry reads — `prior`,
    `destroyed_walls` and `is_wall` — so `bomb_reaches`/`walls_destroyed_by`
    can be evaluated against a hypothetical future wall state without
    mutating the real belief."""

    def __init__(self, belief, opened):
        self.prior = belief.prior
        self.destroyed_walls = belief.destroyed_walls | opened

    def is_wall(self, a, b):
        pair = frozenset({tuple(a), tuple(b)})
        if pair in self.destroyed_walls:
            return False
        return pair in self.prior.wall_between


def breach_bombs_needed(bomb_cell, target, belief, max_bombs):
    """Wall-opening bombs an agent must spend at `bomb_cell` before a bomb
    placed there damages `target`.

    Bombs placed on consecutive ticks detonate on consecutive ticks, so each
    bomb's blast sees every wall its predecessors opened. Returns 0 when
    `bomb_reaches` already holds, k <= max_bombs when k sequential bombs open
    enough destructible walls that bomb k+1 gains line-of-sight, or None when
    `target` is outside the blast radius, the blockage is indestructible, or
    more than `max_bombs` openers would be needed."""
    ox, oy = bomb_cell
    tx, ty = target
    if max(abs(tx - ox), abs(ty - oy)) > BLAST_RADIUS:
        return None
    view, opened = belief, set()
    for k in range(max_bombs + 1):
        if bomb_reaches(bomb_cell, target, view):
            return k
        new = walls_destroyed_by(bomb_cell, view)
        if not new:
            return None                  # fixpoint — LOS never opens
        opened |= new
        view = _WallsOpenedView(belief, opened)
    return None


def replay_blasts(bombs, target, belief):
    """Walk in-flight bombs in detonation order; walls open as they blow.

    `bombs` is an iterable of (cell, counts_for_damage) sorted by detonation
    tick. Returns (hits, opened): how many counted bombs damage `target`
    (a destructible wall opened by an earlier blast no longer blocks a later
    one), and every wall pair the whole sequence opens. The plain per-bomb
    `bomb_reaches` undercounts exactly the self-breach dump this enables —
    openers at a tile without LOS, damaging bombs behind them."""
    view, opened, hits = belief, set(), 0
    for cell, counted in bombs:
        if counted and bomb_reaches(cell, target, view):
            hits += 1
        new = walls_destroyed_by(cell, view)
        if new:
            opened |= new
            view = _WallsOpenedView(belief, opened)
    return hits, opened
