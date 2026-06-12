"""Pure geometry constants and helpers — numpy-free, container-safe.

view_to_world is vendored from til_environment.helpers.view_to_world so the
served container needs no til_environment dependency.
"""

# Action integers.
FORWARD, BACKWARD, LEFT, RIGHT, STAY, PLACE_BOMB = 0, 1, 2, 3, 4, 5
# Direction integers.
DIR_RIGHT, DIR_DOWN, DIR_LEFT, DIR_UP = 0, 1, 2, 3
# Direction -> (dx, dy) movement.
MOVE = {DIR_RIGHT: (1, 0), DIR_DOWN: (0, 1), DIR_LEFT: (-1, 0), DIR_UP: (0, -1)}


def chebyshev(a, b):
    """Chebyshev (chessboard) distance between two (x, y) tiles."""
    return max(abs(a[0] - b[0]), abs(a[1] - b[1]))


def view_to_world(loc, facing, view_coord):
    """Map a viewcone offset to a world (x, y) tuple.

    Args:
        loc: agent (x, y).
        facing: agent direction int (0=RIGHT,1=DOWN,2=LEFT,3=UP).
        view_coord: (vx, vy) where vx is the forward-axis offset (row - behind)
            and vy is the lateral-axis offset (col - left).
    """
    ax, ay = loc
    vx, vy = view_coord
    if facing == DIR_RIGHT:
        return (ax + vx, ay + vy)
    if facing == DIR_DOWN:
        return (ax - vy, ay + vx)
    if facing == DIR_LEFT:
        return (ax - vx, ay - vy)
    return (ax + vy, ay - vx)  # DIR_UP
