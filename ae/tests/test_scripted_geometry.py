"""view_to_world must match til_environment.helpers.view_to_world exactly."""
import numpy as np
from til_environment.helpers import view_to_world as env_v2w

from scripted.geometry import view_to_world


def test_matches_env_for_all_directions():
    for facing in range(4):
        for vx in range(-2, 5):
            for vy in range(-2, 3):
                ours = view_to_world((14, 9), facing, (vx, vy))
                env = env_v2w(np.array([14, 9]), facing, np.array([vx, vy]))
                assert tuple(int(c) for c in env) == ours, (facing, vx, vy)


def test_self_cell_is_identity():
    for facing in range(4):
        assert view_to_world((7, 7), facing, (0, 0)) == (7, 7)


def test_chebyshev_distance():
    from scripted.geometry import chebyshev
    assert chebyshev((0, 0), (3, 1)) == 3
    assert chebyshev((2, 5), (2, 5)) == 0
    assert chebyshev((1, 1), (-2, 0)) == 3
