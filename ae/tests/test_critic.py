"""CentralizedCritic: consumes an encoded global state, outputs a scalar value.

The critic no longer reads env.state() (a walls-only grid, constant within an
episode). It consumes `encode_global_state`, an agent-relative encoding built
from the entity registry.
"""
import numpy as np
import torch

from critic import (CentralizedCritic, encode_global_state, STATE_PLANES,
                    STATE_SCALARS, P_SELF, P_ENEMIES, P_COLLECT)
from til_environment import bomberman_env
from til_environment.config import default_config


def _env(seed=0):
    cfg = default_config()
    cfg.env.novice = True
    env = bomberman_env.basic_env(env_wrappers=[], cfg=cfg)
    env.reset(seed=seed)
    return env


def test_critic_forward_shape():
    net = CentralizedCritic()
    planes = torch.zeros(5, STATE_PLANES, 16, 16)
    scalars = torch.zeros(5, STATE_SCALARS)
    value = net(planes, scalars)
    assert value.shape == (5,)


def test_encode_global_state_from_real_env():
    env = _env()
    planes, scalars = encode_global_state(env.unwrapped.dynamics, "agent_0", 0)
    env.close()
    assert planes.shape == (STATE_PLANES, 16, 16)
    assert planes.dtype == np.float32
    assert scalars.shape == (STATE_SCALARS,)
    # the encoding is non-trivial: agents/bases/collectibles are marked
    assert planes.sum() > 0
    assert planes[P_COLLECT].sum() > 0
    # exactly one 'self' cell, and the five other agents on the enemy plane
    assert planes[P_SELF].sum() == 1.0
    assert planes[P_ENEMIES].sum() == 5.0


def test_encode_is_agent_relative():
    """The blind env.state() gave every slot an identical input. The registry
    encoding must differ per slot — that is the whole point of the fix."""
    env = _env()
    dyn = env.unwrapped.dynamics
    p0, s0 = encode_global_state(dyn, "agent_0", 0)
    p1, s1 = encode_global_state(dyn, "agent_1", 0)
    env.close()
    assert not np.array_equal(p0, p1)          # self/enemy planes swap
    assert not np.array_equal(p0[P_SELF], p1[P_SELF])


def test_blast_plane_matches_env_los_occlusion():
    """P_BLAST must equal til_environment._directional_blast Pass 1 exactly:
    the Chebyshev blast square minus tiles a wall occludes (no LOS)."""
    import random
    from til_environment.entities import Bomb
    from til_environment.helpers import _los_to_tile
    from critic import P_BLAST

    env = _env(seed=3)
    dyn = env.unwrapped.dynamics
    # roll forward until bombs are on the board
    random.seed(3)
    n = 0
    for slot in env.agent_iter():
        obs, _, term, trunc, _ = env.last()
        if term or trunc:
            env.step(None); continue
        legal = np.flatnonzero(np.asarray(obs["action_mask"], bool).reshape(-1))
        env.step(int(random.choice(legal)) if len(legal) else 4)
        n += 1
        if n > 180:
            break
    bombs = [b for b in dyn.registry.query().type(Bomb) if b.alive]
    planes, _ = encode_global_state(dyn, "agent_0", 0)

    state = dyn.arena_state._state
    gs = dyn.grid_size
    ref = set()
    for b in bombs:
        x, y, r = int(b.position[0]), int(b.position[1]), b.blast_radius
        for tx in range(max(0, x - r), min(gs, x + r + 1)):
            for ty in range(max(0, y - r), min(gs, y + r + 1)):
                if _los_to_tile(x, y, tx, ty, state):
                    ref.add((tx, ty))
    env.close()
    assert bombs, "expected bombs on the board for this test"
    encoded = set(map(tuple, np.argwhere(planes[P_BLAST] > 0)))
    assert encoded == ref


def test_critic_consumes_encoded_state():
    env = _env()
    planes, scalars = encode_global_state(env.unwrapped.dynamics, "agent_0", 0)
    env.close()
    value = CentralizedCritic()(
        torch.from_numpy(planes).unsqueeze(0),
        torch.from_numpy(scalars).unsqueeze(0),
    )
    assert value.shape == (1,)
