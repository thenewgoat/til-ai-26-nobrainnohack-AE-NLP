"""Centralized critic for AE self-play training (CTDE / MAPPO).

Training-only: never exported, never served. Zero inference cost in the
container.

The critic does NOT consume env.state() — that is a (16,16) uint8 wall-edge
grid which is constant within an episode (verified: 0/256 cells change over 40
steps) and identical for every agent slot, so it carries no value signal.

Instead `encode_global_state` assembles a rich GLOBAL state from the env's
entity registry — agent/bomb/base/collectible positions, health, bomb timers.
The encoding is agent-RELATIVE: the querying slot's own location/health/base
are marked on dedicated planes, so the single-output critic can distinguish
the several learner slots that share one env step (each has its own reward).
"""
import numpy as np
import torch
import torch.nn as nn

from til_environment.entities import Agent, Base, Bomb, Mission, Recon, Resource
from til_environment.helpers import _los_to_tile

GRID_SIZE = 16
STATE_PLANES = 9
STATE_SCALARS = 6

# --- grid plane indices (frozen contract) ---
P_WALLS = 0         # tile carries any intact wall edge
P_SELF = 1          # the querying slot's agent location (one-hot)
P_ENEMIES = 2       # every other agent's location
P_AGENT_HP = 3      # per-agent tile -> health / MAX_HEALTH
P_BOMB_TIMER = 4    # per-bomb tile -> blast imminence (1 = detonating now)
P_BLAST = 5         # tiles a live bomb will hit (LOS-occluded), imminence-weighted
P_SELF_BASE = 6     # the querying slot's base tile -> base_health / 100
P_ENEMY_BASE = 7    # enemy base tiles -> base_health / 100
P_COLLECT = 8       # summed collectible reward value per tile, normalized

# --- scalar vector indices ---
SC_HEALTH = 0       # self health / MAX_HEALTH
SC_FROZEN = 1       # self frozen_ticks / FROZEN_TICKS_NORM (clipped to [0, 1])
SC_RESOURCES = 2    # team_resources / RESOURCE_SCALE
SC_BOMBS = 3        # team_bombs / BOMB_SCALE
SC_STEP = 4         # step / NUM_ITERS
SC_TEAM = 5         # team / num_teams

MAX_HEALTH = 60.0
BASE_MAX_HEALTH = 100.0
# Normalizer for the SC_FROZEN scalar — a free choice, NOT the env's
# `freeze_turns`. env.freeze_turns=3 caps `frozen_ticks` at 3, so a /3.0 norm
# saturates exactly when frozen; /10.0 compresses to [0, 0.3] with headroom
# (and matches the value used in the pretrained critic checkpoint). Don't
# read this as "the freeze duration".
FROZEN_TICKS_NORM = 10.0
BOMB_FUSE = 4.0
NUM_ITERS = 200.0
RESOURCE_SCALE = 5.0
BOMB_SCALE = 10.0
COLLECT_NORM = 5.0


def _layer_init(layer, std=2.0 ** 0.5, bias_const=0.0):
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias_const)
    return layer


def encode_global_state(dynamics, slot, step):
    """Assemble the centralized critic's input from the entity registry.

    Args:
        dynamics: env.unwrapped.dynamics — holds the entity registry and the
            per-team resource/bomb pools.
        slot: the acting agent id ("agent_0".."agent_5"); its own
            location/health/base are marked on the *self* planes.
        step: the env step index, for the step scalar.
    Returns:
        (planes, scalars): float32 arrays, shape [STATE_PLANES,16,16] and
        [STATE_SCALARS].
    """
    reg = dynamics.registry
    planes = np.zeros((STATE_PLANES, GRID_SIZE, GRID_SIZE), np.float32)
    gs = GRID_SIZE

    def _ok(x, y):
        return 0 <= x < gs and 0 <= y < gs

    # walls: a tile carrying any intact wall edge (bits 0-3 of the arena grid)
    wall_grid = np.asarray(dynamics.arena_state._state, np.uint8)
    planes[P_WALLS] = ((wall_grid & 0x0F) != 0).astype(np.float32)

    self_ent = reg.get(slot)
    self_team = self_ent.team

    # agents — self vs the rest, plus a per-agent health field
    for ag in reg.query().type(Agent):
        if not ag.alive:
            continue
        x, y = int(ag.position[0]), int(ag.position[1])
        if not _ok(x, y):
            continue
        planes[P_SELF if ag.entity_id == slot else P_ENEMIES, x, y] = 1.0
        planes[P_AGENT_HP, x, y] = ag.health / MAX_HEALTH

    # bombs — location/imminence, plus the exact blast footprint
    for bomb in reg.query().type(Bomb):
        if not bomb.alive:
            continue
        x, y = int(bomb.position[0]), int(bomb.position[1])
        if not _ok(x, y):
            continue
        imminence = max(0.0, 1.0 - bomb.timer / BOMB_FUSE)
        planes[P_BOMB_TIMER, x, y] = max(planes[P_BOMB_TIMER, x, y], imminence)
        # blast footprint: every tile within the bomb's Chebyshev blast_radius
        # that the bomb has line-of-sight to — a wall (destructible OR not)
        # crossing the straight line occludes the blast. This is
        # til_environment.Dynamics._directional_blast Pass 1, using the env's
        # own _los_to_tile against the live wall grid.
        r = bomb.blast_radius
        for tx in range(max(0, x - r), min(gs, x + r + 1)):
            for ty in range(max(0, y - r), min(gs, y + r + 1)):
                if _los_to_tile(x, y, tx, ty, wall_grid):
                    planes[P_BLAST, tx, ty] = max(planes[P_BLAST, tx, ty],
                                                  imminence)

    # bases — self vs enemy, valued by remaining health
    for base in reg.query().type(Base):
        if not base.alive:
            continue
        x, y = int(base.position[0]), int(base.position[1])
        if not _ok(x, y):
            continue
        plane = P_SELF_BASE if base.team == self_team else P_ENEMY_BASE
        planes[plane, x, y] = base.health / BASE_MAX_HEALTH

    # collectibles — summed reward value per tile
    for cls in (Resource, Mission, Recon):
        for item in reg.query().type(cls):
            if not item.alive:
                continue
            x, y = int(item.position[0]), int(item.position[1])
            if not _ok(x, y):
                continue
            val = float(getattr(item, "reward_value",
                                getattr(item, "amount", 0.0)))
            planes[P_COLLECT, x, y] += val
    planes[P_COLLECT] = np.minimum(planes[P_COLLECT] / COLLECT_NORM, 1.0)

    # scalars — the acting agent's private status
    res = float(dynamics.team_resources.get(self_team, 0.0))
    bombs = float(dynamics.team_bombs.get(self_team, 0))
    n_teams = max(1, dynamics.num_teams)
    scalars = np.zeros(STATE_SCALARS, np.float32)
    scalars[SC_HEALTH] = self_ent.health / MAX_HEALTH
    scalars[SC_FROZEN] = min(self_ent.frozen_ticks / FROZEN_TICKS_NORM, 1.0)
    scalars[SC_RESOURCES] = min(res / RESOURCE_SCALE, 1.0)
    scalars[SC_BOMBS] = min(bombs / BOMB_SCALE, 1.0)
    scalars[SC_STEP] = step / NUM_ITERS
    scalars[SC_TEAM] = (self_team or 0) / n_teams
    return planes, scalars


class CentralizedCritic(nn.Module):
    """Transformer over global-state planes + scalar token -> scalar value.

    Mirrors the SymbolicTransformerActor architecture so policy and critic are
    tuned alike. Per-cell tokens of the 9-plane global state (256 tokens) +
    one scalar token + CLS = 258 tokens. CLS -> value head. Reuses the policy's
    pre-LN `_TransformerBlock` so attention/FFN scaling stay in sync.
    """

    def __init__(self, d_model=64, n_layers=4, n_heads=4, ffn_dim=None,
                 dropout=0.1):
        super().__init__()
        # local import: critic is only imported by training code that already
        # imports policy, so the cycle is impossible — but keeping it local
        # keeps critic.py's top-level imports framework-only.
        from policy import _TransformerBlock                # noqa: PLC0415
        if ffn_dim is None:
            ffn_dim = 4 * d_model
        self.cfg = {"d_model": d_model, "n_layers": n_layers,
                    "n_heads": n_heads, "ffn_dim": ffn_dim, "dropout": dropout}
        self.plane_embed = nn.Linear(STATE_PLANES, d_model)
        self.scalar_embed = nn.Sequential(
            nn.Linear(STATE_SCALARS, d_model), nn.GELU())
        self.spatial_pos = nn.Parameter(
            torch.zeros(GRID_SIZE * GRID_SIZE, d_model))
        # two token groups: tile and scalar (CLS uses its own parameter)
        self.type_embed = nn.Parameter(torch.zeros(2, d_model))
        self.cls = nn.Parameter(torch.zeros(1, 1, d_model))
        for p in (self.spatial_pos, self.type_embed, self.cls):
            nn.init.normal_(p, std=0.02)
        self.blocks = nn.ModuleList([
            _TransformerBlock(d_model, n_heads, ffn_dim, dropout)
            for _ in range(n_layers)
        ])
        self.head_norm = nn.LayerNorm(d_model)
        self.value = _layer_init(nn.Linear(d_model, 1), std=1.0)

    def forward(self, planes, scalars):
        n = planes.shape[0]
        # [N,9,16,16] -> [N,16,16,9] -> [N,256,9]
        tiles = planes.permute(0, 2, 3, 1).reshape(
            n, GRID_SIZE * GRID_SIZE, STATE_PLANES)
        tile_tok = (self.plane_embed(tiles) + self.spatial_pos
                    + self.type_embed[0])
        scalar_tok = (self.scalar_embed(scalars).unsqueeze(1)
                      + self.type_embed[1])
        cls_tok = self.cls.expand(n, -1, -1)
        x = torch.cat([cls_tok, tile_tok, scalar_tok], dim=1)   # [N,258,d]
        for blk in self.blocks:
            x = blk(x)
        return self.value(self.head_norm(x[:, 0])).squeeze(-1)
