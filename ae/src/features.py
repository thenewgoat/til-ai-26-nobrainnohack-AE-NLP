"""Symbolic feature builder for the AE self-play agent.

Renders the scripted/ decision modules' world model into the five-tensor
observation contract consumed by SymbolicTransformerActor: an abstraction
branch (17-channel tile grid + 5x11 per-base matrix + 10 scalars) and a raw
residual branch (the raw viewcones kept whole). Identical in training and
serving. The builder is STATEFUL — it owns a per-slot Belief; reset it via
build() seeing step == 0.

Channel/field/scalar layouts are frozen here; policy.py mirrors the dimensions
and test_feature_contract.py / test_policy_contract.py assert they agree.
"""
import json
import math
from collections import deque

import numpy as np

from scripted.belief import Belief
from scripted.blast import bomb_reaches
from scripted.danger import DangerMap
from scripted.geometry import MOVE
from scripted.layers import (BOMB_ATTACK, _base_doomed, _effective_hp,
                             _target_base)
from scripted.map_prior import MapPrior
from scripted.pathfind import build_planner
from scripted.strategies import STRATEGIES

GRID_SIZE = 16

# --- abstraction tile grid: 17 channels (frozen contract) ---
GRID_CHANNELS = 17
# Wall channels 0..3 align with scripted.geometry direction codes
# (DIR_RIGHT=0, DIR_DOWN=1, DIR_LEFT=2, DIR_UP=3): channel d holds the wall on
# the edge toward neighbor (x,y)+MOVE[d]. 0.0 open, 0.5 intact destructible,
# 1.0 indestructible or grid boundary.
CH_WALL_R = 0
CH_WALL_D = 1
CH_WALL_L = 2
CH_WALL_U = 3
CH_SELF = 4               # one-hot agent location
CH_ENEMY_LIVE = 5         # believed live (non-frozen) enemies
CH_ENEMY_FROZEN = 6       # believed frozen enemies
CH_COLLECTIBLE = 7        # remaining collectible value, /5
CH_RESPAWN = 8            # steps until a collected tile refills, /40
CH_DANGER = 9             # blast imminence 1 - ticks_to_danger/FUSE on dangerous
                          # tiles; 0 = safe OR full-fuse (overlap ch. resolves)
CH_DANGER_OVERLAP = 10    # bombs covering the tile, /4
CH_ENEMY_BOMB = 11        # enemy bomb here, timer/FUSE
CH_OWN_BOMB = 12          # own/ally bombs here, soonest timer/FUSE
CH_PLANNER_DIST = 13      # planner cost-to-go from the agent, /DIJKSTRA_NORM
CH_DIR_TARGET = 14        # next-step direction code toward the chosen target
CH_CONFIDENCE = 15        # per-tile belief staleness; 1 = seen this step
CH_BASE_MARK = 16         # +1 our base, -1 each enemy base

# --- per-base tokens: 5 bases x 11 fields (frozen contract) ---
NUM_BASES = 5
BASE_FIELDS = 11
BF_EFFECTIVE_HP = 0       # _effective_hp(belief, base) / 100
BF_OBSERVED_HP = 1        # enemy_base_health.get(base, 1.0)
BF_BOMBS_NEEDED = 2       # ceil(effective_hp / 20) / 5
BF_IS_DOOMED = 3          # _base_doomed(belief, base) -> 0/1
BF_IS_TARGET = 4          # 1 if this base is the _target_base pick
BF_IS_SOFTEN_PHASE = 5    # 1 if effective_hp > soften_floor
BF_LEASH_RADIUS = 6       # sweep leash /GRID_SIZE (0 if not Phase-B target)
BF_ARRIVAL = 7            # planner ticks to the nearest hit-tile, /DIJKSTRA_NORM
BF_IN_RANGE_NOW = 8       # 1 if a bomb at the agent's tile hits this base
BF_OWN_IN_FLIGHT = 9      # own bombs already reaching this base, /5
BF_IS_DEAD = 10           # 1 if base is in belief.dead_bases

# --- raw residual branch: viewcones kept whole (frozen contract) ---
RAW_AGENT_SHAPE = (7, 5, 25)
RAW_BASE_SHAPE = (7, 7, 25)

# --- scalar token: 10 fields (frozen contract) ---
FEATURE_SCALARS = 10
SC_STEP = 0               # step / 200
SC_TEAM_BOMBS = 1         # min(team_bombs / BOMB_SCALE, 1)
SC_RESOURCES = 2          # min(team_resources / 5, 1)
SC_HEALTH = 3             # health / 60
SC_BASE_HEALTH = 4        # our base_health / 100
SC_FROZEN = 5             # frozen_ticks / 3
SC_TEAM_ID = 6            # team / 5
SC_FACING = 7             # direction / 3
SC_OWN_BOMBS_IN_FLIGHT = 8  # min(len(own_bombs) / 5, 1)
SC_LIVE_ENEMY_BASES = 9   # live enemy base count / 5

# --- frame-stack contract (K consecutive observations stacked along the
# channel/scalar axis) ---
STACK = 5
STACKED_GRID_CHANNELS = GRID_CHANNELS * STACK     # 85
STACKED_SCALARS = FEATURE_SCALARS * STACK         # 50

BOMB_SCALE = 10.0
MAX_HEALTH = 60.0
BASE_MAX_HEALTH = 100.0
NUM_ITERS = 200.0
BOMB_FUSE = 4.0
INF = float("inf")
DIJKSTRA_NORM = 64.0      # KNOB: distance normalizer (16x16 worst-case path)

# Derive from MapPrior's already-working path so this resolves correctly in
# any layout where scripted/ is importable — dev (ae/src/) and the served
# container's flat /workspace/ — not just the dev-only "parents[1]/src/..."
# dance.
from scripted.map_prior import _DEFAULT_PATH as _MAP_PATH
_RESPAWN_PATH = _MAP_PATH.parent / "respawn_map.json"


def _scalar(x):
    return float(np.asarray(x, dtype=np.float32).reshape(-1)[0])


class FeatureBuilder:
    """Per-slot feature builder. One instance per learner-controlled slot."""

    # shared, immutable across instances
    _prior_template = None
    _respawn_map = None

    def __init__(self, teacher_strategy="balanced_extreme_opening"):
        if FeatureBuilder._prior_template is None:
            FeatureBuilder._prior_template = MapPrior.load(_MAP_PATH)
            FeatureBuilder._respawn_map = np.array(
                json.loads(_RESPAWN_PATH.read_text()), dtype=np.int32
            )
        # each builder gets its own MapPrior copy so identify_team is per-slot
        self.prior = MapPrior.load(_MAP_PATH)
        self.respawn_map = FeatureBuilder._respawn_map
        self.belief = Belief()
        self.strategy = STRATEGIES[teacher_strategy]
        self._started = False
        # per-tile respawn countdown state, maintained across steps
        self._respawn_left = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.int32)
        # per-tile last-seen step, for belief confidence
        self._last_seen = np.full((GRID_SIZE, GRID_SIZE), -1, dtype=np.int32)
        self._grid_history = deque(maxlen=STACK)
        self._scalar_history = deque(maxlen=STACK)

    # ------------------------------------------------------------------ #
    def build(self, observation):
        """Fold one observation into the belief and render the five tensors:
        (grid, base_feats, raw_agent, raw_base, scalar)."""
        step = int(_scalar(observation["step"]))
        if step == 0 or not self._started:
            self.prior.identify_team(observation["base_location"])
            self.belief.reset(self.prior)
            self._started = True
            self._respawn_left[:] = 0
            self._last_seen[:] = -1
            self._grid_history.clear()
            self._scalar_history.clear()

        prev_collected = set(self.belief.collected)
        self.belief.update(observation)
        self._tick_respawn(prev_collected, step)

        danger = DangerMap(self.belief.enemy_bombs, self.belief)
        planner = (build_planner(self.belief, danger)
                   if self.belief.location is not None else None)
        target = (_target_base(self.belief, planner, self.strategy.params)
                  if planner is not None else None)

        grid = np.zeros((GRID_CHANNELS, GRID_SIZE, GRID_SIZE), dtype=np.float32)
        self._fill_walls(grid)
        self._fill_entities(grid)
        self._fill_danger(grid, danger)
        self._fill_pathing(grid, planner, target)
        self._fill_respawn_confidence(grid, step)
        self._fill_base_mark(grid)

        base_feats = self._build_base_feats(planner, target)
        raw_agent, raw_base = self._build_raw(observation)
        scalar = self._build_scalar(observation, step)

        # Append newest snapshot to the per-slot history.
        self._grid_history.append(grid)
        self._scalar_history.append(scalar)

        # Stack newest-first; zero-pad missing past slots before the deque
        # has filled (warmup).
        pad_g = STACK - len(self._grid_history)
        pad_s = STACK - len(self._scalar_history)
        grid_parts = list(reversed(self._grid_history))
        if pad_g:
            grid_parts.append(
                np.zeros((pad_g * GRID_CHANNELS, GRID_SIZE, GRID_SIZE),
                         dtype=np.float32))
        grid_stack = np.concatenate(grid_parts, axis=0)

        scalar_parts = list(reversed(self._scalar_history))
        if pad_s:
            scalar_parts.append(
                np.zeros(pad_s * FEATURE_SCALARS, dtype=np.float32))
        scalar_stack = np.concatenate(scalar_parts, axis=0)
        return grid_stack, base_feats, raw_agent, raw_base, scalar_stack

    def _build_raw(self, observation):
        """The raw residual branch: viewcones kept whole, no channels stripped.
        A destroyed base does NOT shrink its viewcone — the env still emits a
        full [7,7,25] "ghost" base_viewcone centred on the dead base's tile
        (base_health == 0). Emit a zero [7,7,25] tensor in that case."""
        raw_agent = np.asarray(observation["agent_viewcone"], dtype=np.float32)
        raw_base = np.asarray(observation["base_viewcone"], dtype=np.float32)
        if raw_agent.shape != RAW_AGENT_SHAPE:
            raw_agent = np.zeros(RAW_AGENT_SHAPE, dtype=np.float32)
        if raw_base.shape != RAW_BASE_SHAPE or _scalar(observation["base_health"]) <= 0.0:
            raw_base = np.zeros(RAW_BASE_SHAPE, dtype=np.float32)
        return raw_agent, raw_base

    def _fill_walls(self, grid):
        """Channels 0-3: per-tile per-direction wall edge state."""
        b = self.belief
        gs = self.prior.grid_size
        for x in range(gs):
            for y in range(gs):
                for d, (dx, dy) in MOVE.items():
                    nx, ny = x + dx, y + dy
                    if not (0 <= nx < gs and 0 <= ny < gs):
                        grid[d, x, y] = 1.0
                    elif b.is_wall((x, y), (nx, ny)):
                        grid[d, x, y] = (
                            0.5 if b.is_destructible((x, y), (nx, ny)) else 1.0
                        )

    def _fill_entities(self, grid):
        """Channels SELF / ENEMY_LIVE / ENEMY_FROZEN / COLLECTIBLE."""
        b = self.belief
        gs = self.prior.grid_size
        if b.location is not None:
            grid[CH_SELF, b.location[0], b.location[1]] = 1.0
        for (ex, ey) in b.live_enemies():
            if 0 <= ex < gs and 0 <= ey < gs:
                grid[CH_ENEMY_LIVE, ex, ey] = 1.0
        for (ex, ey) in b.frozen_enemies:
            if 0 <= ex < gs and 0 <= ey < gs:
                grid[CH_ENEMY_FROZEN, ex, ey] = 1.0
        for (cx, cy), value in b.remaining_collectibles().items():
            if 0 <= cx < gs and 0 <= cy < gs:
                grid[CH_COLLECTIBLE, cx, cy] = min(value / 5.0, 1.0)

    def _fill_danger(self, grid, danger):
        """Channels DANGER / DANGER_OVERLAP / ENEMY_BOMB / OWN_BOMB."""
        b = self.belief
        gs = self.prior.grid_size
        for x in range(gs):
            for y in range(gs):
                if danger.is_dangerous((x, y)):
                    t = danger.ticks_to_danger((x, y))
                    grid[CH_DANGER, x, y] = max(
                        0.0, 1.0 - min(t, BOMB_FUSE) / BOMB_FUSE)
                    grid[CH_DANGER_OVERLAP, x, y] = min(
                        danger.overlap((x, y)) / 4.0, 1.0)
        for (bx, by), timer in b.enemy_bombs.items():
            if 0 <= bx < gs and 0 <= by < gs:
                grid[CH_ENEMY_BOMB, bx, by] = min(timer, BOMB_FUSE) / BOMB_FUSE
        own = dict(b.ally_bombs)
        for cell, timer in b.own_bombs:
            own[cell] = min(own.get(cell, 999), timer)
        for (bx, by), timer in own.items():
            if 0 <= bx < gs and 0 <= by < gs:
                grid[CH_OWN_BOMB, bx, by] = min(timer, BOMB_FUSE) / BOMB_FUSE

    def _fill_pathing(self, grid, planner, target):
        """Channels PLANNER_DIST / DIR_TARGET."""
        b = self.belief
        gs = self.prior.grid_size
        if planner is None or b.location is None:
            return
        for x in range(gs):
            for y in range(gs):
                d = planner.dist_to((x, y))
                grid[CH_PLANNER_DIST, x, y] = (
                    min(d / DIJKSTRA_NORM, 1.0) if d != INF else 1.0)
        if target is not None:
            base = target[0]
            best, best_tile = INF, None
            for x in range(gs):
                for y in range(gs):
                    if bomb_reaches((x, y), base, b):
                        d = planner.dist_to((x, y))
                        if d < best:
                            best, best_tile = d, (x, y)
            if best_tile is not None and best != INF:
                action = planner.first_action(best_tile)
                grid[CH_DIR_TARGET, b.location[0], b.location[1]] = \
                    self._action_to_dir_code(action, b.facing)

    def _fill_respawn_confidence(self, grid, step):
        """Channels RESPAWN / CONFIDENCE."""
        grid[CH_RESPAWN] = np.minimum(
            self._respawn_left.astype(np.float32) / 40.0, 1.0)
        if self.belief.location is not None:
            self._last_seen[self.belief.location] = step
        # all visible enemies, including frozen — we've seen the tile regardless
        for (ex, ey) in self.belief.enemies:
            if 0 <= ex < GRID_SIZE and 0 <= ey < GRID_SIZE:
                self._last_seen[ex, ey] = step
        age = step - self._last_seen
        conf = np.where(self._last_seen < 0, 0.0,
                        np.maximum(0.0, 1.0 - age / 40.0))
        grid[CH_CONFIDENCE] = conf.astype(np.float32)

    def _fill_base_mark(self, grid):
        """Channel BASE_MARK: +1 our base, -1 each enemy base.

        Base coords come from the loaded map prior — always in bounds.
        """
        if self.prior.our_base is not None:
            ox, oy = self.prior.our_base
            grid[CH_BASE_MARK, ox, oy] = 1.0
        for (bx, by) in self.belief.live_enemy_bases():
            grid[CH_BASE_MARK, bx, by] = -1.0

    def _build_base_feats(self, planner, target):
        """The 5x11 per-base abstraction matrix — the channels the BC clone was
        blind to. Base order is prior.enemy_bases (fixed in the novice map)."""
        b = self.belief
        gs = self.prior.grid_size
        params = self.strategy.params
        target_base = target[0] if target is not None else None
        feats = np.zeros((NUM_BASES, BASE_FIELDS), dtype=np.float32)
        for i, base in enumerate(self.prior.enemy_bases):
            eff = _effective_hp(b, base)                       # raw HP 0..100
            observed = b.enemy_base_health.get(base, 1.0)      # ratio 0..1
            bombs_needed = math.ceil(eff / BOMB_ATTACK)
            is_soften = eff > params.soften_floor
            is_target = (base == target_base)
            arrival = INF
            if planner is not None:
                for x in range(gs):
                    for y in range(gs):
                        if bomb_reaches((x, y), base, b):
                            d = planner.dist_to((x, y))
                            if d < arrival:
                                arrival = d
            arrival_n = (min(arrival / DIJKSTRA_NORM, 1.0)
                         if arrival != INF else 1.0)
            in_range = (b.location is not None
                        and bomb_reaches(b.location, base, b))
            own_in_flight = sum(1 for cell, _ in b.own_bombs
                                if bomb_reaches(cell, base, b))
            leash = 0.0
            if is_target and not is_soften:
                leash = (bombs_needed + 1) / GRID_SIZE
            feats[i, BF_EFFECTIVE_HP] = eff / BASE_MAX_HEALTH
            feats[i, BF_OBSERVED_HP] = observed
            feats[i, BF_BOMBS_NEEDED] = bombs_needed / 5.0
            feats[i, BF_IS_DOOMED] = float(_base_doomed(b, base))
            feats[i, BF_IS_TARGET] = float(is_target)
            feats[i, BF_IS_SOFTEN_PHASE] = float(is_soften)
            feats[i, BF_LEASH_RADIUS] = leash
            feats[i, BF_ARRIVAL] = arrival_n
            feats[i, BF_IN_RANGE_NOW] = float(in_range)
            feats[i, BF_OWN_IN_FLIGHT] = min(own_in_flight / 5.0, 1.0)
            feats[i, BF_IS_DEAD] = float(base in b.dead_bases)
        return feats

    def _build_scalar(self, observation, step):
        """The 10-field scalar token (frozen contract)."""
        b = self.belief
        s = np.zeros(FEATURE_SCALARS, dtype=np.float32)
        s[SC_STEP] = step / NUM_ITERS
        s[SC_TEAM_BOMBS] = min(
            _scalar(observation["team_bombs"]) / BOMB_SCALE, 1.0)
        s[SC_RESOURCES] = min(
            _scalar(observation["team_resources"]) / 5.0, 1.0)
        s[SC_HEALTH] = _scalar(observation["health"]) / MAX_HEALTH
        s[SC_BASE_HEALTH] = (
            _scalar(observation["base_health"]) / BASE_MAX_HEALTH)
        s[SC_FROZEN] = min(_scalar(observation["frozen_ticks"]) / 3.0, 1.0)
        team = self.prior.team if self.prior.team is not None else 0
        s[SC_TEAM_ID] = team / 5.0
        s[SC_FACING] = int(_scalar(observation["direction"])) / 3.0
        s[SC_OWN_BOMBS_IN_FLIGHT] = min(len(b.own_bombs) / 5.0, 1.0)
        s[SC_LIVE_ENEMY_BASES] = len(b.live_enemy_bases()) / NUM_BASES
        return s

    @staticmethod
    def _action_to_dir_code(action, facing):
        """Map a planner FORWARD/BACKWARD/LEFT/RIGHT action to a world
        direction code 1..4 (RIGHT/DOWN/LEFT/UP +1); 0 if no movement."""
        from scripted.geometry import FORWARD, BACKWARD, LEFT, RIGHT
        if action == FORWARD:
            return float(facing) + 1.0
        if action == BACKWARD:
            return float((facing + 2) % 4) + 1.0
        if action == LEFT:
            return float((facing + 3) % 4) + 1.0
        if action == RIGHT:
            return float((facing + 1) % 4) + 1.0
        return 0.0

    def _tick_respawn(self, prev_collected, step):
        """Maintain the per-tile respawn countdown.

        When belief.collected gains a tile this step, arm its countdown to that
        tile's respawn delay. Every step, decrement all live countdowns. The
        env re-queues the collectible after respawn_map[x,y] steps.
        """
        # decrement all live countdowns
        np.subtract(self._respawn_left, 1, out=self._respawn_left,
                    where=self._respawn_left > 0)
        # arm countdowns for newly collected tiles
        newly = self.belief.collected - prev_collected
        for (cx, cy) in newly:
            if 0 <= cx < GRID_SIZE and 0 <= cy < GRID_SIZE:
                self._respawn_left[cx, cy] = int(self.respawn_map[cx, cy])
        # a tile no longer in `collected` (re-seen present) has refilled
        refilled = prev_collected - self.belief.collected
        for (cx, cy) in refilled:
            if 0 <= cx < GRID_SIZE and 0 <= cy < GRID_SIZE:
                self._respawn_left[cx, cy] = 0

