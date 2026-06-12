import numpy as np
from til_environment import bomberman_env
from til_environment.config import default_config

from scripted.belief import Belief
from scripted.map_prior import MapPrior


def _novice_obs():
    """Reset the novice env and return agent_0's first observation dict."""
    cfg = default_config()
    cfg.env.novice = True
    env = bomberman_env.basic_env(env_wrappers=[], cfg=cfg)
    env.reset(seed=0)
    obs = env.observe("agent_0")
    env.close()
    return obs


def test_reset_initialises_from_prior():
    m = MapPrior.load()
    b = Belief()
    b.reset(m)
    assert b.collected == set()
    assert b.enemy_bombs == {}
    assert b.ally_bombs == {}
    assert b.destroyed_walls == set()
    assert b.prior is m


def test_update_records_agent_state():
    m = MapPrior.load()
    b = Belief()
    b.reset(m)
    obs = _novice_obs()
    b.update(obs)
    assert b.location == (14, 9)        # agent_0 novice spawn
    assert b.facing == 0
    assert b.team_bombs == 3
    assert b.step == 0


def test_update_marks_visible_collectibles():
    """A known collectible cell, seen empty, is recorded as collected."""
    m = MapPrior.load()
    b = Belief()
    b.reset(m)
    cc = next(iter(m.collectibles))                 # a known collectible cell
    vc = np.zeros((7, 5, 25), dtype=np.float32)
    vc[2, 2, 5] = 1.0                               # CH TILE_EMPTY at agent's own cell
    obs = {"agent_viewcone": vc, "direction": 0, "location": list(cc),
           "base_viewcone": np.zeros((7, 7, 25), dtype=np.float32),
           "base_location": [0, 0],
           "team_bombs": 3, "team_resources": 0.0, "step": 5, "frozen_ticks": 0,
           "health": [60.0], "base_health": [100.0]}
    b.update(obs)
    assert cc in b.collected
    assert cc not in b.remaining_collectibles()


def test_walls_are_monotonic():
    """A wall in the prior is never re-added once dropped."""
    m = MapPrior.load()
    b = Belief()
    b.reset(m)
    pair = next(p for p, d in m.wall_between.items() if d)  # a destructible wall
    a, c = tuple(pair)
    b.destroyed_walls.add(pair)
    assert b.is_wall(a, c) is False     # destroyed stays destroyed


def test_update_records_destroyed_wall():
    """update() records a prior wall the viewcone shows as gone, and it stays
    destroyed on a later update that shows the wall present again (monotonic)."""
    m = MapPrior.load()
    # An in-grid wall pair (both tiles within 0..15).
    pair = next(p for p in m.wall_between
                if all(0 <= c < 16 for t in p for c in t))
    a, c = tuple(pair)
    delta = (c[0] - a[0], c[1] - a[1])
    b = Belief()
    b.reset(m)
    # Synthetic obs: agent at `a` facing RIGHT(0); its own cell (i=2,j=2) is
    # visible (TILE_EMPTY) with ALL wall channels 0 -> prior walls at `a` gone.
    vc = np.zeros((7, 5, 25), dtype=np.float32)
    vc[2, 2, 5] = 1.0                               # CH TILE_EMPTY
    obs = {"agent_viewcone": vc, "direction": 0, "location": list(a),
           "base_viewcone": np.zeros((7, 7, 25), dtype=np.float32),
           "base_location": [0, 0],
           "team_bombs": 3, "team_resources": 0.0, "step": 5, "frozen_ticks": 0,
           "health": [60.0], "base_health": [100.0]}
    b.update(obs)
    assert frozenset({a, c}) in b.destroyed_walls
    assert b.is_wall(a, c) is False
    # Monotonic: a later observation showing the wall present must NOT revive it.
    wall_ch = {(1, 0): 1, (0, 1): 2, (-1, 0): 3, (0, -1): 4}[delta]
    vc2 = np.zeros((7, 5, 25), dtype=np.float32)
    vc2[2, 2, 5] = 1.0
    vc2[2, 2, wall_ch] = 1.0
    obs2 = dict(obs)
    obs2["agent_viewcone"] = vc2
    b.update(obs2)
    assert b.is_wall(a, c) is False                 # still destroyed


def test_ally_and_enemy_bombs_tracked_separately():
    """Enemy bombs feed the danger model (enemy_bombs); our own ally bomb is
    tracked in ally_bombs but kept OUT of enemy_bombs — it cannot hurt us."""
    m = MapPrior.load()
    b = Belief()
    b.reset(m)
    loc = (5, 5)
    vc = np.zeros((7, 5, 25), dtype=np.float32)
    # Agent's own cell (i=2,j=2 -> world `loc`): visible, an ALLY bomb here.
    vc[2, 2, 5] = 1.0       # TILE_EMPTY -> visible
    vc[2, 2, 17] = 1.0      # ALLY_BOMB
    vc[2, 2, 19] = 3.0      # ALLY_BOMB_TIMER
    # One tile ahead (i=3,j=2 -> world (6,5)): visible, an ENEMY bomb here.
    vc[3, 2, 5] = 1.0       # TILE_EMPTY -> visible
    vc[3, 2, 18] = 1.0      # ENEMY_BOMB
    vc[3, 2, 20] = 2.0      # ENEMY_BOMB_TIMER
    obs = {"agent_viewcone": vc, "direction": 0, "location": list(loc),
           "base_viewcone": np.zeros((7, 7, 25), dtype=np.float32),
           "base_location": [0, 0],
           "team_bombs": 3, "team_resources": 0.0, "step": 5, "frozen_ticks": 0,
           "health": [60.0], "base_health": [100.0]}
    b.update(obs)
    # Belief stores `env_timer + 1` (lethal phase from the planner's POV —
    # detonation happens in the step AFTER our move). Viewcone timers 2 and 3
    # become stored 3 and 4 respectively.
    assert b.enemy_bombs.get((6, 5)) == 3     # enemy bomb -> danger model
    assert (5, 5) not in b.enemy_bombs        # our own bomb -> NOT in danger model
    assert b.ally_bombs.get((5, 5)) == 4      # our own bomb -> tracked separately


def test_ally_bomb_detonation_opens_walls():
    """When an ally bomb's timer runs out it has detonated — the destructible
    walls its blast reaches are recorded as destroyed (LOS predictor)."""
    m = MapPrior.load()
    b = Belief()
    b.reset(m)
    # A destructible wall fully in-grid; put an ally bomb on one of its tiles.
    pair = next(p for p, d in m.wall_between.items()
                if d and all(0 <= c < 16 for t in p for c in t))
    bomb_cell = tuple(next(iter(pair)))
    b.ally_bombs = {bomb_cell: 1}             # timer 1 -> detonates this update
    vc = np.zeros((7, 5, 25), dtype=np.float32)   # empty viewcone
    obs = {"agent_viewcone": vc, "direction": 0, "location": list(bomb_cell),
           "base_viewcone": np.zeros((7, 7, 25), dtype=np.float32),
           "base_location": [0, 0],
           "team_bombs": 3, "team_resources": 0.0, "step": 5, "frozen_ticks": 0,
           "health": [60.0], "base_health": [100.0]}
    b.update(obs)
    assert bomb_cell not in b.ally_bombs      # bomb consumed
    assert pair in b.destroyed_walls          # wall on the bomb's cell opened


def _base_obs(loc, step, enemy_base_present):
    """Synthetic observation: agent on `loc`, that cell visible, with the
    ENEMY_BASE channel set or not."""
    vc = np.zeros((7, 5, 25), dtype=np.float32)
    vc[2, 2, 5] = 1.0                          # TILE_EMPTY -> cell visible
    if enemy_base_present:
        vc[2, 2, 12] = 1.0                     # ENEMY_BASE channel
    return {"agent_viewcone": vc, "direction": 0, "location": list(loc),
            "base_viewcone": np.zeros((7, 7, 25), dtype=np.float32),
            "base_location": [0, 0],
            "team_bombs": 3, "team_resources": 0.0, "step": step, "frozen_ticks": 0,
            "health": [60.0], "base_health": [100.0]}


def test_enemy_base_seen_destroyed_stays_dead():
    """A known enemy-base tile seen WITHOUT the ENEMY_BASE channel is recorded
    as destroyed — and a destroyed base never respawns, so it stays dead."""
    m = MapPrior.load()
    m.identify_team((13, 9))                   # we are team 0
    b = Belief()
    b.reset(m)
    enemy_base = m.enemy_bases[0]
    b.update(_base_obs(enemy_base, step=50, enemy_base_present=False))
    assert b.base_alive(enemy_base) is False
    b.step = 199
    assert b.base_alive(enemy_base) is False    # bases never respawn — still dead


def test_enemy_base_health_is_tracked():
    """A visible live base records its health ratio from ENEMY_BASE_HEALTH."""
    m = MapPrior.load()
    m.identify_team((13, 9))
    b = Belief()
    b.reset(m)
    enemy_base = m.enemy_bases[0]
    obs = _base_obs(enemy_base, step=20, enemy_base_present=True)
    obs["agent_viewcone"][2, 2, 24] = 0.6      # ENEMY_BASE_HEALTH ratio
    b.update(obs)
    assert b.base_alive(enemy_base) is True
    assert abs(b.enemy_base_health[enemy_base] - 0.6) < 1e-5   # float32 round-trip


def _enemy_obs(loc, enemy_health):
    """Synthetic obs: agent on `loc` facing RIGHT(0); an enemy agent one tile
    ahead (view cell i=3,j=2 -> world (loc[0]+1, loc[1])) at health ratio
    `enemy_health` (0.0 == frozen)."""
    vc = np.zeros((7, 5, 25), dtype=np.float32)
    vc[2, 2, 5] = 1.0                       # own cell visible (TILE_EMPTY)
    vc[3, 2, 5] = 1.0                       # tile ahead visible
    vc[3, 2, 10] = 1.0                      # ENEMY_AGENT channel
    vc[3, 2, 22] = enemy_health             # ENEMY_AGENT_HEALTH ratio
    return {"agent_viewcone": vc, "direction": 0, "location": list(loc),
            "base_viewcone": np.zeros((7, 7, 25), dtype=np.float32),
            "base_location": [0, 0],
            "team_bombs": 3, "team_resources": 0.0, "step": 5, "frozen_ticks": 0,
            "health": [60.0], "base_health": [100.0]}


def test_update_records_live_enemy_not_frozen():
    m = MapPrior.load()
    b = Belief()
    b.reset(m)
    b.update(_enemy_obs((5, 5), enemy_health=1.0))
    assert (6, 5) in b.enemies
    assert (6, 5) not in b.frozen_enemies


def test_update_records_frozen_enemy():
    m = MapPrior.load()
    b = Belief()
    b.reset(m)
    b.update(_enemy_obs((5, 5), enemy_health=0.0))
    assert (6, 5) in b.enemies
    assert (6, 5) in b.frozen_enemies


def test_live_enemies_excludes_frozen():
    m = MapPrior.load()
    b = Belief()
    b.reset(m)
    b.update(_enemy_obs((5, 5), enemy_health=0.0))
    assert b.live_enemies() == set()
    b.update(_enemy_obs((5, 5), enemy_health=1.0))
    assert b.live_enemies() == {(6, 5)}


def test_frozen_enemies_cleared_each_update():
    m = MapPrior.load()
    b = Belief()
    b.reset(m)
    b.update(_enemy_obs((5, 5), enemy_health=0.0))
    assert b.frozen_enemies == {(6, 5)}
    # An update whose viewcone shows no enemy on that tile clears it.
    gone = _enemy_obs((5, 5), enemy_health=0.0)
    gone["agent_viewcone"][3, 2, 10] = 0.0          # remove ENEMY_AGENT
    gone["agent_viewcone"][3, 2, 22] = 0.0          # remove ENEMY_AGENT_HEALTH
    b.update(gone)
    assert b.frozen_enemies == set()


def test_reset_clears_frozen_enemies():
    m = MapPrior.load()
    b = Belief()
    b.reset(m)
    b.frozen_enemies.add((1, 1))
    b.reset(m)
    assert b.frozen_enemies == set()


# --- base viewcone folding -------------------------------------------------

def _dual_obs(agent_loc, base_loc, base_vc, base_health=100.0):
    """Observation with an empty agent viewcone and a caller-supplied base
    viewcone, so the base viewcone is the only source of folded entities."""
    return {"agent_viewcone": np.zeros((7, 5, 25), dtype=np.float32),
            "base_viewcone": base_vc,
            "direction": 0, "location": list(agent_loc),
            "base_location": list(base_loc),
            "team_bombs": 3, "team_resources": 0.0, "step": 5, "frozen_ticks": 0,
            "health": [60.0], "base_health": [base_health]}


def test_base_viewcone_records_enemy_near_home():
    """An enemy seen only in base_viewcone lands in belief.enemies."""
    m = MapPrior.load()
    b = Belief()
    b.reset(m)
    base_vc = np.zeros((7, 7, 25), dtype=np.float32)
    # cell (4,3) with base at (8,8), R=3 -> world (8+4-3, 8+3-3) = (9, 8)
    base_vc[4, 3, 5] = 1.0       # TILE_EMPTY -> visible
    base_vc[4, 3, 10] = 1.0      # ENEMY_AGENT
    base_vc[4, 3, 22] = 1.0      # ENEMY_AGENT_HEALTH -> alive
    b.update(_dual_obs(agent_loc=(0, 0), base_loc=(8, 8), base_vc=base_vc))
    assert (9, 8) in b.enemies
    assert (9, 8) not in b.frozen_enemies


def test_base_viewcone_records_enemy_bomb_near_home():
    """An enemy bomb seen only in base_viewcone lands in belief.enemy_bombs."""
    m = MapPrior.load()
    b = Belief()
    b.reset(m)
    base_vc = np.zeros((7, 7, 25), dtype=np.float32)
    base_vc[4, 3, 5] = 1.0       # TILE_EMPTY -> visible; world (9, 8)
    base_vc[4, 3, 18] = 1.0      # ENEMY_BOMB
    base_vc[4, 3, 20] = 2.0      # ENEMY_BOMB_TIMER
    b.update(_dual_obs((0, 0), (8, 8), base_vc))
    # Stored timer = viewcone_timer + 1 (lethal-phase shift).
    assert b.enemy_bombs.get((9, 8)) == 3


def test_base_viewcone_centre_cell_maps_to_the_base_tile():
    """The centre cell (R,R) of base_viewcone maps to the base tile itself."""
    m = MapPrior.load()
    b = Belief()
    b.reset(m)
    base_vc = np.zeros((7, 7, 25), dtype=np.float32)
    base_vc[3, 3, 5] = 1.0       # centre cell, TILE_EMPTY -> visible
    base_vc[3, 3, 10] = 1.0      # ENEMY_AGENT
    base_vc[3, 3, 22] = 1.0      # alive
    b.update(_dual_obs((0, 0), (6, 7), base_vc))
    assert (6, 7) in b.enemies   # centre maps to base_location


def test_base_and_agent_viewcones_fold_idempotently():
    """A tile seen by both viewcones yields one consistent belief entry."""
    m = MapPrior.load()
    b = Belief()
    b.reset(m)
    # Agent at (8,8) facing RIGHT; base at (8,8). World tile (9,8) is in BOTH
    # cones: agent cell (3,2) and base cell (4,3).
    agent_vc = np.zeros((7, 5, 25), dtype=np.float32)
    agent_vc[3, 2, 5] = 1.0
    agent_vc[3, 2, 10] = 1.0
    agent_vc[3, 2, 22] = 1.0
    base_vc = np.zeros((7, 7, 25), dtype=np.float32)
    base_vc[4, 3, 5] = 1.0
    base_vc[4, 3, 10] = 1.0
    base_vc[4, 3, 22] = 1.0
    obs = {"agent_viewcone": agent_vc, "base_viewcone": base_vc,
           "direction": 0, "location": [8, 8], "base_location": [8, 8],
           "team_bombs": 3, "team_resources": 0.0, "step": 5, "frozen_ticks": 0,
           "health": [60.0], "base_health": [100.0]}
    b.update(obs)
    assert b.enemies == {(9, 8)}     # exactly one entry, no duplication


def test_destroyed_base_skips_base_viewcone_fold():
    """base_health == 0 means the base is destroyed — its viewcone must be
    ignored even if it still carries entities."""
    m = MapPrior.load()
    b = Belief()
    b.reset(m)
    base_vc = np.zeros((7, 7, 25), dtype=np.float32)
    base_vc[4, 3, 5] = 1.0
    base_vc[4, 3, 10] = 1.0
    base_vc[4, 3, 22] = 1.0
    b.update(_dual_obs((0, 0), (8, 8), base_vc, base_health=0.0))
    assert b.enemies == set()        # base viewcone not folded


def test_destroyed_base_degenerate_viewcone_does_not_crash():
    """A destroyed base yields the env's degenerate (1,1,25) zero viewcone and
    a [0,0] base_location — update() must tolerate it."""
    m = MapPrior.load()
    b = Belief()
    b.reset(m)
    dead_vc = np.zeros((1, 1, 25), dtype=np.float32)
    b.update(_dual_obs((5, 5), (0, 0), dead_vc, base_health=0.0))
    assert b.enemies == set()


# --- own-bomb tracking -----------------------------------------------------

def test_record_own_bomb_appends_at_location():
    from scripted.pathfind import BOMB_TIMER
    m = MapPrior.load()
    b = Belief()
    b.reset(m)
    b.location = (5, 5)
    b.record_own_bomb()
    # Initial timer = BOMB_TIMER + 1: the bomb is placed at action 1 of the
    # current obs (planner phase 1), so it detonates at phase 1 + BOMB_TIMER.
    # The next belief.update decrements once, landing on BOMB_TIMER — matching
    # the viewcone-sourced bomb's lethal-phase semantics.
    assert b.own_bombs == [((5, 5), BOMB_TIMER + 1)]
    b.record_own_bomb()                      # a stack: two entries on one tile
    assert b.own_bombs == [((5, 5), BOMB_TIMER + 1), ((5, 5), BOMB_TIMER + 1)]


def test_own_bombs_decrement_and_expire_on_update():
    m = MapPrior.load()
    b = Belief()
    b.reset(m)
    b.own_bombs = [((3, 3), 2), ((4, 4), 1)]
    obs = {"agent_viewcone": np.zeros((7, 5, 25), dtype=np.float32),
           "base_viewcone": np.zeros((7, 7, 25), dtype=np.float32),
           "base_location": [0, 0], "direction": 0, "location": [3, 3],
           "team_bombs": 3, "team_resources": 0.0, "step": 5, "frozen_ticks": 0,
           "health": [60.0], "base_health": [100.0]}
    b.update(obs)
    assert b.own_bombs == [((3, 3), 1)]      # (4,4) timer 1 -> 0 -> dropped


def test_reset_clears_own_bombs():
    m = MapPrior.load()
    b = Belief()
    b.reset(m)
    b.location = (1, 1)
    b.record_own_bomb()
    b.reset(m)
    assert b.own_bombs == []


def test_adaptive_state_is_empty_dict_after_reset():
    m = MapPrior.load()
    b = Belief()
    assert b.adaptive_state == {}            # set in __init__
    b.adaptive_state["x"] = 1
    b.reset(m)
    assert b.adaptive_state == {}            # cleared on reset


def test_realised_yield_credits_a_tile_the_agent_collected():
    m = MapPrior.load()
    b = Belief()
    b.reset(m)
    cc = next(iter(m.collectibles))                 # a known collectible cell
    vc = np.zeros((7, 5, 25), dtype=np.float32)
    vc[2, 2, 5] = 1.0                               # CH TILE_EMPTY at agent's own cell
    obs = {"agent_viewcone": vc, "direction": 0, "location": list(cc),
           "base_viewcone": np.zeros((7, 7, 25), dtype=np.float32),
           "base_location": [0, 0],
           "team_bombs": 3, "team_resources": 0.0, "step": 5, "frozen_ticks": 0,
           "health": [60.0], "base_health": [100.0]}
    b.update(obs)
    assert cc in b.collected
    assert b.realised_yield(window=10) == m.collectibles[cc] / 10


def test_realised_yield_is_zero_with_no_collections():
    m = MapPrior.load()
    b = Belief()
    b.reset(m)
    assert b.realised_yield(window=10) == 0.0


def test_realised_yield_prunes_entries_outside_the_window():
    b = Belief()
    b._yield_window.append((0, 5.0))         # old — should be pruned
    b._yield_window.append((50, 5.0))        # recent — should be kept
    b.step = 55
    # window [45, 55] keeps only the step-50 entry: 5.0 / 10.
    assert b.realised_yield(window=10) == 5.0 / 10


