"""Each strategy plays a full episode against random opponents without
crashing, and exhibits its archetype. Uses the same env wiring as
test_scripted_e2e.py.
"""
from til_environment import bomberman_env
from til_environment.config import default_config

from scripted.belief import Belief
from scripted.danger import DangerMap
from scripted.decide import act
from scripted.layers import strike
from scripted.map_prior import MapPrior
from scripted.pathfind import build_planner
from scripted.strategies import STRATEGIES, StrategyParams

LEARNER = "agent_0"

# ---------------------------------------------------------------------------
# Episode helper
# ---------------------------------------------------------------------------


def _run_episode(strategy, seed=0):
    """Play one episode with `strategy` controlling agent_0. Returns the list
    of actions agent_0 took."""
    env = bomberman_env.basic_env(env_wrappers=[], cfg=default_config())
    env.reset(seed=seed)
    prior = MapPrior.load()
    belief = Belief()
    started = False
    actions = []
    for agent in env.agent_iter():
        obs, reward, term, trunc, _ = env.last()
        if agent != LEARNER:
            env.step(None if (term or trunc) else env.action_space(agent).sample())
            continue
        if term or trunc:
            env.step(None)
            continue
        if not started:
            prior.identify_team(obs["base_location"])
            belief.reset(prior)
            started = True
        belief.update(obs)
        a = act(belief, obs["action_mask"], strategy)
        actions.append(a)
        env.step(a)
    env.close()
    return actions


# ---------------------------------------------------------------------------
# Direct-layer belief builder (mirrors test_scripted_decide_layers pattern).
# ---------------------------------------------------------------------------

def _make_belief(loc, facing=0, team_bombs=3, grid_size=7,
                 enemy_bases=((3, 3),), our_base=(0, 0),
                 dead_bases=(), ally_bombs=None, enemy_bombs=None,
                 enemies=(), collectibles=None):
    """Minimal synthetic Belief for direct layer/act tests."""

    class _Prior:
        pass

    prior = _Prior()
    prior.grid_size = grid_size
    prior.wall_between = {}          # open map — no walls
    prior.collectibles = collectibles or {}
    prior.enemy_bases = list(enemy_bases)
    prior.our_base = our_base

    b = Belief()
    b.prior = prior
    b.destroyed_walls = set()
    b.collected = set()
    b.dead_bases = set(dead_bases)
    b.enemy_base_health = {}
    b.ally_bombs = ally_bombs or {}
    b.enemy_bombs = enemy_bombs or {}
    b.enemies = set(enemies)
    b.location = loc
    b.facing = facing
    b.team_bombs = team_bombs
    b.step = 10
    b.frozen_ticks = 0
    b.health = 1.0
    b.base_health = 1.0
    return b


def _all_legal_mask(n=6):
    """Action mask with every action legal."""
    return [1] * n


# ---------------------------------------------------------------------------
# Existing smoke tests
# ---------------------------------------------------------------------------


def test_every_strategy_completes_an_episode():
    for name, strategy in STRATEGIES.items():
        actions = _run_episode(strategy, seed=0)
        assert len(actions) > 0, f"{name} took no actions"
        assert all(0 <= a <= 5 for a in actions), f"{name} produced an illegal action"


def test_base_rusher_skips_collectibles_relative_to_collector():
    """base_rusher ignores collectibles; collector chases them. They should
    not produce identical action streams on the same seed."""
    rusher = _run_episode(STRATEGIES["base_rusher"], seed=1)
    collector = _run_episode(STRATEGIES["collector"], seed=1)
    assert rusher != collector


# ---------------------------------------------------------------------------
# A1 — collector never uses the strike layer
# ---------------------------------------------------------------------------


def test_collector_lacks_strike_layer_structural():
    """Structural: the collector strategy's layer tuple must not include strike."""
    assert strike not in STRATEGIES["collector"].layers, (
        "collector should not have a strike layer"
    )


def test_collector_does_not_place_bomb_at_enemy_base():
    """Behavioral: with bombs available and an enemy base in blast range,
    collector does NOT place a bomb (no strike/hunt layer to trigger it)."""
    # Agent at (3, 5), base at (3, 3) — same column, 2 tiles away, open map
    # => bomb_reaches((3,5), (3,3), belief) is True on an open grid.
    b = _make_belief(loc=(3, 5), team_bombs=3, grid_size=7,
                     enemy_bases=((3, 3),))
    danger = DangerMap({}, b)
    planner = build_planner(b, danger)
    action = act(b, _all_legal_mask(), STRATEGIES["collector"])
    assert action != 5, (
        "collector placed a bomb at an enemy base, but it has no strike layer"
    )


# ---------------------------------------------------------------------------
# A2 — extreme archetypes strike from a dangerous cell; normal rusher flees
# ---------------------------------------------------------------------------


def test_extreme_strikes_despite_danger():
    """base_rusher_extreme and balanced_extreme put strike before survive, so
    they place a bomb even when standing on a dangerous cell. The normal
    base_rusher (survive first) should NOT place a bomb in the same situation.

    Setup: agent at (3, 5), enemy base at (3, 3) (in range on open grid),
    bombs available, cell (3, 5) marked dangerous by an enemy bomb.
    """
    # Enemy bomb at (3, 5) with timer=3 makes (3,5) dangerous.
    b = _make_belief(loc=(3, 5), team_bombs=3, grid_size=7,
                     enemy_bases=((3, 3),),
                     enemy_bombs={(3, 5): 3})
    mask = _all_legal_mask()

    action_extreme_rusher = act(b, mask, STRATEGIES["base_rusher_extreme"])
    assert action_extreme_rusher == 5, (
        "base_rusher_extreme should place a bomb before fleeing danger"
    )

    action_extreme_balanced = act(b, mask, STRATEGIES["balanced_extreme"])
    assert action_extreme_balanced == 5, (
        "balanced_extreme should place a bomb before fleeing danger"
    )

    # Contrast: normal base_rusher has survive first — it should flee, not bomb.
    action_normal_rusher = act(b, mask, STRATEGIES["base_rusher"])
    assert action_normal_rusher != 5, (
        "base_rusher (survive before strike) should flee danger, not place a bomb"
    )


# ---------------------------------------------------------------------------
# A3 — balanced places bombs targeting visible enemy agents
# ---------------------------------------------------------------------------


def test_balanced_places_bomb_targeting_visible_enemy():
    """balanced strategy bombs a visible enemy agent in blast range when no
    enemy base is alive (so only the hunt layer can trigger PLACE_BOMB)."""
    # All enemy bases dead so strike layer yields; enemy agent adjacent.
    # Agent at (3, 3), enemy at (3, 4) — 1 tile away, open map => in blast range.
    b = _make_belief(loc=(3, 3), team_bombs=3, grid_size=7,
                     enemy_bases=((6, 6),),   # exists but dead
                     dead_bases=((6, 6),),
                     enemies=((3, 4),))
    danger = DangerMap({}, b)
    planner = build_planner(b, danger)

    from scripted.blast import bomb_reaches
    # Verify our setup: the agent IS in blast range of the enemy.
    assert bomb_reaches((3, 3), (3, 4), b), "test setup: enemy not in blast range"

    action = act(b, _all_legal_mask(), STRATEGIES["balanced"])
    assert action == 5, (
        "balanced should place a bomb when a visible enemy is in blast range"
    )


# ---------------------------------------------------------------------------
# A4 — camper returns home when outside its leash
# ---------------------------------------------------------------------------


def test_camper_returns_home_when_outside_leash():
    """camper strategy's camp layer returns a move-toward-home action when the
    agent is further than camp_leash (4) Chebyshev from our_base."""
    # our_base = (0, 0); place agent at (0, 6) — Chebyshev 6 > leash 4.
    b = _make_belief(loc=(0, 6), team_bombs=0, grid_size=13,
                     enemy_bases=((12, 12),),
                     our_base=(0, 0),
                     enemies=())
    action = act(b, _all_legal_mask(), STRATEGIES["camper"])
    # Must be a move (0–3), not STAY (4) or PLACE_BOMB (5).
    assert 0 <= action <= 3, (
        f"camper outside leash should move toward home, got action={action}"
    )
