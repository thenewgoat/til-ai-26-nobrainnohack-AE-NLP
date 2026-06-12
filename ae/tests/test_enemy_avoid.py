"""Enemy-proximity ('agent bias') devaluation for the forage layers.

A collectible inside a VISIBLE enemy's bomb-blast footprint (LOS + Chebyshev
BLAST_RADIUS) is devalued by `enemy_avoid_factor ** (BLAST_RADIUS + 1 - cheb)`,
so closer-to-enemy tiles are penalised more and the penalty compounds across
enemies. factor == 1.0 disables it. Walls (no LOS) and frozen enemies exempt a
cell.
"""
from scripted.belief import Belief
from scripted.blast import BLAST_RADIUS

F = 0.75


class _Prior:
    def __init__(self, grid_size, wall_between):
        self.grid_size = grid_size
        self.wall_between = wall_between
        self.collectibles = {}


def _belief(grid_size=8, wall_between=None, enemies=(), frozen=()):
    b = Belief()
    b.prior = _Prior(grid_size, wall_between or {})
    b.destroyed_walls = set()
    b.enemies = set(enemies)
    b.frozen_enemies = set(frozen)
    return b


def _penalty(b, cell, factor=F):
    from scripted.layers import _enemy_threat_penalty
    return _enemy_threat_penalty(b, cell, factor)


def test_adjacent_to_enemy_is_penalised_most():
    b = _belief(enemies=[(2, 2)])
    # Chebyshev 1, clear LOS -> 0.75 ** (2 + 1 - 1) = 0.75**2
    assert _penalty(b, (2, 3)) == F ** 2


def test_two_tiles_from_enemy_penalised_less():
    b = _belief(enemies=[(2, 2)])
    # Chebyshev 2, clear LOS -> 0.75 ** (2 + 1 - 2) = 0.75**1
    assert _penalty(b, (2, 4)) == F ** 1


def test_outside_blast_radius_no_penalty():
    b = _belief(enemies=[(2, 2)])
    # Chebyshev 3 -> outside BLAST_RADIUS -> no devaluation
    assert _penalty(b, (2, 5)) == 1.0


def test_factor_one_disables():
    b = _belief(enemies=[(2, 2)])
    assert _penalty(b, (2, 3), factor=1.0) == 1.0


def test_wall_blocks_los_exempts_cell():
    # Wall between (2,3) and (2,4) blocks LOS from enemy at (2,2) to (2,4).
    b = _belief(enemies=[(2, 2)],
                wall_between={frozenset({(2, 3), (2, 4)}): False})
    assert _penalty(b, (2, 4)) == 1.0


def test_frozen_enemy_does_not_threaten():
    b = _belief(enemies=[(2, 2)], frozen=[(2, 2)])
    assert _penalty(b, (2, 3)) == 1.0


def test_two_enemies_compound():
    b = _belief(enemies=[(2, 2), (2, 4)])
    # cell (2,3) is Chebyshev 1 from BOTH enemies, clear LOS -> (0.75**2)**2
    assert _penalty(b, (2, 3)) == (F ** 2) ** 2


def test_blast_radius_is_two():
    # guards the formula's reliance on the env blast radius
    assert BLAST_RADIUS == 2


# --- integration: the penalty must actually steer sweep's target ---------- #

class _SweepPrior:
    def __init__(self):
        self.grid_size = 11
        self.wall_between = {}
        self.enemy_bases = []
        self.our_base = (0, 0)
        self.resource_cells = None
        # A is closer to the agent (wins on raw rate); B is the safe option.
        self.collectibles = {(3, 2): 1.0, (7, 4): 1.0}


def _sweep_belief():
    b = Belief()
    b.prior = _SweepPrior()
    b.destroyed_walls = set()
    b.collected = set()
    b.location = (5, 0)
    b.facing = 0
    b.enemy_bombs = {}
    b.own_bombs = []
    b.enemies = {(3, 1)}          # threatens A=(3,2) (Chebyshev 1, clear LOS)
    b.frozen_enemies = set()
    b.dead_bases = set()
    b.enemy_base_health = {}
    return b


def _sweep_action(factor):
    from scripted.danger import DangerMap
    from scripted.layers import sweep
    from scripted.pathfind import build_planner
    from scripted.strategies import StrategyParams
    b = _sweep_belief()
    danger = DangerMap({}, b)
    planner = build_planner(b, danger)
    params = StrategyParams(centre_value_weight=0.0, enemy_avoid_factor=factor)
    action = sweep(b, danger, planner, params)
    return action, planner


def test_sweep_targets_near_enemy_tile_when_bias_disabled():
    action, planner = _sweep_action(factor=1.0)
    assert action == planner.first_action((3, 2))   # closer A wins on raw rate


def test_sweep_avoids_near_enemy_tile_when_bias_enabled():
    action, planner = _sweep_action(factor=0.75)
    assert action == planner.first_action((7, 4))   # safe B wins after penalty


# --- strategy wiring: only balanced_extreme_opening changes --------------- #

def test_avoider_strategies_enable_bias_and_disable_centre():
    # Both the opening agent and the forager radiate the 0.75 ** (3 - cheb)
    # enemy-proximity penalty and drop the centre/peripheral collectible bias.
    from scripted.strategies import STRATEGIES
    for name in ("balanced_extreme_opening", "forager"):
        p = STRATEGIES[name].params
        assert p.enemy_avoid_factor == 0.75, name
        assert p.centre_value_weight == 0.0, name


def test_other_strategies_keep_defaults():
    from scripted.strategies import STRATEGIES, StrategyParams
    d = StrategyParams()
    for name in ("balanced", "collector", "adaptive"):
        p = STRATEGIES[name].params
        assert p.enemy_avoid_factor == d.enemy_avoid_factor      # 1.0 (off)
        assert p.centre_value_weight == d.centre_value_weight    # -0.4


# --- visibility penalty: shrink the forager's search space ---------------- #

def test_visibility_penalty_devalues_unseen_tiles():
    from scripted.layers import _visibility_penalty
    b = _belief()
    b.last_visible_cells = {(2, 2)}
    assert _visibility_penalty(b, (2, 2), 0.5) == 1.0     # visible -> no penalty
    assert _visibility_penalty(b, (5, 5), 0.5) == 0.5     # unseen -> devalued
    assert _visibility_penalty(b, (5, 5), 1.0) == 1.0     # factor 1.0 disables


def test_forager_halves_unseen_collectibles():
    from scripted.strategies import STRATEGIES
    assert STRATEGIES["forager"].params.unseen_value_factor == 0.5


def test_sweep_prefers_visible_collectible():
    # Two equal-value collectibles: A=(3,2) closer to the agent but out of
    # view; B=(7,4) farther but currently visible. With the penalty off, the
    # closer A wins on raw rate; halving unseen tiles flips the choice to B.
    from scripted.danger import DangerMap
    from scripted.layers import sweep
    from scripted.pathfind import build_planner
    from scripted.strategies import StrategyParams

    def _action(factor):
        b = _sweep_belief()
        b.enemies = set()                 # isolate the visibility effect
        b.last_visible_cells = {(7, 4)}   # only the far tile is in view
        danger = DangerMap({}, b)
        planner = build_planner(b, danger)
        params = StrategyParams(centre_value_weight=0.0, unseen_value_factor=factor)
        return sweep(b, danger, planner, params), planner

    action_off, planner = _action(1.0)
    assert action_off == planner.first_action((3, 2))   # closer A wins on raw rate
    action_on, planner = _action(0.5)
    assert action_on == planner.first_action((7, 4))    # visible B wins after penalty
