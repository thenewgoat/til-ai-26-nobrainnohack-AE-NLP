"""Tests for the adaptive cascade layers (ae/src/scripted/adaptive_layers.py)."""
from dataclasses import replace

from scripted import adaptive_layers
from scripted.belief import Belief
from scripted.danger import DangerMap
from scripted.geometry import FORWARD
from scripted.pathfind import build_planner
from scripted.strategies import StrategyParams
from scripted.adaptive_layers import forage_loop, defend_intercept


def test_loops_loaded_from_shipped_json():
    assert isinstance(adaptive_layers.LOOPS, list)
    assert adaptive_layers.LOOPS, "expected at least one shipped loop"
    for loop in adaptive_layers.LOOPS:
        assert len(loop["waypoints"]) >= 2
        assert loop["period"] > 0
        assert "yield_attack" in loop and "yield_endgame" in loop


def test_teams_cover_all_six_teams():
    assert set(adaptive_layers.TEAMS) == {"0", "1", "2", "3", "4", "5"}


class _Prior:
    """Minimal hand-built prior: open grid, no walls, no collectibles."""

    def __init__(self, grid_size=5, enemy_bases=(), our_base=None):
        self.grid_size = grid_size
        self.wall_between = {}
        self.collectibles = {}
        self.enemy_bases = list(enemy_bases)
        self.our_base = our_base


def _ctx(loc, facing, grid_size=5, enemy_bases=()):
    """Build (belief, danger, planner, params) on an open grid."""
    b = Belief()
    b.prior = _Prior(grid_size, enemy_bases)
    b.location = loc
    b.facing = facing
    b.step = 0
    danger = DangerMap({}, b)
    planner = build_planner(b, danger)
    return b, danger, planner, StrategyParams()


_SQUARE = {"waypoints": [[2, 0], [2, 2], [0, 2], [0, 0]], "period": 12,
           "yield_attack": 1.0, "yield_endgame": 1.0, "resource_leaning": False}


def test_forage_loop_walks_toward_the_first_waypoint(monkeypatch):
    monkeypatch.setattr(adaptive_layers, "LOOPS", [dict(_SQUARE)])
    b, danger, planner, params = _ctx((0, 0), 0)        # facing RIGHT
    action = forage_loop(b, danger, planner, params)
    assert action == FORWARD                            # (0,0) -> waypoint (2,0)
    assert b.adaptive_state["forage_active_loop"] == 0
    assert b.adaptive_state["forage_waypoint_index"] == 0


def test_forage_loop_advances_waypoint_on_arrival(monkeypatch):
    monkeypatch.setattr(adaptive_layers, "LOOPS", [dict(_SQUARE)])
    b, danger, planner, params = _ctx((2, 0), 1)        # AT waypoint 0, facing DOWN
    action = forage_loop(b, danger, planner, params)
    assert b.adaptive_state["forage_waypoint_index"] == 1   # advanced to (2,2)
    assert action == FORWARD                                # facing DOWN -> (2,2)


def test_forage_loop_yields_when_no_loops(monkeypatch):
    monkeypatch.setattr(adaptive_layers, "LOOPS", [])
    b, danger, planner, params = _ctx((0, 0), 0)
    assert forage_loop(b, danger, planner, params) is None


def test_forage_loop_skips_an_unreachable_waypoint(monkeypatch):
    # Waypoint 0 (1,0) is walled off from the agent at (0,0); the layer must
    # skip it and route to the next reachable waypoint (0,2).
    loop = {"waypoints": [[1, 0], [0, 2], [0, 0]], "period": 8,
            "yield_attack": 1.0, "yield_endgame": 1.0, "resource_leaning": False}
    monkeypatch.setattr(adaptive_layers, "LOOPS", [dict(loop)])
    b, danger, planner, params = _ctx((0, 0), 0)
    # Box (1,0) off completely so it is unreachable.
    b.prior.wall_between = {
        frozenset({(1, 0), (0, 0)}): False,
        frozenset({(1, 0), (2, 0)}): False,
        frozenset({(1, 0), (1, 1)}): False,
    }
    planner = build_planner(b, danger)
    action = forage_loop(b, danger, planner, params)
    assert action is not None                       # routed somewhere
    assert b.adaptive_state["forage_waypoint_index"] == 1   # skipped (1,0), went to (0,2)


_GOOD = {"waypoints": [[2, 0], [2, 2]], "period": 10,
         "yield_attack": 5.0, "yield_endgame": 5.0, "resource_leaning": False}
_POOR = {"waypoints": [[0, 1], [1, 1]], "period": 10,
         "yield_attack": 1.0, "yield_endgame": 1.0, "resource_leaning": False}


def test_forage_loop_switches_off_a_poor_loop_after_commit_window(monkeypatch):
    monkeypatch.setattr(adaptive_layers, "LOOPS", [dict(_POOR), dict(_GOOD)])
    b, danger, planner, params = _ctx((0, 0), 0)
    b.step = 100                                          # past the commit window
    b.adaptive_state.update({"forage_active_loop": 0, "forage_waypoint_index": 0,
                             "forage_switch_step": 0, "forage_phase": False})
    forage_loop(b, danger, planner, params)
    # realised yield is 0 (nothing collected) << 0.6 * 1.0; loop 1 is better.
    assert b.adaptive_state["forage_active_loop"] == 1


def test_forage_loop_holds_a_loop_inside_the_commit_window(monkeypatch):
    monkeypatch.setattr(adaptive_layers, "LOOPS", [dict(_POOR), dict(_GOOD)])
    b, danger, planner, params = _ctx((0, 0), 0)
    b.step = 5                                            # inside the commit window
    b.adaptive_state.update({"forage_active_loop": 0, "forage_waypoint_index": 0,
                             "forage_switch_step": 0, "forage_phase": False})
    forage_loop(b, danger, planner, params)
    assert b.adaptive_state["forage_active_loop"] == 0    # held, not switched


import math
from scripted.adaptive_layers import _hit_tiles, _forage_rate, _base_roi


def _attack_ctx(loc, facing, base, hp_ratio, team_bombs, grid_size=9):
    """A (belief, danger, planner, params) context with one enemy base."""
    b = Belief()
    b.prior = _Prior(grid_size, enemy_bases=(base,))
    b.location = loc
    b.facing = facing
    b.step = 0
    b.team_bombs = team_bombs
    b.enemy_base_health = {base: hp_ratio}
    danger = DangerMap({}, b)
    planner = build_planner(b, danger)
    return b, danger, planner, StrategyParams()


def test_hit_tiles_are_within_blast_range(monkeypatch):
    b, danger, planner, params = _attack_ctx((0, 4), 0, (4, 4), 1.0, 5)
    tiles = _hit_tiles(b, (4, 4))
    # Open grid: every tile within Chebyshev 2 of (4,4) can bomb it — a 5x5 block.
    assert len(tiles) == 25
    assert (4, 4) in tiles and (2, 4) in tiles and (1, 4) not in tiles


def test_base_roi_full_hp_base(monkeypatch):
    monkeypatch.setattr(adaptive_layers, "LOOPS", [])
    b, danger, planner, params = _attack_ctx((0, 4), 0, (4, 4), 1.0, 5)
    roi, bombs_needed, eff_hp, tile = _base_roi(b, planner, (4, 4), params)
    assert bombs_needed == 5                        # 100 HP / 20 per bomb
    assert eff_hp == 100.0
    assert tile is not None and roi > 0.0


def test_base_roi_boosts_a_softened_base(monkeypatch):
    monkeypatch.setattr(adaptive_layers, "LOOPS", [])
    # Two equidistant bases: one full HP, one at 20 HP. The softened one must
    # score a strictly higher ROI (vulture boost).
    full = _attack_ctx((0, 4), 0, (4, 4), 1.0, 5)
    soft = _attack_ctx((0, 4), 0, (4, 4), 0.2, 5)
    roi_full, *_ = _base_roi(full[0], full[2], (4, 4), full[3])
    roi_soft, *_ = _base_roi(soft[0], soft[2], (4, 4), soft[3])
    assert roi_soft > roi_full


def test_forage_rate_uses_best_loop_estimate(monkeypatch):
    monkeypatch.setattr(adaptive_layers, "LOOPS",
                        [{"waypoints": [[0, 0], [1, 0]], "period": 10,
                          "yield_attack": 3.5, "yield_endgame": 1.0,
                          "resource_leaning": False}])
    b, danger, planner, params = _attack_ctx((0, 4), 0, (4, 4), 1.0, 5)
    # realised_yield is 0 (nothing collected) -> the loop estimate dominates.
    assert _forage_rate(b, params) == 3.5


def test_base_roi_unreachable_base(monkeypatch):
    monkeypatch.setattr(adaptive_layers, "LOOPS", [])
    # Box the base region off from the agent so no hit-tile is reachable.
    b, danger, planner, params = _attack_ctx((0, 0), 0, (8, 8), 1.0, 5)
    b.prior.wall_between = {
        frozenset({(8, 8), (7, 8)}): False, frozenset({(8, 8), (8, 7)}): False,
        frozenset({(7, 8), (6, 8)}): False, frozenset({(8, 7), (8, 6)}): False,
        frozenset({(7, 8), (7, 7)}): False, frozenset({(8, 7), (7, 7)}): False,
        frozenset({(6, 8), (6, 7)}): False, frozenset({(8, 6), (7, 6)}): False,
        frozenset({(6, 8), (5, 8)}): False, frozenset({(8, 6), (8, 5)}): False,
        frozenset({(7, 7), (6, 7)}): False, frozenset({(7, 7), (7, 6)}): False,
        frozenset({(6, 7), (6, 6)}): False, frozenset({(7, 6), (6, 6)}): False,
        frozenset({(6, 7), (5, 7)}): False, frozenset({(7, 6), (7, 5)}): False,
    }
    planner = build_planner(b, danger)
    roi, bombs_needed, eff_hp, tile = _base_roi(b, planner, (8, 8), params)
    assert tile is None        # every hit-tile of (8,8) is walled off
    assert roi == 0.0


from scripted.adaptive_layers import _attack
from scripted.geometry import PLACE_BOMB as _PLACE_BOMB


def test_attack_bombs_when_in_range(monkeypatch):
    # Agent on a tile within Chebyshev 2 of the base, holding a bomb.
    b, danger, planner, params = _attack_ctx((2, 4), 0, (4, 4), 1.0, 3)
    assert _attack(b, danger, planner, (4, 4)) == _PLACE_BOMB


def test_attack_routes_toward_base_when_out_of_range(monkeypatch):
    # Agent far from the base, out of blast range -> a move action (0-3).
    b, danger, planner, params = _attack_ctx((0, 0), 0, (8, 8), 1.0, 3)
    action = _attack(b, danger, planner, (8, 8))
    assert action is not None and 0 <= action <= 3


def test_attack_does_not_bomb_with_no_bombs(monkeypatch):
    # In blast range but team_bombs == 0: cannot bomb and cannot breach;
    # already on a hit-tile so there is nothing to route to -> None.
    b, danger, planner, params = _attack_ctx((2, 4), 0, (4, 4), 1.0, 0)
    assert _attack(b, danger, planner, (4, 4)) is None


from scripted.adaptive_layers import rush_roi


def test_rush_roi_yields_when_no_enemy_bases(monkeypatch):
    monkeypatch.setattr(adaptive_layers, "LOOPS", [])
    b = Belief()
    b.prior = _Prior(9, enemy_bases=())
    b.location, b.facing, b.step, b.team_bombs = (0, 0), 0, 0, 3
    danger = DangerMap({}, b)
    planner = build_planner(b, danger)
    assert rush_roi(b, danger, planner, StrategyParams()) is None


def test_rush_roi_attacks_a_reachable_base(monkeypatch):
    monkeypatch.setattr(adaptive_layers, "LOOPS", [])      # forage rate 0
    b, danger, planner, params = _attack_ctx((2, 4), 0, (4, 4), 1.0, 3)
    action = rush_roi(b, danger, planner, params)
    assert action == _PLACE_BOMB                            # in range -> bomb
    assert b.adaptive_state["rush_target"] == (4, 4)


def test_rush_roi_gate_yields_to_a_richer_forage(monkeypatch):
    # A far, full-HP base scores a low ROI; a high-yield loop out-earns it.
    monkeypatch.setattr(adaptive_layers, "LOOPS",
                        [{"waypoints": [[0, 0], [1, 0]], "period": 10,
                          "yield_attack": 100.0, "yield_endgame": 100.0,
                          "resource_leaning": False}])
    b, danger, planner, params = _attack_ctx((0, 0), 0, (8, 8), 1.0, 1)
    assert rush_roi(b, danger, planner, params) is None     # gated -> forage
    # ...but the target is kept committed for hysteresis on re-entry.
    assert b.adaptive_state["rush_target"] == (8, 8)


def test_rush_roi_prefers_the_softened_base(monkeypatch):
    monkeypatch.setattr(adaptive_layers, "LOOPS", [])
    # Two equidistant bases; (4,4) is full HP, (4,0) is at 20 HP -> vulture it.
    b = Belief()
    b.prior = _Prior(9, enemy_bases=((4, 4), (4, 0)))
    b.location, b.facing, b.step, b.team_bombs = (0, 2), 0, 0, 3
    b.enemy_base_health = {(4, 4): 1.0, (4, 0): 0.2}
    danger = DangerMap({}, b)
    planner = build_planner(b, danger)
    rush_roi(b, danger, planner, StrategyParams())
    assert b.adaptive_state["rush_target"] == (4, 0)


def test_rush_roi_sticks_to_its_committed_target(monkeypatch):
    monkeypatch.setattr(adaptive_layers, "LOOPS", [])
    # (4,0) is marginally better (95 HP vs 100) but within roi_gate_margin,
    # so the committed target (4,4) is held — a genuine hysteresis test.
    b = Belief()
    b.prior = _Prior(9, enemy_bases=((4, 4), (4, 0)))
    b.location, b.facing, b.step, b.team_bombs = (0, 2), 0, 0, 3
    b.enemy_base_health = {(4, 4): 1.0, (4, 0): 0.95}
    b.adaptive_state["rush_target"] = (4, 4)                # already committed
    danger = DangerMap({}, b)
    planner = build_planner(b, danger)
    rush_roi(b, danger, planner, StrategyParams())
    assert b.adaptive_state["rush_target"] == (4, 4)        # not flipped to (4,0)


from scripted.adaptive_layers import _projected_hp


def test_projected_hp_is_full_with_no_bombs(monkeypatch):
    b, danger, planner, params = _attack_ctx((0, 0), 0, (4, 4), 1.0, 3)
    assert _projected_hp(b, (4, 4), at_tick=10) == 100.0


def test_projected_hp_subtracts_an_enemy_bomb_by_its_detonation_tick(monkeypatch):
    b, danger, planner, params = _attack_ctx((0, 0), 0, (4, 4), 1.0, 3)
    b.enemy_bombs = {(4, 5): 2}                  # adjacent to base, detonates in 2
    assert _projected_hp(b, (4, 4), at_tick=1) == 100.0   # not yet detonated
    assert _projected_hp(b, (4, 4), at_tick=2) == 80.0    # 100 - 20


def test_projected_hp_counts_our_own_in_flight_bombs(monkeypatch):
    b, danger, planner, params = _attack_ctx((0, 0), 0, (4, 4), 1.0, 3)
    b.own_bombs = [((4, 5), 3)]                  # own_bombs is a list of (cell, timer)
    assert _projected_hp(b, (4, 4), at_tick=3) == 80.0


def test_projected_hp_floors_at_zero(monkeypatch):
    b, danger, planner, params = _attack_ctx((0, 0), 0, (4, 4), 0.2, 3)  # 20 HP
    b.enemy_bombs = {(4, 5): 1, (4, 3): 1}       # two bombs, 40 damage incoming
    assert _projected_hp(b, (4, 4), at_tick=5) == 0.0


def test_projected_hp_ignores_a_bomb_out_of_blast_range(monkeypatch):
    b, danger, planner, params = _attack_ctx((0, 0), 0, (4, 4), 1.0, 3)
    b.enemy_bombs = {(0, 0): 1}                  # Chebyshev 4 from the base — no reach
    assert _projected_hp(b, (4, 4), at_tick=5) == 100.0


def test_base_roi_skips_a_base_the_bomb_horizon_kills(monkeypatch):
    monkeypatch.setattr(adaptive_layers, "LOOPS", [])
    # Base at 20 HP with two enemy bombs incoming (40 damage) — it will be dead
    # before the agent (far away) can arrive, so ROI must be 0.
    b, danger, planner, params = _attack_ctx((0, 0), 0, (8, 8), 0.2, 3)
    b.enemy_bombs = {(8, 7): 1, (7, 8): 1}
    roi, bombs_needed, eff_hp, tile = _base_roi(b, planner, (8, 8), params)
    assert roi == 0.0
    assert eff_hp == 0.0


def test_base_roi_rates_a_softening_base_higher(monkeypatch):
    monkeypatch.setattr(adaptive_layers, "LOOPS", [])
    # Same geometry, same observed HP; one base has an enemy bomb softening it.
    plain = _attack_ctx((0, 4), 0, (4, 4), 1.0, 3)
    soft = _attack_ctx((0, 4), 0, (4, 4), 1.0, 3)
    soft[0].enemy_bombs = {(4, 5): 2}            # 20 damage will land before arrival
    roi_plain, *_ = _base_roi(plain[0], plain[2], (4, 4), plain[3])
    roi_soft, *_ = _base_roi(soft[0], soft[2], (4, 4), soft[3])
    assert roi_soft > roi_plain


def test_base_roi_unaffected_when_no_bombs_in_flight(monkeypatch):
    # With no bombs, projected HP == observed HP — full-HP base still 5 bombs.
    monkeypatch.setattr(adaptive_layers, "LOOPS", [])
    b, danger, planner, params = _attack_ctx((0, 4), 0, (4, 4), 1.0, 5)
    roi, bombs_needed, eff_hp, tile = _base_roi(b, planner, (4, 4), params)
    assert bombs_needed == 5
    assert eff_hp == 100.0
    assert tile is not None and roi > 0.0


def _defend_ctx(loc, facing, base, enemy, base_health, team_bombs, grid_size=9):
    """A (belief, danger, planner, params) context with our base under threat."""
    b = Belief()
    b.prior = _Prior(grid_size, our_base=base)
    b.location, b.facing, b.step = loc, facing, 0
    b.base_health = float(base_health)
    b.team_bombs = team_bombs
    b.enemies = {enemy}
    danger = DangerMap({}, b)
    planner = build_planner(b, danger)
    return b, danger, planner, StrategyParams()


def test_defend_intercept_bombs_an_attacker_in_range(monkeypatch):
    monkeypatch.setattr(adaptive_layers, "LOOPS", [])
    # Agent at (4,3) can bomb the attacker at (4,5); base at (4,4), full HP.
    b, danger, planner, params = _defend_ctx((4, 3), 0, (4, 4), (4, 5), 100.0, 3)
    assert defend_intercept(b, danger, planner, params) == _PLACE_BOMB


def test_defend_intercept_routes_toward_a_reachable_attacker(monkeypatch):
    monkeypatch.setattr(adaptive_layers, "LOOPS", [])
    # Agent far from the attacker but the interception is still in time.
    b, danger, planner, params = _defend_ctx((0, 4), 0, (4, 4), (4, 5), 100.0, 3)
    action = defend_intercept(b, danger, planner, params)
    assert action is not None and 0 <= action <= 3        # a move toward intercept


def test_defend_intercept_yields_when_no_enemy_near_base(monkeypatch):
    monkeypatch.setattr(adaptive_layers, "LOOPS", [])
    # Base at (0,0); the only enemy is at (8,8) — Chebyshev 8, far outside
    # defend_radius (4), so it is not a threat.
    b, danger, planner, params = _defend_ctx((4, 4), 0, (0, 0), (8, 8), 100.0, 3)
    assert defend_intercept(b, danger, planner, params) is None


def test_defend_intercept_yields_when_it_cannot_arrive_in_time(monkeypatch):
    monkeypatch.setattr(adaptive_layers, "LOOPS", [])
    # Base at 20 HP -> attacker time-to-kill is short; agent is far away.
    b, danger, planner, params = _defend_ctx((0, 0), 0, (8, 8), (8, 7), 20.0, 3)
    assert defend_intercept(b, danger, planner, params) is None


def test_defend_intercept_yields_with_no_bombs(monkeypatch):
    monkeypatch.setattr(adaptive_layers, "LOOPS", [])
    b, danger, planner, params = _defend_ctx((4, 3), 0, (4, 4), (4, 5), 100.0, 0)
    assert defend_intercept(b, danger, planner, params) is None


def test_defend_intercept_yields_when_foraging_out_earns_it(monkeypatch):
    # A very high-yield loop makes the forage opportunity cost beat the defense.
    monkeypatch.setattr(adaptive_layers, "LOOPS",
                        [{"waypoints": [[0, 0], [1, 0]], "period": 10,
                          "yield_attack": 1000.0, "yield_endgame": 1000.0,
                          "resource_leaning": False}])
    b, danger, planner, params = _defend_ctx((0, 4), 0, (4, 4), (4, 5), 100.0, 3)
    assert defend_intercept(b, danger, planner, params) is None


from scripted.adaptive_layers import trap


def _trap_ctx(loc, facing, enemy, team_bombs, grid_size=5, enemy_bombs=None,
              ally_bombs=None):
    """A (belief, danger, planner, params) context with one visible enemy."""
    b = Belief()
    b.prior = _Prior(grid_size)
    b.location, b.facing, b.step = loc, facing, 0
    b.team_bombs = team_bombs
    b.enemies = {enemy}
    if enemy_bombs:
        b.enemy_bombs = dict(enemy_bombs)
    if ally_bombs:
        b.ally_bombs = dict(ally_bombs)
    danger = DangerMap({}, b)
    planner = build_planner(b, danger)
    return b, danger, planner, StrategyParams()


def test_trap_fires_when_pair_is_in_killboxes(monkeypatch):
    monkeypatch.setattr("scripted.adaptive_layers.KILLBOXES",
                        frozenset({((1, 3), (1, 1))}))
    b, danger, planner, params = _trap_ctx((1, 3), 0, (1, 1), 1)
    assert trap(b, danger, planner, params) == _PLACE_BOMB


def test_trap_yields_when_pair_not_in_killboxes(monkeypatch):
    monkeypatch.setattr("scripted.adaptive_layers.KILLBOXES", frozenset())
    b, danger, planner, params = _trap_ctx((1, 3), 0, (1, 1), 1)
    assert trap(b, danger, planner, params) is None


def test_trap_yields_when_disabled(monkeypatch):
    monkeypatch.setattr("scripted.adaptive_layers.KILLBOXES",
                        frozenset({((1, 3), (1, 1))}))
    b, danger, planner, params = _trap_ctx((1, 3), 0, (1, 1), 1)
    params = replace(params, trap_enabled=False)
    assert trap(b, danger, planner, params) is None


def test_trap_yields_with_no_bombs(monkeypatch):
    monkeypatch.setattr("scripted.adaptive_layers.KILLBOXES",
                        frozenset({((1, 3), (1, 1))}))
    b, danger, planner, params = _trap_ctx((1, 3), 0, (1, 1), 0)
    assert trap(b, danger, planner, params) is None


def test_trap_yields_when_enemy_bomb_observed(monkeypatch):
    # Any enemy bomb on the board could open a destructible wall and break
    # the offline-computed killbox; the layer self-disables for safety.
    monkeypatch.setattr("scripted.adaptive_layers.KILLBOXES",
                        frozenset({((1, 3), (1, 1))}))
    b, danger, planner, params = _trap_ctx(
        (1, 3), 0, (1, 1), 1, enemy_bombs={(0, 0): 3})
    assert trap(b, danger, planner, params) is None


def test_trap_yields_when_ally_bomb_observed(monkeypatch):
    monkeypatch.setattr("scripted.adaptive_layers.KILLBOXES",
                        frozenset({((1, 3), (1, 1))}))
    b, danger, planner, params = _trap_ctx(
        (1, 3), 0, (1, 1), 1, ally_bombs={(2, 2): 2})
    assert trap(b, danger, planner, params) is None
