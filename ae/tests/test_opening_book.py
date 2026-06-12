"""Tests for the adaptive waypoint opening book (scripted.gates.scripted_opening)."""
from scripted import gates
from scripted.belief import Belief
from scripted.danger import DangerMap
from scripted.gates import BOMB, scripted_opening
from scripted.geometry import FORWARD, PLACE_BOMB, STAY
from scripted.pathfind import build_planner
from scripted.strategies import StrategyParams


class _Prior:
    """Minimal hand-built prior: open grid, no walls, no collectibles."""

    def __init__(self, grid_size=5, team=1):
        self.grid_size = grid_size
        self.wall_between = {}
        self.collectibles = {}
        self.enemy_bases = []
        self.our_base = None
        self.team = team


def _ctx(loc=(0, 0), facing=0, team=1, team_bombs=0, enemy_bombs=None):
    """Build (belief, danger, planner, params) on an open grid."""
    b = Belief()
    b.prior = _Prior(team=team)
    b.location = loc
    b.facing = facing
    b.step = 0
    b.team_bombs = team_bombs
    danger = DangerMap(enemy_bombs or {}, b)
    planner = build_planner(b, danger)
    return b, danger, planner, StrategyParams()


def _gate(b, danger, planner, params, action=STAY):
    return scripted_opening(b, danger, planner, params, action)


def test_no_book_for_slot_yields(monkeypatch):
    monkeypatch.setattr(gates, "_OPENING_BOOKS", {1: [(2, 0)]})
    b, danger, planner, params = _ctx(team=3)
    assert _gate(b, danger, planner, params) is None
    assert "opening_idx" not in b.adaptive_state        # untouched


def test_routes_toward_first_waypoint(monkeypatch):
    monkeypatch.setattr(gates, "_OPENING_BOOKS", {1: [(2, 0)]})
    b, danger, planner, params = _ctx((0, 0), 0)        # facing RIGHT
    assert _gate(b, danger, planner, params) == FORWARD
    assert b.adaptive_state["opening_idx"] == 0


def test_arrival_advances_to_next_waypoint(monkeypatch):
    monkeypatch.setattr(gates, "_OPENING_BOOKS", {1: [(2, 0), (4, 0)]})
    b, danger, planner, params = _ctx((2, 0), 0)        # AT waypoint 0
    assert _gate(b, danger, planner, params) == FORWARD  # -> (4, 0)
    assert b.adaptive_state["opening_idx"] == 1


def test_replans_around_a_wall(monkeypatch):
    # Direct edge (0,0)->(1,0) is walled; the book still reaches (1,0) by
    # detour because the waypoint is re-routed through the planner.
    monkeypatch.setattr(gates, "_OPENING_BOOKS", {1: [(1, 0)]})
    b, danger, planner, params = _ctx((0, 0), 0)
    b.prior.wall_between = {frozenset({(0, 0), (1, 0)}): False}
    planner = build_planner(b, danger)
    assert _gate(b, danger, planner, params) is not None
    assert b.adaptive_state.get("opening_aborted") is not True


def test_book_complete_latches(monkeypatch):
    monkeypatch.setattr(gates, "_OPENING_BOOKS", {1: [(2, 0)]})
    b, danger, planner, params = _ctx((2, 0), 0)        # already at the last waypoint
    assert _gate(b, danger, planner, params) is None
    assert b.adaptive_state["opening_aborted"] is True
    assert _gate(b, danger, planner, params) is None    # stays handed over


def test_bomb_entry_places_then_advances(monkeypatch):
    monkeypatch.setattr(gates, "_OPENING_BOOKS", {1: [(2, 0), BOMB, (4, 0)]})
    b, danger, planner, params = _ctx((2, 0), 0, team_bombs=3)
    assert _gate(b, danger, planner, params) == PLACE_BOMB
    # Next tick: the bomb is down on our tile -> advance past BOMB to (4,0).
    b.record_own_bomb()
    assert _gate(b, danger, planner, params) == FORWARD
    assert b.adaptive_state["opening_idx"] == 2


def test_bomb_entry_without_bombs_aborts(monkeypatch):
    monkeypatch.setattr(gates, "_OPENING_BOOKS", {1: [(2, 0), BOMB]})
    b, danger, planner, params = _ctx((2, 0), 0, team_bombs=0)
    assert _gate(b, danger, planner, params) is None
    assert b.adaptive_state["opening_aborted"] is True


def test_danger_aborts_and_latches(monkeypatch):
    monkeypatch.setattr(gates, "_OPENING_BOOKS", {1: [(2, 0)]})
    b, danger, planner, params = _ctx((0, 0), 0)
    hot = DangerMap({(0, 0): 3}, b)                     # enemy bomb on our tile
    assert _gate(b, hot, planner, params) is None       # survive owns the tick
    assert b.adaptive_state["opening_aborted"] is True
    assert _gate(b, danger, planner, params) is None    # safe again, still latched


def test_body_block_aborts(monkeypatch):
    monkeypatch.setattr(gates, "_OPENING_BOOKS", {1: [(2, 0)]})
    b, danger, planner, params = _ctx((0, 0), 0)
    b.stuck_ticks = params.stuck_trigger_ticks
    assert _gate(b, danger, planner, params) is None
    assert b.adaptive_state["opening_aborted"] is True


def test_unreachable_waypoint_aborts(monkeypatch):
    monkeypatch.setattr(gates, "_OPENING_BOOKS", {1: [(1, 0)]})
    b, danger, planner, params = _ctx((0, 0), 0)
    b.prior.wall_between = {                            # (1,0) boxed off entirely
        frozenset({(1, 0), (0, 0)}): False,
        frozenset({(1, 0), (2, 0)}): False,
        frozenset({(1, 0), (1, 1)}): False,
    }
    planner = build_planner(b, danger)
    assert _gate(b, danger, planner, params) is None
    assert b.adaptive_state["opening_aborted"] is True


def test_timeout_aborts(monkeypatch):
    monkeypatch.setattr(gates, "_OPENING_BOOKS", {1: [(2, 0)]})
    b, danger, planner, params = _ctx((0, 0), 0)
    b.step = gates.OPENING_MAX_TICKS + 1
    assert _gate(b, danger, planner, params) is None
    assert b.adaptive_state["opening_aborted"] is True
