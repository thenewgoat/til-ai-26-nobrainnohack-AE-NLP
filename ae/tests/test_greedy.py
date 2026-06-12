"""Tests for the greedy scripted strategy.

Tiny hand-rolled MapPrior + Belief instances — no til_environment, no
self-play. Each test builds exactly the world it needs.
"""
from scripted.belief import Belief
from scripted.map_prior import MapPrior


def make_prior(grid_size=5, walls=None, enemy_bases=None, our_base=(0, 0)):
    """Build a MapPrior with walls and bases hand-set; bypasses arena_map.json.

    `walls` is an iterable of (a, b) pairs of adjacent cells separated by a
    wall — pass `True` as the third element to mark a wall destructible.
    `enemy_bases` defaults to a single base at the grid corner. The team
    fields (`our_base`, `enemy_bases`) are set directly; `identify_team` is
    not used because we control the team here.
    """
    wall_between = {}
    for entry in (walls or []):
        a, b, *rest = entry
        wall_between[frozenset({tuple(a), tuple(b)})] = bool(rest[0]) if rest else False
    bases_map = {0: tuple(our_base)}
    for i, b in enumerate(enemy_bases or [(grid_size - 1, grid_size - 1)], start=1):
        bases_map[i] = tuple(b)
    prior = MapPrior(grid_size=grid_size, wall_between=wall_between,
                     bases=bases_map, spawns={}, collectibles={})
    prior.team = 0
    prior.our_base = tuple(our_base)
    prior.enemy_bases = [b for t, b in bases_map.items() if t != 0]
    return prior


def make_belief(prior, *, location=(0, 0), facing=0, team_bombs=0,
                frozen_ticks=0, dead_bases=(), destroyed_walls=()):
    """Build a Belief with prior set and the listed fields populated. Does not
    call `belief.update` — the test controls every field directly."""
    belief = Belief()
    belief.prior = prior
    belief._enemy_base_set = set(prior.enemy_bases)
    belief.location = tuple(location)
    belief.facing = int(facing)
    belief.team_bombs = int(team_bombs)
    belief.frozen_ticks = int(frozen_ticks)
    belief.dead_bases = set(dead_bases)
    belief.destroyed_walls = set(frozenset({tuple(a), tuple(b)})
                                  for a, b in destroyed_walls)
    return belief


def test_make_belief_smoke():
    """Smoke test the helpers — they don't crash and `is_wall` works."""
    prior = make_prior(grid_size=3, walls=[((0, 0), (1, 0))])
    belief = make_belief(prior)
    assert belief.is_wall((0, 0), (1, 0)) is True
    assert belief.is_wall((0, 0), (0, 1)) is False


def test_astar_finds_shortest_path_open_grid():
    from scripted.greedy import _astar, _astar_cost
    prior = make_prior(grid_size=5)
    belief = make_belief(prior)
    path = _astar(belief, (0, 0), (4, 4))
    assert path[0] == (0, 0)
    assert path[-1] == (4, 4)
    # 4-connected unit-cost: optimal length is Manhattan + 1 (incl. start cell).
    assert len(path) == 9
    assert _astar_cost(belief, (0, 0), (4, 4)) == 8
    # Each consecutive pair is 4-adjacent.
    for a, b in zip(path, path[1:]):
        assert abs(a[0] - b[0]) + abs(a[1] - b[1]) == 1


def test_astar_routes_around_wall():
    from scripted.greedy import _astar_cost
    # Horizontal wall row separating y=1 from y=2, except the rightmost edge.
    walls = [((x, 1), (x, 2)) for x in range(4)]
    prior = make_prior(grid_size=5, walls=walls)
    belief = make_belief(prior)
    # Open detour through column 4: cost 8 (open) -> still 8 (the gap is at x=4).
    # Build a clearer case: wall everywhere except column 4 — direct path is
    # blocked, so the route goes via (4, 1) -> (4, 2).
    assert _astar_cost(belief, (0, 0), (0, 4)) > 4  # forced detour
    # Without the wall, open-grid cost would be 4.


def test_astar_returns_empty_when_unreachable():
    from scripted.greedy import _astar, _astar_cost
    # Wall off (0, 0) on every side.
    walls = [((0, 0), (1, 0)), ((0, 0), (0, 1))]
    prior = make_prior(grid_size=3, walls=walls)
    belief = make_belief(prior)
    assert _astar(belief, (0, 0), (2, 2)) == []
    assert _astar_cost(belief, (0, 0), (2, 2)) == float("inf")


def test_astar_reroutes_after_wall_destroyed():
    from scripted.greedy import _astar_cost
    walls = [((x, 1), (x, 2)) for x in range(4)]  # leave x=4 open
    prior = make_prior(grid_size=5, walls=walls)
    belief = make_belief(prior)
    blocked_cost = _astar_cost(belief, (0, 0), (0, 4))
    # Open one cell in the wall row at x=0 — now the direct path works.
    belief.destroyed_walls.add(frozenset({(0, 1), (0, 2)}))
    open_cost = _astar_cost(belief, (0, 0), (0, 4))
    assert open_cost < blocked_cost
    assert open_cost == 4  # direct path


def test_first_legal_prefers_preference_order():
    from scripted.greedy import _first_legal
    from scripted.geometry import BACKWARD, FORWARD, LEFT, RIGHT, STAY
    # mask: FORWARD=0, BACKWARD=1, LEFT=1, RIGHT=0, STAY=1, PLACE_BOMB=0
    mask = [0, 1, 1, 0, 1, 0]
    assert _first_legal(mask, [FORWARD, BACKWARD]) == BACKWARD
    assert _first_legal(mask, [LEFT, RIGHT, BACKWARD]) == LEFT


def test_first_legal_falls_back_to_any_legal():
    from scripted.greedy import _first_legal
    mask = [0, 1, 0, 0, 0, 0]  # only BACKWARD legal
    # Preference contains no legal action — fallback finds BACKWARD.
    assert _first_legal(mask, [0, 2, 3]) == 1


def test_first_legal_last_resort_is_stay():
    from scripted.greedy import _first_legal
    from scripted.geometry import STAY
    mask = [0, 0, 0, 0, 0, 0]
    assert _first_legal(mask, [0, 1]) == STAY


def test_cell_to_action_forward_when_facing_step():
    from scripted.greedy import _cell_to_action
    from scripted.geometry import FORWARD, DIR_RIGHT
    mask = [1, 1, 1, 1, 1, 0]
    # Agent at (0, 0) facing RIGHT, step into (1, 0) — FORWARD aligns.
    assert _cell_to_action((1, 0), (0, 0), DIR_RIGHT, mask) == FORWARD


def test_cell_to_action_backward_when_facing_away():
    from scripted.greedy import _cell_to_action
    from scripted.geometry import BACKWARD, DIR_RIGHT
    mask = [1, 1, 1, 1, 1, 0]
    # Agent at (1, 0) facing RIGHT, step into (0, 0) — BACKWARD saves a turn.
    assert _cell_to_action((0, 0), (1, 0), DIR_RIGHT, mask) == BACKWARD


def test_cell_to_action_turns_to_face_step():
    from scripted.greedy import _cell_to_action
    from scripted.geometry import LEFT, RIGHT, DIR_RIGHT
    mask = [1, 1, 1, 1, 1, 0]
    # Agent at (0, 0) facing RIGHT, step into (0, 1) (DIR_DOWN) — turn RIGHT.
    assert _cell_to_action((0, 1), (0, 0), DIR_RIGHT, mask) == RIGHT
    # Agent at (0, 1) facing RIGHT, step into (0, 0) (DIR_UP) — turn LEFT.
    assert _cell_to_action((0, 0), (0, 1), DIR_RIGHT, mask) == LEFT


def test_cell_to_action_falls_back_when_preferred_masked():
    from scripted.greedy import _cell_to_action
    from scripted.geometry import DIR_RIGHT
    # Want FORWARD but mask blocks it — fall through to first_legal.
    mask = [0, 1, 1, 1, 1, 0]
    # Returns any legal action (not STAY), here BACKWARD (1) is first legal.
    a = _cell_to_action((1, 0), (0, 0), DIR_RIGHT, mask)
    assert mask[a] == 1


def test_act_returns_stay_when_frozen():
    from scripted.greedy import act
    from scripted.geometry import STAY
    prior = make_prior(grid_size=5)
    belief = make_belief(prior, frozen_ticks=3)
    mask = [0, 0, 0, 0, 1, 0]  # only STAY legal — matches the env's frozen behaviour
    assert act(belief, mask) == STAY


def test_act_falls_back_when_no_live_bases():
    from scripted.greedy import act
    prior = make_prior(grid_size=5, enemy_bases=[(4, 4)])
    belief = make_belief(prior, dead_bases=[(4, 4)])
    mask = [1, 0, 0, 0, 1, 0]  # FORWARD and STAY legal
    a = act(belief, mask)
    assert mask[a] == 1
    assert a in (0, 4)  # FORWARD preferred over STAY


def test_act_places_bomb_when_adjacent_to_live_base():
    from scripted.greedy import act
    from scripted.geometry import PLACE_BOMB
    prior = make_prior(grid_size=5, enemy_bases=[(2, 2)])
    belief = make_belief(prior, location=(2, 1), team_bombs=1)
    mask = [1, 1, 1, 1, 1, 1]  # PLACE_BOMB legal
    assert act(belief, mask) == PLACE_BOMB


def test_act_does_not_bomb_with_zero_bombs():
    from scripted.greedy import act
    from scripted.geometry import PLACE_BOMB
    prior = make_prior(grid_size=5, enemy_bases=[(2, 2)])
    belief = make_belief(prior, location=(2, 1), team_bombs=0)
    mask = [1, 1, 1, 1, 1, 0]
    assert act(belief, mask) != PLACE_BOMB


def test_act_does_not_bomb_outside_range():
    from scripted.greedy import act
    from scripted.geometry import PLACE_BOMB
    prior = make_prior(grid_size=10, enemy_bases=[(9, 9)])
    belief = make_belief(prior, location=(0, 0), team_bombs=1)
    mask = [1, 1, 1, 1, 1, 1]
    assert act(belief, mask) != PLACE_BOMB


def test_act_targets_only_live_base():
    from scripted.greedy import act
    from scripted.geometry import PLACE_BOMB
    # Two bases: agent is adjacent to the dead one and far from the live one.
    # The dead base must not be a target, so no PLACE_BOMB.
    prior = make_prior(grid_size=10, enemy_bases=[(1, 1), (9, 9)])
    belief = make_belief(prior, location=(1, 0), team_bombs=1,
                         dead_bases=[(1, 1)])
    mask = [1, 1, 1, 1, 1, 1]
    assert act(belief, mask) != PLACE_BOMB


def test_act_moves_toward_target_when_out_of_range():
    from scripted.greedy import act
    from scripted.geometry import FORWARD, DIR_RIGHT
    prior = make_prior(grid_size=10, enemy_bases=[(9, 0)])
    # Agent at (0, 0) facing RIGHT — A* step is into (1, 0), aligned with facing.
    belief = make_belief(prior, location=(0, 0), facing=DIR_RIGHT, team_bombs=1)
    mask = [1, 1, 1, 1, 1, 1]
    assert act(belief, mask) == FORWARD


def test_act_turns_to_face_target_step():
    from scripted.greedy import act
    from scripted.geometry import RIGHT, DIR_RIGHT
    prior = make_prior(grid_size=10, enemy_bases=[(0, 9)])
    # Agent at (0, 0) facing RIGHT — A* step is into (0, 1) (DOWN), so turn RIGHT.
    belief = make_belief(prior, location=(0, 0), facing=DIR_RIGHT)
    mask = [1, 1, 1, 1, 1, 0]
    assert act(belief, mask) == RIGHT


def test_act_falls_back_when_target_unreachable():
    from scripted.greedy import act
    # Agent fully walled off from the only enemy base.
    walls = [((0, 0), (1, 0)), ((0, 0), (0, 1))]
    prior = make_prior(grid_size=5, walls=walls, enemy_bases=[(4, 4)])
    belief = make_belief(prior, location=(0, 0))
    mask = [0, 0, 0, 0, 1, 0]  # only STAY legal — unreachable + no legal move
    a = act(belief, mask)
    assert mask[a] == 1


def test_ae_manager_routes_to_greedy(monkeypatch):
    """AE_STRATEGY=greedy makes AEManager.ae return greedy.act's result."""
    import ae_manager as ae_manager_mod
    monkeypatch.setenv("AE_STRATEGY", "greedy")
    captured = {}

    def fake_act(belief, mask):
        captured["called"] = True
        return 4  # STAY
    monkeypatch.setattr(ae_manager_mod.greedy, "act", fake_act)

    mgr = ae_manager_mod.AEManager()
    # Minimal observation matching the AE input schema. Viewcones are zeroed —
    # belief.update will run but record nothing; the greedy path doesn't need
    # the viewcone contents for this test.
    import numpy as np
    obs = {
        "agent_viewcone": np.zeros((7, 5, 25), dtype=np.float32),
        "base_viewcone": np.zeros((5, 5, 25), dtype=np.float32),
        "direction": 0,
        "location": [0, 0],
        "base_location": list(mgr.prior.bases[next(iter(mgr.prior.bases))]),
        "health": [60.0],
        "frozen_ticks": 0,
        "base_health": [100.0],
        "team_resources": [0.0],
        "team_bombs": 0,
        "step": 0,
        "action_mask": [1, 1, 1, 1, 1, 0],
    }
    assert mgr.ae(obs) == 4
    assert captured.get("called") is True


def test_ae_manager_unknown_strategy_errors(monkeypatch):
    import ae_manager as ae_manager_mod
    monkeypatch.setenv("AE_STRATEGY", "not_a_strategy")
    try:
        ae_manager_mod.AEManager()
    except ValueError as e:
        assert "not_a_strategy" in str(e)
    else:
        raise AssertionError("expected ValueError")
