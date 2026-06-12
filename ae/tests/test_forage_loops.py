"""Unit tests for the offline forage-loop solver (ae/tools/build_forage_loops.py)."""
import json
import sys
from pathlib import Path

# ae/tools is not on the path conftest sets up — add it.
_TOOLS = Path(__file__).resolve().parents[1] / "tools"
sys.path.insert(0, str(_TOOLS))

from build_forage_loops import (  # noqa: E402
    INF, chebyshev, load_arena, load_respawn, route_ticks,
    cluster_collectibles, nearest_neighbour_tour, two_opt,
    loop_period, v_eff, loop_yield_rate, build_loops, assign_teams,
)

_SRC = Path(__file__).resolve().parents[1] / "src"


def test_load_arena_parses_walls_collectibles_bases(tmp_path):
    arena_json = {
        "grid_size": 4,
        "walls": [[0, 0, 0, False]],          # wall on RIGHT side of (0,0)
        "collectibles": [[1, 1, "mission"], [2, 2, "resource"]],
        "bases": {"0": [0, 0], "1": [3, 3]},
        "spawns": {},
    }
    p = tmp_path / "arena.json"
    p.write_text(json.dumps(arena_json))
    arena = load_arena(p)
    assert arena["grid_size"] == 4
    assert frozenset({(0, 0), (1, 0)}) in arena["blocked"]
    assert arena["collectibles"] == {(1, 1): "mission", (2, 2): "resource"}
    assert arena["bases"] == {0: (0, 0), 1: (3, 3)}


def test_load_respawn_indexes_by_x_then_y(tmp_path):
    # respawn_map.json is a nested list indexed [x][y] (matches dynamics.py).
    p = tmp_path / "respawn.json"
    p.write_text(json.dumps([[10, 11], [12, 13]]))
    respawn = load_respawn(p)
    assert respawn[(0, 0)] == 10
    assert respawn[(0, 1)] == 11
    assert respawn[(1, 0)] == 12


def test_route_ticks_straight_line_no_turns():
    # (0,0) facing RIGHT to (3,0): 3 FORWARD, no turns.
    ticks, facing = route_ticks(set(), 5, (0, 0), 0, (3, 0))
    assert ticks == 3
    assert facing == 0


def test_route_ticks_counts_a_turn():
    # (0,0) facing RIGHT to (0,2): turn to face DOWN (1), then 2 FORWARD.
    ticks, _ = route_ticks(set(), 5, (0, 0), 0, (0, 2))
    assert ticks == 3


def test_route_ticks_same_tile_is_zero():
    assert route_ticks(set(), 5, (2, 2), 0, (2, 2)) == (0, 0)


def test_route_ticks_wall_forces_detour():
    # Indestructible wall between (0,0) and (1,0); facing RIGHT.
    # Detour (0,0)->(0,1)->(1,1)->(1,0): 3 moves + turns.
    blocked = {frozenset({(0, 0), (1, 0)})}
    ticks, _ = route_ticks(blocked, 5, (0, 0), 0, (1, 0))
    assert ticks == 6        # turn DOWN, F, turn RIGHT, F, turn UP, F


def test_route_ticks_unreachable_returns_inf():
    # Box (1,1) in completely.
    blocked = {frozenset({(1, 1), (0, 1)}), frozenset({(1, 1), (2, 1)}),
               frozenset({(1, 1), (1, 0)}), frozenset({(1, 1), (1, 2)})}
    ticks, _ = route_ticks(blocked, 5, (0, 0), 0, (1, 1))
    assert ticks == INF


def test_cluster_groups_nearby_tiles():
    collectibles = {
        (0, 0): "recon", (1, 1): "recon",        # cluster A
        (10, 10): "mission", (11, 11): "recon",  # cluster B
    }
    clusters = cluster_collectibles(collectibles, radius=2)
    assert len(clusters) == 2
    # Mission (10,10) is the highest-value seed -> its cluster comes first.
    assert clusters[0] == [(10, 10), (11, 11)]
    assert clusters[1] == [(0, 0), (1, 1)]


def test_cluster_seeds_from_highest_value():
    # Three tiles in a row, radius 1. Seeding from the mission tile in the
    # middle absorbs both neighbours into one cluster; seeding from an end
    # tile would split them. A single 3-tile cluster confirms mission seeded.
    collectibles = {(5, 5): "recon", (6, 5): "mission", (7, 5): "recon"}
    clusters = cluster_collectibles(collectibles, radius=1)
    assert clusters == [[(5, 5), (6, 5), (7, 5)]]


def _grid_dist(a, b):
    """Plain Manhattan distance — a deterministic stand-in metric for tests."""
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def test_nearest_neighbour_tour_visits_all_once():
    tiles = [(0, 0), (0, 5), (0, 1), (0, 2)]
    tour = nearest_neighbour_tour(tiles, _grid_dist)
    assert sorted(tour) == sorted(tiles)
    assert tour[0] == (0, 0)
    assert tour[1] == (0, 1)            # nearest to the start


def test_two_opt_does_not_lengthen_a_tour():
    # A deliberately crossed tour around a square; 2-opt should not worsen it.
    square = [(0, 0), (0, 4), (4, 0), (4, 4)]
    before = sum(_grid_dist(square[i], square[(i + 1) % 4]) for i in range(4))
    improved = two_opt(square, _grid_dist)
    after = sum(_grid_dist(improved[i], improved[(i + 1) % 4])
                for i in range(4))
    assert after <= before
    assert sorted(improved) == sorted(square)


def test_two_opt_uncrosses_a_square():
    # Optimal cycle of a 4-corner square has length 16; the crossed order is 24.
    crossed = [(0, 0), (4, 4), (4, 0), (0, 4)]
    improved = two_opt(crossed, _grid_dist)
    length = sum(_grid_dist(improved[i], improved[(i + 1) % 4])
                 for i in range(4))
    assert length == 16


def test_loop_period_walks_the_cycle_with_facing():
    # Square loop (0,0)->(2,0)->(2,2)->(0,2) on an open 5x5 grid.
    # 8 moves around the square + 3 turns (corners 2-4; corner 1 is free
    # when starting facing RIGHT) = 11 ticks, minimised over start facings.
    tour = [(0, 0), (2, 0), (2, 2), (0, 2)]
    assert loop_period(tour, set(), 5) == 11


def test_loop_period_inf_when_a_leg_is_unroutable():
    blocked = {frozenset({(1, 1), (0, 1)}), frozenset({(1, 1), (2, 1)}),
               frozenset({(1, 1), (1, 0)}), frozenset({(1, 1), (1, 2)})}
    assert loop_period([(0, 0), (1, 1)], blocked, 5) == INF


def test_v_eff_adds_bomb_fuel_only_for_resource():
    assert v_eff("mission", bomb_value=10.0) == 5.0
    assert v_eff("recon", bomb_value=10.0) == 1.0
    # resource: 2.0 direct + (0.5 / 1.5) * 10 bomb-fuel.
    assert v_eff("resource", bomb_value=10.0) == 2.0 + (0.5 / 1.5) * 10.0
    assert v_eff("resource", bomb_value=0.0) == 2.0


def test_loop_yield_rate_formula():
    # Two mission tiles, period 10, both respawn every 5 steps (P >= d, full).
    tour = [(0, 0), (1, 0)]
    collectibles = {(0, 0): "mission", (1, 0): "mission"}
    respawn = {(0, 0): 5, (1, 0): 5}
    # yield = (5*1 + 5*1) / 10 = 1.0
    assert loop_yield_rate(tour, 10, collectibles, respawn, 0.0) == 1.0


def test_loop_yield_rate_discounts_slow_respawn():
    # Period 10, respawn 20 -> tile is ready only half the revisits.
    tour = [(0, 0)]
    collectibles = {(0, 0): "mission"}
    respawn = {(0, 0): 20}
    # yield = 5 * min(1, 10/20) / 10 = 5 * 0.5 / 10 = 0.25
    assert loop_yield_rate(tour, 10, collectibles, respawn, 0.0) == 0.25


def test_build_loops_produces_scored_loops():
    arena = {
        "grid_size": 8,
        "blocked": set(),
        "collectibles": {(1, 1): "resource", (2, 1): "resource",
                         (1, 2): "mission", (2, 2): "recon"},
        "bases": {0: (0, 0)},
    }
    respawn = {(x, y): 15 for x in range(8) for y in range(8)}
    loops = build_loops(arena, respawn, cluster_radius=3,
                        bomb_value_attack=10.0, bomb_value_endgame=0.0)
    assert len(loops) == 1
    loop = loops[0]
    assert sorted(tuple(w) for w in loop["waypoints"]) == \
        sorted(arena["collectibles"])
    assert loop["period"] > 0
    # Attack regime values resource bomb-fuel; endgame does not.
    assert loop["yield_attack"] > loop["yield_endgame"]
    assert loop["resource_leaning"] is True       # 2 of 4 tiles are resource


def test_assign_teams_picks_nearest_home_loop():
    loops = [
        {"waypoints": [[0, 0], [1, 0]], "yield_attack": 1.0},
        {"waypoints": [[9, 9], [8, 9]], "yield_attack": 2.0},
    ]
    teams = assign_teams(loops, {0: (0, 0), 1: (9, 9)})
    assert teams["0"]["home_loop"] == 0
    assert teams["1"]["home_loop"] == 1
    assert teams["0"]["order"] == [1, 0]          # sorted by yield, desc


def test_shipped_forage_loops_json_is_well_formed():
    # Runs after Step 4 has generated the file from the real map.
    path = _SRC / "forage_loops.json"
    assert path.exists(), "run build_forage_loops.py first (Step 4)"
    data = json.loads(path.read_text())
    assert data["loops"], "expected at least one loop"
    arena = load_arena(_SRC / "arena_map.json")
    for loop in data["loops"]:
        assert len(loop["waypoints"]) >= 2
        assert loop["period"] > 0
        for w in loop["waypoints"]:
            assert tuple(w) in arena["collectibles"]
    # Every one of the 6 teams gets a home loop and a full ordering.
    assert set(data["teams"]) == {str(t) for t in arena["bases"]}
    for team in data["teams"].values():
        assert 0 <= team["home_loop"] < len(data["loops"])
        assert sorted(team["order"]) == list(range(len(data["loops"])))
