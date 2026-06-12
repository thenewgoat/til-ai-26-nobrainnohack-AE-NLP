from scripted.belief import Belief
from scripted.danger import DangerMap
from scripted.layers import forage, sweep
from scripted.map_prior import MapPrior
from scripted.pathfind import build_planner
from scripted.strategies import StrategyParams


def _setup():
    prior = MapPrior.load()
    prior.identify_team(prior.bases[0])
    b = Belief()
    b.reset(prior)
    return b, prior


def test_forage_runs_with_bases_alive_when_endgame_not_required():
    b, prior = _setup()
    b.location = prior.spawns[0]["pos"]
    b.facing = prior.spawns[0]["facing"]
    b.step = 1
    b.team_bombs = 3
    b.enemy_bombs = {}
    danger = DangerMap({}, b)
    planner = build_planner(b, danger)
    # default params: forage disabled while bases live -> None
    assert forage(b, danger, planner, StrategyParams()) is None
    # forage_requires_endgame=False -> forage may act (None only if nothing in reach)
    relaxed = forage(b, danger, planner, StrategyParams(forage_requires_endgame=False))
    assert relaxed is None or 0 <= relaxed <= 5


def test_sweep_respects_camp_leash():
    b, prior = _setup()
    b.location = prior.spawns[0]["pos"]
    b.facing = prior.spawns[0]["facing"]
    b.step = 1
    b.team_bombs = 3
    b.enemy_bombs = {}
    danger = DangerMap({}, b)
    planner = build_planner(b, danger)
    # leash 0 around our base: no collectible qualifies unless on the base tile.
    leashed = sweep(b, danger, planner, StrategyParams(camp_leash=0))
    # unrestricted sweep almost certainly finds a target on the novice map.
    unrestricted = sweep(b, danger, planner, StrategyParams())
    assert unrestricted is not None
    assert leashed is None or 0 <= leashed <= 5


def test_strategy_params_survive_defaults():
    p = StrategyParams()
    assert p.openness_radius == 4
    assert p.openness_weight == 1.5
    assert p.bomb_drop_min == 5
    assert p.bomb_drop_buffer == 1
