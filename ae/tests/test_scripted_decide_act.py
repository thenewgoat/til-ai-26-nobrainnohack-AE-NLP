from scripted.belief import Belief
from scripted.decide import _first_legal, act
from scripted.layers import default, sweep
from scripted.strategies import StrategyParams
from scripted.danger import DangerMap
from scripted.map_prior import MapPrior
from scripted.pathfind import build_planner


def test_first_legal_falls_back_to_any_legal_action():
    # No preferred action is legal, but action 1 is -> return it.
    assert _first_legal([0, 1, 0, 0, 0, 0], [0, 2, 3]) == 1
    # Nothing legal at all -> STAY (last resort).
    assert _first_legal([0, 0, 0, 0, 0, 0], [0, 1, 2, 3]) == 4


def _belief(loc, facing=0):
    m = MapPrior.load()
    m.identify_team((13, 9))
    b = Belief()
    b.reset(m)
    b.location = loc
    b.facing = facing
    b.team_bombs = 3
    b.step = 10
    return b


def test_sweep_returns_a_move():
    b = _belief((8, 8))
    danger = DangerMap({}, b)
    p = build_planner(b, danger)
    assert sweep(b, danger, p, StrategyParams()) in (0, 1, 2, 3)


def test_default_returns_a_move():
    b = _belief((8, 8))
    danger = DangerMap({}, b)
    p = build_planner(b, danger)
    assert default(b, danger, p, StrategyParams()) in (0, 1, 2, 3)


def test_act_respects_action_mask():
    b = _belief((8, 8))
    # Mask forbids FORWARD and PLACE_BOMB.
    mask = [0, 1, 1, 1, 1, 0]
    a = act(b, mask)
    assert mask[a] == 1


def test_act_stay_when_frozen():
    b = _belief((8, 8))
    b.frozen_ticks = 2
    a = act(b, [1, 1, 1, 1, 1, 1])
    assert a == 4                            # STAY


def test_act_records_last_layer():
    b = _belief((8, 8))
    act(b, [1, 1, 1, 1, 1, 1])
    # the default `balanced` strategy reaches only these layers, plus the
    # _first_legal fallback (camp/hold are camper-only)
    assert b.last_layer in {
        "survive", "hunt", "strike", "forage", "sweep", "default",
        "first_legal",
    }


def test_last_layer_starts_none():
    b = _belief((8, 8))
    assert b.last_layer is None


def test_act_records_frozen_as_last_layer():
    b = _belief((8, 8))
    b.frozen_ticks = 2
    act(b, [1, 1, 1, 1, 1, 1])
    assert b.last_layer == "frozen"


def test_record_logs_own_bomb_only_on_place_bomb():
    from scripted.decide import _record
    from scripted.geometry import PLACE_BOMB, FORWARD
    b = _belief((4, 4))
    _record(b, FORWARD, "sweep")
    assert b.own_bombs == []                 # a move records nothing
    _record(b, PLACE_BOMB, "strike")
    assert len(b.own_bombs) == 1             # PLACE_BOMB records one
    assert b.own_bombs[0][0] == (4, 4)       # at the agent's tile
