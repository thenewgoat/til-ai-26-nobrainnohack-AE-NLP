from scripted.belief import Belief
from scripted.decide import record_final_action
from scripted.geometry import BACKWARD, FORWARD, PLACE_BOMB, STAY
from scripted.map_prior import MapPrior


def _belief(loc=(3, 3), facing=0):
    prior = MapPrior.load()
    prior.identify_team(prior.bases[0])
    b = Belief()
    b.reset(prior)
    b.location = loc
    b.facing = facing                 # 0 = RIGHT
    return b


def test_forward_sets_expected_location_ahead():
    b = _belief((3, 3), facing=0)      # facing RIGHT -> +x
    record_final_action(b, FORWARD, "rl_layer")
    assert b.expected_location == (4, 3)
    assert b.last_layer == "rl_layer"


def test_backward_sets_expected_location_behind():
    b = _belief((3, 3), facing=0)      # backward of RIGHT -> -x
    record_final_action(b, BACKWARD, "rl_layer")
    assert b.expected_location == (2, 3)


def test_stay_and_turns_expect_current_location():
    b = _belief((3, 3), facing=0)
    record_final_action(b, STAY, "forced_escape")
    assert b.expected_location == (3, 3)
    assert b.last_layer == "forced_escape"


def test_place_bomb_logs_own_bomb_and_returns_action():
    b = _belief((3, 3), facing=0)
    before = len(b.own_bombs)
    out = record_final_action(b, PLACE_BOMB, "gate:strike_gate")
    assert out == PLACE_BOMB
    assert len(b.own_bombs) == before + 1
    assert b.last_layer == "gate:strike_gate"
