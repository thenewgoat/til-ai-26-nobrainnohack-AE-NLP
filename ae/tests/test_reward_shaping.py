"""Anti-idle reward shaping anneals to zero; eval path is never shaped."""
from train_selfplay import AntiIdleShaper
from scripted.geometry import STAY


def test_penalty_applied_to_stay_action():
    shaper = AntiIdleShaper(penalty=0.1, total_updates=100)
    shaper.set_update(0)                       # full strength
    r = shaper(reward=0.0, action=STAY, step=10)
    assert r < 0.0                             # idling is penalized
    r_move = shaper(reward=0.0, action=0, step=10)
    assert r_move == 0.0                       # moving is not penalized


def test_penalty_anneals_to_zero():
    shaper = AntiIdleShaper(penalty=0.1, total_updates=100)
    shaper.set_update(0)
    early = shaper(0.0, STAY, 10)
    shaper.set_update(50)
    mid = shaper(0.0, STAY, 10)
    shaper.set_update(100)
    late = shaper(0.0, STAY, 10)
    assert early < mid < 0.0
    assert late == 0.0                         # fully annealed at the end


def test_real_reward_passes_through():
    shaper = AntiIdleShaper(penalty=0.1, total_updates=100)
    shaper.set_update(0)
    # a genuine +5 mission reward survives shaping (penalty only adds to STAY)
    assert shaper(5.0, action=0, step=10) == 5.0
