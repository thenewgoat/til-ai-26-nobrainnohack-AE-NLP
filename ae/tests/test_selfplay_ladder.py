"""Rung ladder: opponent set per rung; promotion gated on win-rate."""
from train_selfplay import RungLadder
from league import League
from scripted.strategies import STRATEGIES


def test_rung1_opponents_match_strategy_count():
    ladder = RungLadder(League(), promote_winrate=0.7)
    opps = ladder.current_opponents()
    assert len(opps) == len(STRATEGIES)
    assert all(m.kind == "scripted" for m in opps)
    assert ladder.rung == 1


def test_promotion_only_when_winrate_clears_threshold():
    ladder = RungLadder(League(), promote_winrate=0.7)
    assert ladder.try_promote(win_rate=0.5) is False
    assert ladder.rung == 1
    assert ladder.try_promote(win_rate=0.75) is True
    assert ladder.rung == 2


def test_rung2_uses_pfsp_checkpoint_pool(tmp_path):
    lg = League()
    lg.snapshot(str(tmp_path / "c0.pt"), update=1)
    lg.snapshot(str(tmp_path / "c1.pt"), update=2)
    ladder = RungLadder(lg, promote_winrate=0.7)
    ladder.try_promote(0.8)                     # -> rung 2
    opps = ladder.current_opponents()
    assert all(m.kind == "checkpoint" for m in opps)


def test_ladder_caps_at_rung3():
    ladder = RungLadder(League(), promote_winrate=0.7)
    ladder.try_promote(0.9)     # 1->2
    ladder.try_promote(0.9)     # 2->3
    assert ladder.try_promote(0.9) is False     # no rung 4
    assert ladder.rung == 3
