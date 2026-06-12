"""League pool: scripted anchors are permanent; snapshots accumulate."""
from league import League
from scripted.strategies import STRATEGIES


def test_every_strategy_is_an_anchor():
    lg = League()
    names = {m.name for m in lg.members()}
    for strat in STRATEGIES.keys():
        assert f"scripted:{strat}" in names


def test_anchor_count_matches_strategies_count():
    lg = League()
    anchors = [m for m in lg.members() if m.kind == "scripted"]
    assert len(anchors) == len(STRATEGIES)


def test_snapshot_adds_a_checkpoint_member(tmp_path):
    lg = League()
    n0 = len(lg.members())
    lg.snapshot(str(tmp_path / "ckpt_0.pt"), update=10)
    assert len(lg.members()) == n0 + 1
    assert any(m.kind == "checkpoint" for m in lg.members())


def test_anchors_never_removed_when_pool_capped(tmp_path):
    lg = League(max_checkpoints=2)
    for i in range(5):
        lg.snapshot(str(tmp_path / f"ckpt_{i}.pt"), update=i)
    ckpts = [m for m in lg.members() if m.kind == "checkpoint"]
    anchors = [m for m in lg.members() if m.kind == "scripted"]
    assert len(ckpts) == 2          # pool capped
    assert len(anchors) == len(STRATEGIES)   # anchors untouched
