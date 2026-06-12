"""collect_rollout returns per-opponent-slot outcomes; main() records them."""
from train_selfplay import OpponentRegistry, collect_rollout
from league import League
from policy import SymbolicTransformerActor


def test_collect_rollout_returns_outcomes():
    """collect_rollout returns (buf, outcomes); outcomes is a list of
    (Member, learner_won) pairs, one per (episode, opponent slot)."""
    actor = SymbolicTransformerActor()
    league = League()
    reg = OpponentRegistry(league)
    anchors = league.anchors()           # 6 scripted Members
    buf, outcomes = collect_rollout(
        actor, reg, learner_slots=("agent_0", "agent_1", "agent_2"),
        num_episodes=2, seed0=0, opponent_members=anchors,
        return_outcomes=True)
    # 3 opponent slots * 2 episodes = 6 recorded outcomes
    assert len(outcomes) == 6
    for member, won in outcomes:
        assert member in anchors
        assert won in (True, False)


def test_record_result_gives_members_nondefault_winrate():
    """After recording rollout outcomes, faced members have games>0 and a
    win-rate no longer at the 0.5 default; PFSP weights consequently differ."""
    actor = SymbolicTransformerActor()
    league = League()
    reg = OpponentRegistry(league)
    anchors = league.anchors()
    _, outcomes = collect_rollout(
        actor, reg, learner_slots=("agent_0", "agent_1", "agent_2"),
        num_episodes=2, seed0=0, opponent_members=anchors,
        return_outcomes=True)
    for member, won in outcomes:
        league.record_result(member, won)
    faced = {m for m, _ in outcomes}
    assert faced, "expected at least one faced member"
    for m in faced:
        assert m.games > 0
    # at least one faced member's win-rate has moved off 0.5
    assert any(m.learner_winrate() != 0.5 for m in faced)
    # PFSP weights are no longer all equal across members
    weights = league.pfsp_weights(league.members())
    assert len(set(round(w, 6) for w in weights)) > 1


def test_collect_rollout_default_returns_only_buffer():
    """Backwards-compatible: without return_outcomes, returns the buffer."""
    actor = SymbolicTransformerActor()
    reg = OpponentRegistry(League())
    buf = collect_rollout(actor, reg, learner_slots=("agent_0",),
                          num_episodes=1, seed0=0)
    assert hasattr(buf, "size")
