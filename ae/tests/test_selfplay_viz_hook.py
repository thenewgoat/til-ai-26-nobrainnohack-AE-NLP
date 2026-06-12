"""train_selfplay's viz hook builds the replay slot_agents from the matchup."""
from train_selfplay import Args, _build_viz_slot_agents
from league import League
from policy import SymbolicTransformerActor


def test_args_has_viz_every():
    assert hasattr(Args(), "viz_every")
    assert Args().viz_every == 25


def test_build_viz_slot_agents_maps_all_six_slots():
    """The replay matchup: learner (NeuralAgent) in learner_slots, sampled
    opponent Members in the rest, all six slots labelled."""
    actor = SymbolicTransformerActor()
    league = League()
    learner_slots = ("agent_0", "agent_1", "agent_2")
    opponents = league.anchors()[:3]     # 3 scripted Members for 3 opp slots
    slot_agents = _build_viz_slot_agents(actor, learner_slots, opponents,
                                         update=25)
    assert set(slot_agents) == {f"agent_{k}" for k in range(6)}
    # every entry is (agent, label) with a non-empty label
    for slot, (agent, label) in slot_agents.items():
        assert hasattr(agent, "action") and hasattr(agent, "reset")
        assert isinstance(label, str) and label
    # learner slots are labelled with the update number
    for s in learner_slots:
        assert f"upd{25}" in slot_agents[s][1]
