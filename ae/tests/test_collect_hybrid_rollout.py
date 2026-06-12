import numpy as np

from hybrid_rollout import collect_hybrid_rollout
from policy import SymbolicTransformerActor
from scripted.handover import HandoverTrigger


def _actor():
    return SymbolicTransformerActor(d_model=16, n_layers=1, n_heads=2)


def test_selfplay_frozen_hybrid_opponents_run(tmp_path):
    # frozen checkpoints become HybridAgent opponents (self-play). selfplay_prob=1
    # forces every opponent slot to a frozen hybrid; buffer must still be valid.
    ckpt = str(tmp_path / "snap.pt")
    _actor().save_checkpoint(ckpt)
    buf = collect_hybrid_rollout(
        _actor(), ["agent_0"], num_episodes=1, seed0=0,
        trigger=HandoverTrigger(step_fallback=5),
        frozen_paths=[ckpt], selfplay_prob=1.0)
    assert buf.size > 0
    for i in range(buf.size):
        assert buf.masks[i][buf.executed_actions[i]]


def test_randomize_slot_runs_for_nonzero_slots():
    # learner plays a random slot 0-5; the global-state encoding must handle any
    # slot (not just agent_0) without crashing, and produce a valid buffer.
    buf = collect_hybrid_rollout(
        _actor(), ["agent_0"], num_episodes=4, seed0=0,
        trigger=HandoverTrigger(step_fallback=5), randomize_slot=True,
        opponent_names=["balanced", "adaptive"])
    assert buf.size > 0
    for i in range(buf.size):
        assert buf.masks[i][buf.executed_actions[i]]      # executed action legal


def test_collects_only_post_handover_ticks_with_required_fields():
    buf = collect_hybrid_rollout(
        _actor(), learner_slots=["agent_0"], num_episodes=1, seed0=0,
        trigger=HandoverTrigger(min_destroyed_enemy_bases=1, step_fallback=5),
        forward_bias=0.0)
    assert buf.size > 0
    assert buf.size <= 200
    assert buf.proposed_actions.shape == (buf.size,)
    assert buf.actor_queried.dtype == bool
    assert buf.gstate.shape[0] == buf.size
    assert buf.masks.shape == (buf.size, 6)
    for i in range(buf.size):
        assert buf.masks[i][buf.executed_actions[i]]      # executed action always legal
    assert buf.dones.sum() >= 1.0                          # run ends with a backfilled done


def test_handover_fallback_excludes_early_ticks():
    early = collect_hybrid_rollout(
        _actor(), ["agent_0"], 1, 0, trigger=HandoverTrigger(step_fallback=5))
    late = collect_hybrid_rollout(
        _actor(), ["agent_0"], 1, 0, trigger=HandoverTrigger(step_fallback=150))
    assert late.size < early.size


def test_anti_idle_penalizes_only_proposed_stay_actor_queried():
    pen = 0.1
    buf = collect_hybrid_rollout(
        _actor(), ["agent_0"], 1, 0,
        trigger=HandoverTrigger(step_fallback=5), anti_idle_penalty=pen)
    diff = buf.env_rewards - buf.rewards
    penalized = np.abs(diff - pen) < 1e-6
    for i in range(buf.size):
        if penalized[i]:
            assert buf.actor_queried[i] and buf.proposed_actions[i] == 4    # STAY=4
    for i in range(buf.size):
        if not (buf.actor_queried[i] and buf.proposed_actions[i] == 4):
            assert abs(diff[i]) < 1e-6


def test_scripted_opponents_run():
    # opponents drawn from named scripted strategies (not RandomAgent)
    buf = collect_hybrid_rollout(
        _actor(), ["agent_0"], 1, 0,
        trigger=HandoverTrigger(step_fallback=5),
        opponent_names=["balanced", "balanced_extreme_opening"])
    assert buf.size > 0
    for i in range(buf.size):
        assert buf.masks[i][buf.executed_actions[i]]
