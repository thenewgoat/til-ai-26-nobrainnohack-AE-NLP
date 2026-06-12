import numpy as np

from hybrid_eval import HybridAgent, intervention_rates, summarize_paired_deltas
from hybrid_rollout import build_hybrid_buffer
from policy import SymbolicTransformerActor
from scripted.handover import HandoverTrigger
from critic import STATE_PLANES, STATE_SCALARS
from features import (NUM_BASES, BASE_FIELDS, RAW_AGENT_SHAPE, RAW_BASE_SHAPE,
                      STACKED_GRID_CHANNELS, STACKED_SCALARS)
from policy import NUM_ACTIONS

GRID = 16


def _obs(mask, step=0):
    return {
        "agent_viewcone": np.zeros((7, 5, 25), np.float32).tolist(),
        "base_viewcone": np.zeros((7, 7, 25), np.float32).tolist(),
        "direction": 0, "location": [3, 3], "base_location": [1, 1],
        "health": [60.0], "frozen_ticks": 0, "base_health": [100.0],
        "team_resources": [0.0], "team_bombs": 2, "step": step,
        "action_mask": mask,
    }


def test_hybrid_agent_acts_legally_and_latches_handover():
    actor = SymbolicTransformerActor(d_model=16, n_layers=1, n_heads=2)
    agent = HybridAgent(actor, trigger=HandoverTrigger(step_fallback=5))
    a0 = agent.action(_obs([1, 0, 0, 0, 0, 0], step=0))
    assert a0 == 0
    assert agent.controller.handover_fired is False
    a1 = agent.action(_obs([1, 1, 1, 1, 1, 1], step=60))
    assert 0 <= a1 < 6
    assert agent.controller.handover_fired is True


def test_hybrid_agent_reset_rebuilds_controller():
    actor = SymbolicTransformerActor(d_model=16, n_layers=1, n_heads=2)
    agent = HybridAgent(actor, trigger=HandoverTrigger(step_fallback=5))
    agent.action(_obs([1, 1, 1, 1, 1, 1], step=60))
    assert agent.controller.handover_fired is True
    agent.reset()
    assert agent.controller.handover_fired is False


def _tr(actor_queried, proposed, executed):
    return dict(
        grid=np.zeros((STACKED_GRID_CHANNELS, GRID, GRID), np.float32),
        base_feats=np.zeros((NUM_BASES, BASE_FIELDS), np.float32),
        raw_agent=np.zeros(RAW_AGENT_SHAPE, np.float32),
        raw_base=np.zeros(RAW_BASE_SHAPE, np.float32),
        scalar=np.zeros(STACKED_SCALARS, np.float32),
        gstate=np.zeros((STATE_PLANES, GRID, GRID), np.float32),
        gscalar=np.zeros(STATE_SCALARS, np.float32),
        mask=np.ones(NUM_ACTIONS, bool),
        proposed=proposed, executed=executed, actor_queried=actor_queried,
        logp=0.0, reward=0.0, env_reward=0.0, done=0.0)


def test_intervention_rates():
    buf = build_hybrid_buffer([
        _tr(True, 1, 1), _tr(True, 1, 5), _tr(False, 4, 4), _tr(True, 2, 2),
    ], forward_bias=0.0)
    r = intervention_rates(buf)
    assert r["n"] == 4
    assert abs(r["actor_query_rate"] - 0.75) < 1e-9
    assert abs(r["forced_escape_rate"] - 0.25) < 1e-9
    assert abs(r["gate_override_rate"] - 0.25) < 1e-9
    assert abs(r["proposal_executed_disagreement"] - 0.25) < 1e-9


def test_intervention_rates_empty():
    buf = build_hybrid_buffer([], forward_bias=0.0)
    r = intervention_rates(buf)
    assert r["n"] == 0 and r["actor_query_rate"] == 0.0


def test_summarize_positive_deltas():
    s = summarize_paired_deltas([1.0, 2.0, 3.0, 2.0], seed=0)
    assert s["n"] == 4
    assert abs(s["mean"] - 2.0) < 1e-9
    assert s["ci_lo"] <= s["mean"] <= s["ci_hi"]
    assert s["ci_lo"] > 0


def test_summarize_empty():
    s = summarize_paired_deltas([])
    assert s["n"] == 0 and s["mean"] == 0.0
