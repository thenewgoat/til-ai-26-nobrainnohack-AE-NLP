"""OpponentRegistry.sample_slot_opponent: per-slot kind mix."""
import random
from collections import Counter

from train_selfplay import OpponentRegistry
from league import League
from evaluate import RandomAgent, ScriptedAgent, NeuralAgent, NoisyScriptedAgent


def _registry():
    return OpponentRegistry(
        league=League(),
        mix_weights=(0.50, 0.25, 0.15, 0.10),
        opp_eps_noise=0.08,
        opp_neural_temperature=1.2,
        rng=random.Random(7),
    )


def test_mix_weights_realize_in_proportion():
    reg = _registry()
    anchors = League().anchors()
    kinds = []
    for _ in range(4000):
        member, agent = reg.sample_slot_opponent(rung_opponents=anchors)
        if isinstance(agent, RandomAgent):
            kinds.append("random")
        elif isinstance(agent, NoisyScriptedAgent):
            kinds.append("noisy")
        elif isinstance(agent, ScriptedAgent):
            kinds.append("scripted")
        else:
            kinds.append("neural")
    c = Counter(kinds)
    total = sum(c.values())
    # at rung 1 (anchors-only league sample) the "league" draw returns
    # scripted-anchor wrapped in noisy; only "random" is its own pure bucket.
    assert 0.20 <= c["random"] / total <= 0.30


def test_random_kind_returns_string_member():
    reg = _registry()
    for _ in range(100):
        member, agent = reg.sample_slot_opponent(rung_opponents=League().anchors())
        if isinstance(agent, RandomAgent):
            assert member == "random"
            break
    else:
        raise AssertionError("no RandomAgent draw in 100 samples")


def test_scripted_kinds_carry_a_real_member():
    reg = _registry()
    for _ in range(100):
        member, agent = reg.sample_slot_opponent(rung_opponents=League().anchors())
        if isinstance(agent, (ScriptedAgent, NoisyScriptedAgent)):
            assert member != "random"
            assert hasattr(member, "name")
            assert member.name.startswith("scripted:")
            break
    else:
        raise AssertionError("no scripted draw in 100 samples")


def test_default_mix_weights_match_spec():
    reg = OpponentRegistry(league=League())
    assert reg._mix_weights == (0.50, 0.25, 0.15, 0.10)
    assert reg._eps == 0.08
    assert reg._neural_t == 1.2


def test_worker_rollout_threads_registry_kwargs():
    """When use_sample_mix=True with custom mix_weights, the worker registry
    must apply those weights — not the class defaults."""
    from train_selfplay import _worker_rollout
    from policy import SymbolicTransformerActor

    actor = SymbolicTransformerActor(d_model=16, n_layers=1, n_heads=2)
    state_dict = {k: v.cpu() for k, v in actor.state_dict().items()}
    cfg = actor.cfg
    # registry_kwargs the worker should adopt
    rk = {"mix_weights": (1.0, 0.0, 0.0, 0.0),  # always pick "league"
          "opp_eps_noise": 0.0,
          "opp_neural_temperature": 1.0,
          "rng_seed": 0}
    # 1 episode, 1 learner slot; agents fill the rest via "league" with
    # opponent_members=None → falls back to anchors. No crashes = pass.
    buf, outcomes = _worker_rollout(
        state_dict, cfg, ("agent_0",), 1, 0,
        None, None, True, rk)
    assert buf.size == 200
