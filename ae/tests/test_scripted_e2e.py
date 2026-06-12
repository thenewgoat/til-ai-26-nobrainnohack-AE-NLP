"""Run a full novice episode: scripted agent_0 vs 5 random agents.

Asserts the agent is legal every step and scores clearly above an all-random
control run.
"""
from til_environment import bomberman_env
from til_environment.config import default_config

from ae_manager import AEManager


def _run(scripted: bool, seed: int) -> float:
    cfg = default_config()
    cfg.env.novice = True
    env = bomberman_env.basic_env(env_wrappers=[], cfg=cfg)
    env.reset(seed=seed)
    mgr = AEManager() if scripted else None
    total = 0.0
    for agent in env.agent_iter():
        obs, reward, term, trunc, _ = env.last()
        if agent == "agent_0":
            total += float(reward)
        if term or trunc:
            env.step(None)
            continue
        if agent == "agent_0" and scripted:
            action = mgr.ae(obs)
            assert obs["action_mask"][action] == 1, "illegal action"
        else:
            action = env.action_space(agent).sample()
        env.step(action)
    env.close()
    return total


def test_scripted_beats_random_control():
    scripted_score = _run(scripted=True, seed=0)
    random_score = _run(scripted=False, seed=0)
    print(f"scripted={scripted_score:.1f}  random={random_score:.1f}")
    assert scripted_score > 0.0
    assert scripted_score > random_score
