"""AEManager serves the scripted agent end to end."""
from til_environment import bomberman_env
from til_environment.config import default_config

from ae_manager import AEManager


def _novice_obs(agent="agent_0"):
    cfg = default_config()
    cfg.env.novice = True
    env = bomberman_env.basic_env(env_wrappers=[], cfg=cfg)
    env.reset(seed=0)
    obs = env.observe(agent)
    env.close()
    return obs


def test_returns_legal_action_at_step_0():
    mgr = AEManager()
    obs = _novice_obs()
    action = mgr.ae(obs)
    assert isinstance(action, int)
    assert obs["action_mask"][action] == 1


def test_identifies_team_from_base_location():
    mgr = AEManager()
    obs = _novice_obs()
    mgr.ae(obs)
    assert mgr.prior.team == 0               # agent_0 base is (13,9)


def test_step_0_resets_internal_state():
    mgr = AEManager()
    obs = _novice_obs()
    mgr.ae(obs)
    mgr.belief.collected.add((0, 0))
    obs2 = dict(obs)
    obs2["step"] = 0
    mgr.ae(obs2)                             # step 0 again => fresh episode
    assert (0, 0) not in mgr.belief.collected
