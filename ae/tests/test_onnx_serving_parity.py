"""End-to-end serving parity: ONNX NeuralAEManager == torch actor argmax."""
import numpy as np
import torch

from ae_manager import NeuralAEManager
from export_onnx import export_actor
from features import FeatureBuilder
from policy import SymbolicTransformerActor
from til_environment import bomberman_env
from til_environment.config import default_config


def test_served_action_matches_torch_argmax(tmp_path):
    torch.manual_seed(0)
    actor = SymbolicTransformerActor()
    actor.eval()
    onnx_path = tmp_path / "policy.onnx"
    export_actor(actor, str(onnx_path))

    cfg = default_config()
    cfg.env.novice = True
    env = bomberman_env.basic_env(env_wrappers=[], cfg=cfg)
    env.reset(seed=0)

    mgr = NeuralAEManager(onnx_path=str(onnx_path))
    ref_fb = FeatureBuilder()

    checked = 0
    for slot in env.agent_iter():
        obs, reward, term, trunc, _ = env.last()
        if term or trunc:
            env.step(None)
            continue
        if slot == "agent_0":
            served = mgr.ae(obs)
            grid, base_feats, raw_agent, raw_base, scalar = ref_fb.build(obs)
            mask = np.asarray(obs["action_mask"], dtype=bool).reshape(-1)
            with torch.no_grad():
                logits = actor(
                    torch.from_numpy(grid).unsqueeze(0),
                    torch.from_numpy(base_feats).unsqueeze(0),
                    torch.from_numpy(raw_agent).unsqueeze(0),
                    torch.from_numpy(raw_base).unsqueeze(0),
                    torch.from_numpy(scalar).unsqueeze(0),
                )[0].numpy()
            ref = int(np.argmax(np.where(mask, logits, -1e8)))
            assert served == ref
            checked += 1
            env.step(served)
        else:
            env.step(env.action_space(slot).sample())
        if checked >= 25:
            break
    env.close()
    assert checked >= 25
