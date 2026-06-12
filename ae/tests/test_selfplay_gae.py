"""GAE over a multi-slot buffer using the centralized critic's values."""
import numpy as np
import torch

from train_selfplay import compute_advantages
from critic import CentralizedCritic


def test_gae_shapes_and_per_slot_episode_boundaries():
    # 2 learner slots, 4-step episodes -> buffer laid out slot-major
    rewards = np.array([0, 0, 0, 1, 0, 0, 0, 2], np.float32)
    dones = np.array([0, 0, 0, 1, 0, 0, 0, 1], np.float32)
    values = np.zeros(8, np.float32)
    adv, ret = compute_advantages(rewards, values, dones,
                                  gamma=0.99, gae_lambda=0.95)
    assert adv.shape == (8,) and ret.shape == (8,)
    # the terminal transition's advantage equals its reward (value 0, no
    # bootstrap past a done)
    assert abs(adv[3] - 1.0) < 1e-5
    assert abs(adv[7] - 2.0) < 1e-5


def test_critic_values_feed_gae():
    from critic import STATE_PLANES, STATE_SCALARS
    critic = CentralizedCritic()
    planes = torch.zeros(8, STATE_PLANES, 16, 16)
    scalars = torch.zeros(8, STATE_SCALARS)
    with torch.no_grad():
        values = critic(planes, scalars).numpy()
    assert values.shape == (8,)
