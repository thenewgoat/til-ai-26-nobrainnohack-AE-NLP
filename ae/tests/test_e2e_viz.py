"""End-to-end: a short self-play loop emits all viz artifacts under viz/."""
import glob
import os

import numpy as np
import torch

from critic import CentralizedCritic
from league import League
from metrics import MetricsLogger
from policy import SymbolicTransformerActor
from train_selfplay import (OpponentRegistry, RungLadder, AntiIdleShaper,
                            PPOConfig, collect_rollout, compute_advantages,
                            critic_values, ppo_update, make_optimizer,
                            EPISODE_LEN, _build_viz_slot_agents)
from visualize import render_episode


def test_short_loop_emits_all_viz_artifacts(tmp_path):
    torch.manual_seed(0)
    np.random.seed(0)
    viz_dir = os.path.join(str(tmp_path), "viz")
    os.makedirs(viz_dir, exist_ok=True)
    device = torch.device("cpu")

    actor = SymbolicTransformerActor()
    critic = CentralizedCritic()
    league = League()
    ladder = RungLadder(league)
    registry = OpponentRegistry(league)
    shaper = AntiIdleShaper(penalty=0.05, total_updates=2)
    cfg = PPOConfig()
    opt = make_optimizer(actor, critic, cfg)
    logger = MetricsLogger(viz_dir)
    viz_every = 1            # capture every update for the test

    learner_slots = ("agent_0", "agent_1", "agent_2")
    for update in range(1, 3):
        shaper.set_update(update)
        opponents = ladder.sample_opponents()
        buf, outcomes = collect_rollout(
            actor, registry, learner_slots, num_episodes=1,
            seed0=update * 7, opponent_members=opponents,
            reward_shaper=shaper, return_outcomes=True)
        for member, won in outcomes:
            league.record_result(member, won)
        values = critic_values(critic, buf.gstate, buf.gscalar, device)
        adv, ret = compute_advantages(buf.rewards, values, buf.dones,
                                      0.99, 0.95)
        losses = ppo_update(actor, critic, opt, buf, adv, ret, cfg, device)
        logger.log(update, {
            "policy_loss": losses["policy_loss"],
            "value_loss": losses["value_loss"],
            "entropy": losses["entropy"],
            "mean_return": float(np.mean(buf.rewards)),
            "rung": ladder.rung,
            "pool_size": len(league.checkpoints()),
            "anti_idle_coef": shaper.penalty * shaper._frac,
        })
        if update % viz_every == 0:
            logger.plot(os.path.join(viz_dir, f"metrics_u{update}.png"))
            logger.leaderboard(
                league, update,
                os.path.join(viz_dir, "leaderboard.csv"),
                os.path.join(viz_dir, f"leaderboard_u{update}.png"))
            slot_agents = _build_viz_slot_agents(
                actor, learner_slots, opponents, update)
            render_episode(slot_agents,
                           os.path.join(viz_dir, f"replay_u{update}.mp4"),
                           fps=5, max_steps=20, seed=update)
    logger.close()

    # all five artifact kinds are present under viz/
    assert glob.glob(os.path.join(viz_dir, "replay_u*.mp4"))
    assert os.path.exists(os.path.join(viz_dir, "metrics.csv"))
    assert glob.glob(os.path.join(viz_dir, "metrics_u*.png"))
    assert os.path.exists(os.path.join(viz_dir, "leaderboard.csv"))
    assert glob.glob(os.path.join(viz_dir, "leaderboard_u*.png"))
    # each artifact file is non-empty
    for f in glob.glob(os.path.join(viz_dir, "*")):
        if os.path.isfile(f):
            assert os.path.getsize(f) > 0, f
