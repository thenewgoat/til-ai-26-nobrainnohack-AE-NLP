"""BC training reduces loss and the clone tracks the teacher."""
import numpy as np
import torch

from bc import collect_dagger_dataset, train_bc, bc_gate
from policy import SymbolicTransformerActor


def test_train_bc_reduces_loss():
    ds = collect_dagger_dataset("balanced", None, 1.0, 3, [0, 1, 2])
    actor = SymbolicTransformerActor()
    history = train_bc(actor, ds, epochs=5, batch_size=64, lr=1e-3)
    assert history[-1] < history[0]        # cross-entropy loss falls


def test_bc_overfits_tiny_set():
    """On a tiny dataset BC should reach near-perfect teacher imitation."""
    ds = collect_dagger_dataset("balanced", None, 1.0, 1, [0])[:32]
    actor = SymbolicTransformerActor()
    train_bc(actor, ds, epochs=200, batch_size=32, lr=1e-3)
    grids = torch.from_numpy(np.stack([s.grid for s in ds]))
    base_feats = torch.from_numpy(np.stack([s.base_feats for s in ds]))
    raw_agents = torch.from_numpy(np.stack([s.raw_agent for s in ds]))
    raw_bases = torch.from_numpy(np.stack([s.raw_base for s in ds]))
    scalars = torch.from_numpy(np.stack([s.scalar for s in ds]))
    masks = torch.from_numpy(np.stack([s.mask for s in ds]))
    labels = torch.tensor([s.action for s in ds])
    with torch.no_grad():
        logits = actor(grids, base_feats, raw_agents, raw_bases, scalars)
        logits = torch.where(masks, logits, torch.full_like(logits, -1e8))
        acc = (logits.argmax(-1) == labels).float().mean().item()
    assert acc > 0.9


def test_bc_gate_returns_bool_and_margin():
    actor = SymbolicTransformerActor()
    passed, detail = bc_gate(actor, teacher_strategy="balanced",
                             seeds=[0, 1], tolerance=0.05)
    assert isinstance(passed, bool)
    assert "clone" in detail and "teacher" in detail
