"""Shared policy definition for the AE challenge.

Defines SymbolicTransformerActor — a two-branch per-token transformer trained
by the spec C self-play trainer (ae/training/train_selfplay.py) and exported
to ONNX for the inference container. Keeping the network here means the
architecture cannot drift between training and serving.

The observation is the five-tensor contract emitted by FeatureBuilder.build():
    grid       [N, 85, 16, 16]  abstraction tile grid (K=5 stacked)
    base_feats [N, 5, 11]       abstraction per-base matrix
    raw_agent  [N, 7, 5, 25]    raw residual viewcone (agent)
    raw_base   [N, 7, 7, 25]    raw residual viewcone (base)
    scalar     [N, 50]          scalar token (K=5 stacked)
These become one token sequence: [CLS]·1 + tiles·256 + bases·5 + raw·84 +
scalar·1 = 347 tokens. Each group carries a learned type embedding so attention
can tell the branches apart. CLS -> action head.

Attention is hand-rolled (explicit matmul + softmax) so ONNX export uses only
standard ops. These dimensions mirror features.py's frozen contract;
test_policy_contract.py / test_feature_contract.py assert they agree.
"""

import torch
import torch.nn as nn
from torch.distributions import Categorical

NUM_ACTIONS = 6

# --- frozen contract dimensions (mirror features.py) --- #
GRID_CHANNELS = 17
NUM_BASES = 5
BASE_FIELDS = 11
RAW_CHANNELS = 25
SYMBOLIC_SCALARS = 10
STACK = 5
STACKED_GRID_CHANNELS = GRID_CHANNELS * STACK     # 85
STACKED_SCALARS = SYMBOLIC_SCALARS * STACK        # 50

GRID = 16
NUM_TILES = GRID * GRID            # 256 tile tokens
NUM_RAW_AGENT = 7 * 5              # 35 raw-agent tokens
NUM_RAW_BASE = 7 * 7               # 49 raw-base tokens
NUM_RAW_TOKENS = NUM_RAW_AGENT + NUM_RAW_BASE          # 84
NUM_TOKENS = 1 + NUM_TILES + NUM_BASES + NUM_RAW_TOKENS + 1   # 347

# token-group ids for the type embedding
TYPE_TILE = 0
TYPE_BASE = 1
TYPE_RAW = 2
TYPE_SCALAR = 3


def _layer_init(layer, std=2.0 ** 0.5, bias_const=0.0):
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias_const)
    return layer


class _MHA(nn.Module):
    """Multi-head self-attention over a full (unmasked) token sequence."""

    def __init__(self, d_model, n_heads, dropout):
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(f"d_model {d_model} not divisible by n_heads "
                             f"{n_heads}")
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.scale = self.d_head ** -0.5
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):                       # x [N, T, d]
        n, t, d = x.shape
        qkv = self.qkv(x).reshape(n, t, 3, self.n_heads, self.d_head)
        qkv = qkv.permute(2, 0, 3, 1, 4)        # [3, N, h, T, d_head]
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.dropout(attn)
        out = attn @ v
        out = out.transpose(1, 2).reshape(n, t, d)
        return self.proj(out)


class _TransformerBlock(nn.Module):
    """Pre-LN transformer block: attention + feed-forward, both residual."""

    def __init__(self, d_model, n_heads, ffn_dim, dropout):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = _MHA(d_model, n_heads, dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ffn_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(ffn_dim, d_model), nn.Dropout(dropout),
        )

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x


class SymbolicTransformerActor(nn.Module):
    """Two-branch transformer actor over the five-tensor feature contract.

    Input: grid [N,85,16,16], base_feats [N,5,11], raw_agent [N,7,5,25],
    raw_base [N,7,7,25], scalar [N,50]. Output: action logits [N,6]. No value
    head — the critic is a separate network. Model scale is configurable;
    self.cfg records the resolved config for checkpointing.
    """

    def __init__(self, d_model=64, n_layers=4, n_heads=4, ffn_dim=None,
                 dropout=0.1):
        super().__init__()
        if ffn_dim is None:
            ffn_dim = 4 * d_model
        self.cfg = {"d_model": d_model, "n_layers": n_layers,
                    "n_heads": n_heads, "ffn_dim": ffn_dim, "dropout": dropout}

        # per-branch token embeddings — tile_embed and scalar_embed consume
        # the K-frame stacked inputs along the channel/scalar axis.
        self.tile_embed = nn.Linear(STACKED_GRID_CHANNELS, d_model)
        self.base_embed = nn.Linear(BASE_FIELDS, d_model)
        self.raw_embed = nn.Linear(RAW_CHANNELS, d_model)
        self.scalar_embed = nn.Sequential(
            nn.Linear(STACKED_SCALARS, d_model), nn.GELU())

        # position tables: world-frame tiles, per-base slots, raw viewcone
        self.spatial_pos = nn.Parameter(torch.zeros(NUM_TILES, d_model))
        self.base_pos = nn.Parameter(torch.zeros(NUM_BASES, d_model))
        self.raw_pos = nn.Parameter(torch.zeros(NUM_RAW_TOKENS, d_model))
        # learned type embedding, one row per token group
        self.type_embed = nn.Parameter(torch.zeros(4, d_model))
        self.cls = nn.Parameter(torch.zeros(1, 1, d_model))
        for p in (self.spatial_pos, self.base_pos, self.raw_pos,
                  self.type_embed, self.cls):
            nn.init.normal_(p, std=0.02)

        self.blocks = nn.ModuleList([
            _TransformerBlock(d_model, n_heads, ffn_dim, dropout)
            for _ in range(n_layers)
        ])
        self.head_norm = nn.LayerNorm(d_model)
        # std=0.5 (not the usual PPO 0.01): the CLS signal is small (~0.2), so a
        # 0.01 head crushed logits to ~0.002 — a near-constant policy that argmax-
        # oscillated in 2 cells and never learned. 0.5 makes the head state-dependent
        # from step 0; the (now low) entropy coef no longer pins it toward uniform.
        self.head = _layer_init(nn.Linear(d_model, NUM_ACTIONS), std=0.5)

    def forward(self, grid, base_feats, raw_agent, raw_base, scalar):
        n = grid.shape[0]

        # abstraction tile tokens: [N,85,16,16] -> [N,256,85] -> [N,256,d]
        tiles = grid.permute(0, 2, 3, 1).reshape(n, NUM_TILES, STACKED_GRID_CHANNELS)
        tile_tok = (self.tile_embed(tiles) + self.spatial_pos
                    + self.type_embed[TYPE_TILE])

        # abstraction per-base tokens: [N,5,11] -> [N,5,d]
        base_tok = (self.base_embed(base_feats) + self.base_pos
                    + self.type_embed[TYPE_BASE])

        # raw residual tokens: [N,35,25] + [N,49,25] -> [N,84,d]
        ra = raw_agent.reshape(n, NUM_RAW_AGENT, RAW_CHANNELS)
        rb = raw_base.reshape(n, NUM_RAW_BASE, RAW_CHANNELS)
        raw = torch.cat([ra, rb], dim=1)
        raw_tok = (self.raw_embed(raw) + self.raw_pos
                   + self.type_embed[TYPE_RAW])

        # scalar token: [N,10] -> [N,1,d]
        scalar_tok = (self.scalar_embed(scalar).unsqueeze(1)
                      + self.type_embed[TYPE_SCALAR])

        cls_tok = self.cls.expand(n, -1, -1)
        x = torch.cat([cls_tok, tile_tok, base_tok, raw_tok, scalar_tok],
                      dim=1)                                  # [N,347,d]
        for blk in self.blocks:
            x = blk(x)
        return self.head(self.head_norm(x[:, 0]))             # CLS -> [N,6]

    def act(self, grid, base_feats, raw_agent, raw_base, scalar, mask,
            action=None, logit_bias=None):
        """Returns (action, log_prob, entropy); illegal actions masked out.

        `logit_bias` (a [A] or [N,A] tensor, default None) is added to the raw
        logits BEFORE the mask, so a biased illegal action stays illegal. Used by
        the training-time forward-bias exploration aid; inert when None."""
        logits = self.forward(grid, base_feats, raw_agent, raw_base, scalar)
        if logit_bias is not None:
            logits = logits + logit_bias
        logits = torch.where(mask, logits, torch.full_like(logits, -1e8))
        dist = Categorical(logits=logits)
        if action is None:
            action = dist.sample()
        return action, dist.log_prob(action), dist.entropy()

    def save_checkpoint(self, path):
        """Save weights + config so the architecture can be reconstructed."""
        torch.save({"state_dict": self.state_dict(), "cfg": self.cfg}, path)

    @classmethod
    def from_checkpoint(cls, path):
        """Rebuild a SymbolicTransformerActor from a save_checkpoint() file."""
        ckpt = torch.load(path, map_location="cpu")
        actor = cls(**ckpt["cfg"])
        actor.load_state_dict(ckpt["state_dict"])
        return actor
