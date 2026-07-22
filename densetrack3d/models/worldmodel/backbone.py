"""Query-token init (§6) + three-attention backbone (§7).

Query init: the token for query point ``n`` is seeded from the SHARED positional code
``pos_enc(x0_norm[n])`` (same module/weights that position the scene tokens ``S`` in §4),
broadcast across ``L_pred`` steps, plus a learned per-step embedding. Nothing step-invariant
(no pooled history / ``ctx``) is added -- the history reaches the model only through the
history-aware ``a_future`` (§5), FiLM-injected below. ``query_feat`` is OFF by default (§4/§6).

Backbone: per layer, in order, per-step FiLM(a_future) -> scene cross-attn(S) -> spatial
self-attn (over N) -> temporal self-attn (over L_pred, non-causal). Composed from
``CrossAttnBlock`` / ``AttnBlock`` in blocks.py (NOT EfficientUpdateFormer.forward, which has
image-grid local attention unsuited to scattered query points -- spec §3). FiLM is the SOLE
action injection: a per-step affine ``(1+gamma)*tok + beta``, zero-init so the model starts
effectively unconditioned (§7.1).
"""
import torch
import torch.nn as nn
from einops import rearrange, repeat

from densetrack3d.models.densetrack3d.blocks import AttnBlock, CrossAttnBlock


def init_query_tokens(x0_norm, pos_enc, step_emb, query_feat=None):
    """x0_norm (B, N, 3) normalized query xyz (== cloud[:, -N:]); pos_enc SHARED (§4).

    step_emb (L_pred, C) learned per-step embedding. Returns tok (B, L_pred, N, C).
    """
    L_pred, C = step_emb.shape
    seed = pos_enc(x0_norm)                                   # (B, N, C) position/identity (SAME code as S)
    if query_feat is not None:                               # OFF by default (§4/§6)
        seed = seed + query_feat                             # content add (already projected to C)
    tok = seed[:, None].expand(-1, L_pred, -1, -1).clone()   # (B, L_pred, N, C) same seed every step
    tok = tok + step_emb[None, :, None]                      # (B, L_pred, N, C) per-step positional
    return tok


class IntentBackbone(nn.Module):
    def __init__(self, C=384, depth=6, heads=6, mlp_ratio=4.0):
        super().__init__()
        self.depth = depth
        # Attention reshapes assuming inner_dim == C, so pin dim_head = C // heads
        # (its default dim_head=48 would give inner_dim=288 != 384 and crash).
        dh = C // heads
        self.cross = nn.ModuleList(CrossAttnBlock(C, C, heads, mlp_ratio, dim_head=dh) for _ in range(depth))
        self.space = nn.ModuleList(AttnBlock(C, heads, mlp_ratio=mlp_ratio, dim_head=dh) for _ in range(depth))
        self.time = nn.ModuleList(AttnBlock(C, heads, mlp_ratio=mlp_ratio, dim_head=dh) for _ in range(depth))
        self.film = nn.ModuleList(nn.Linear(C, 2 * C) for _ in range(depth))
        for f in self.film:                                  # start (1+gamma)=1, beta=0 (§7.1)
            nn.init.zeros_(f.weight)
            nn.init.zeros_(f.bias)

    def forward(self, tok, S, a_future):
        # tok (B, L_pred, N, C); S (B, M, C) position-aware; a_future (B, L_pred, C)
        B, Lp, N, C = tok.shape
        for l in range(self.depth):
            # per-step FiLM: affine, differs across tau -> never smears like a step-invariant bias
            gamma, beta = self.film[l](a_future).chunk(2, -1)          # (B, L_pred, C) each
            tok = (1 + gamma[:, :, None, :]) * tok + beta[:, :, None, :]

            # 1. scene cross-attn: every (tau, token) attends into the M scene tokens S
            x = rearrange(tok, "b l n c -> (b l) n c")
            x = self.cross[l](x, repeat(S, "b m c -> (b l) m c", l=Lp))
            # 2. spatial self-attn: within a step, across the N tokens
            x = self.space[l](x)
            tok = rearrange(x, "(b l) n c -> b l n c", b=B)
            # 3. temporal self-attn: across L_pred steps, per token, non-causal
            y = rearrange(tok, "b l n c -> (b n) l c")
            y = self.time[l](y)
            tok = rearrange(y, "(b n) l c -> b l n c", b=B)
        return tok                                                    # (B, L_pred, N, C)
