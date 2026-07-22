"""Full assembly + loss for the object-flow intent model (spec §9).

``x_{n,t:t+T} = f(P_t, q_{t-T_hist:t}, q_{t:t+T_pred})`` -- one object cloud, hand-pose history
(velocity cue), future hand plan (action). Target = per-step displacement Delta (recover
``x = x0 + sum Delta``) + per-step visibility, camera frame.

Key design points folded in (see spec):
  * query seeds are read from the NORMALIZED cloud (``cloud[:, -N:]``, §2.1);
  * the query seed and the scene tokens ``S`` share ONE ``pos_enc`` instance so cross-attention
    matches positions directly (§4/§6), with ``query_feat`` off by default;
  * the sole hand signal is the history-aware ``a_future`` (no separate history stream, §5);
  * the backbone conditions purely by per-step FiLM (no hand-token axis, §7).
"""
from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from densetrack3d.models.loss import balanced_bce_loss
from densetrack3d.models.worldmodel.backbone import IntentBackbone, init_query_tokens
from densetrack3d.models.worldmodel.hand_encoder import HandEncoder
from densetrack3d.models.worldmodel.heads import Heads
from densetrack3d.models.worldmodel.point_encoder import PointEncoder, PosEnc
from densetrack3d.models.worldmodel.types import IntentBatch, IntentOutput


def _default_sa_cfg():
    return ((256, 0.2, 32, (32, 32, 64)),
            (128, 0.4, 32, (64, 64, 128)),
            (64, 0.8, 32, (128, 128, 256)))


@dataclass
class IntentModelConfig:
    C: int = 384
    depth: int = 6
    heads: int = 6
    mlp_ratio: float = 4.0
    n_query: int = 16                     # N (also the number of appended cloud rows)
    l_pred: int = 8                       # L_pred predicted steps (network runs over all L_pred)
    # Reporting horizon: predict_trajectory returns x_pred[:, :t_pred] (§10)
    t_pred: Optional[int] = None          # report horizon; None -> all l_pred steps (§10)
    # hand encoder
    articulation: str = "ergonomics"
    d_q: int = 26                         # ergonomics 20 + M_rel-6D 6
    hand_kind: str = "bigru"              # non-causal (§5)
    # point encoder -- scene-token width is ALWAYS C (S is summed with the C-width pos_enc code
    # in PointEncoder and cross-attended by the C-width backbone), so it is not a free knob.
    sa_cfg: tuple = field(default_factory=_default_sa_cfg)
    emit_query_feat: bool = False         # cross-attn into position-aware S (§4/§6)
    # shared positional code
    emb_C: int = 128
    # get_3d_embedding ramps to ~984 rad/unit at emb_C=128 (built for pixel inputs); on unit-scale
    # normalized coords pos_scale=1 aliases into a hash. 0.02 keeps smooth cos decay (§11 gate 2b).
    pos_scale: float = 0.02
    # loss
    w_vis: float = 1.0
    reg: str = "mse"                      # or "huber"


class IntentModel(nn.Module):
    def __init__(self, cfg: IntentModelConfig):
        super().__init__()
        self.cfg = cfg
        self.N = cfg.n_query
        self.pos_enc = PosEnc(emb_C=cfg.emb_C, C=cfg.C, scale=cfg.pos_scale)   # SHARED code
        self.pt = PointEncoder(self.pos_enc, out_dim=cfg.C,
                               sa_cfg=cfg.sa_cfg, emit_query_feat=cfg.emit_query_feat)
        self.hand = HandEncoder(d_q=cfg.d_q, hidden=cfg.C, kind=cfg.hand_kind)
        self.backbone = IntentBackbone(C=cfg.C, depth=cfg.depth, heads=cfg.heads,
                                       mlp_ratio=cfg.mlp_ratio)
        self.heads = Heads(C=cfg.C)
        self.step_emb = nn.Parameter(torch.zeros(cfg.l_pred, cfg.C))           # learned per-step
        nn.init.trunc_normal_(self.step_emb, std=0.02)

    def forward(self, batch: IntentBatch) -> IntentOutput:
        cloud = batch["cloud"]                                     # (B, P+N, 3) normalized
        # fail loudly on a dataset/config mismatch (else it surfaces as a cryptic FiLM broadcast error)
        assert cloud.shape[1] > self.N, (
            f"cloud has {cloud.shape[1]} rows <= n_query={self.N}; expected P+N query rows appended")
        assert batch["q_future"].shape[1] == self.cfg.l_pred, (
            f"q_future has {batch['q_future'].shape[1]} steps but cfg.l_pred={self.cfg.l_pred}")
        assert batch["x0"].shape[1] == self.N, (
            f"x0 has {batch['x0'].shape[1]} query points but cfg.n_query={self.N}")
        x0_norm = cloud[:, -self.N:]                               # normalized query seeds (== norm(x0))
        S, qfeat = self.pt(cloud, x0_norm)                         # S (B,M,C) position-aware; qfeat None by default
        a_future = self.hand(batch["q_hist"], batch["q_future"])["a_future"]   # (B,L_pred,C)
        tok = init_query_tokens(x0_norm, self.pos_enc, self.step_emb, qfeat)   # (B,L_pred,N,C)
        tok = self.backbone(tok, S, a_future)                      # (B,L_pred,N,C)
        delta, vis_logit = self.heads(tok)                         # standardized delta, vis logit
        return dict(delta=delta, vis_logit=vis_logit)

    @torch.no_grad()
    def predict_trajectory(
        self, batch: IntentBatch, out: Optional[IntentOutput] = None
        ) -> tuple[torch.Tensor, torch.Tensor]:
        """De-standardize delta and compose the metric trajectory anchored on METRIC x0 (§8)."""
        if out is None:
            out = self.forward(batch)
        std = batch["dxyz_std"][:, None, None]                     # (B,1,1,3)
        mean = batch["dxyz_mean"][:, None, None]
        d_metric = out["delta"] * std + mean                       # (B,L_pred,N,3) metres
        x_pred = batch["x0"][:, None] + d_metric.cumsum(dim=1)     # (B,L_pred,N,3)
        vis_prob = out["vis_logit"].sigmoid()
        # report only the real target horizon; tail steps exist for the endpoint-velocity pad (§4.2)
        if self.cfg.t_pred is not None:
            x_pred = x_pred[:, :self.cfg.t_pred]
            vis_prob = vis_prob[:, :self.cfg.t_pred]
        return x_pred, vis_prob


def intent_loss(
        out: IntentOutput, batch: IntentBatch, w_vis: float = 1.0, reg: str = "mse"
        ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Per-channel standardized MSE on per-step displacement (visibility-masked) + balanced BCE."""
    # ground-truth per-step displacement in metres
    traj = torch.cat([batch["x0"][:, None], batch["target"]], dim=1)   # (B, L_pred+1, N, 3)
    d_gt = torch.diff(traj, dim=1)                                     # (B, L_pred, N, 3) metres
    std = batch["dxyz_std"][:, None, None]                             # (B,1,1,3)
    mean = batch["dxyz_mean"][:, None, None]
    d_gt_std = (d_gt - mean) / std                                    # standardized target
    d_pred = out["delta"]                                            # already standardized space

    vis = batch["target_vis"]                                        # (B, L_pred, N) bool
    # TWO-SIDED delta mask: Delta_tau = x_tau - x_{tau-1} needs BOTH endpoints valid.
    prev_vis = torch.cat([torch.ones_like(vis[:, :1]), vis[:, :-1]], dim=1)
    m = (vis & prev_vis)[..., None].float()                         # (B, L_pred, N, 1)

    # Sanity check: Occluded points carry NaN coords, which are masked out (m=0).
    nan_dgt = ~torch.isfinite(d_gt)
    assert not bool((nan_dgt & (m > 0)).any()), \
        "non-finite target displacement on an unmasked (visible) step -- bad depth or a broken track"
    d_gt_std = torch.nan_to_num(d_gt_std) * m

    if reg == "huber":
        reg_l = F.huber_loss(d_pred * m, d_gt_std, reduction="none")
    else:
        reg_l = (d_pred * m - d_gt_std) ** 2
    reg_l = (reg_l * m).sum() / m.sum().clamp_min(1.0)

    vis_l = balanced_bce_loss(out["vis_logit"][..., None], vis.float()[..., None])
    total = reg_l + w_vis * vis_l
    return total, {"reg": reg_l.detach(), "vis": vis_l.detach()}
