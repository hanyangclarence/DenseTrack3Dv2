"""Displacement + visibility heads (spec §8).

``delta`` is predicted in STANDARDIZED space (unit-variance per channel); de-standardize with
the item's ``dxyz_std/mean`` before composing a metric trajectory (done in intent_model /
eval, not here). The delta head is init'd small (trunc_normal std 1e-3) so the model starts
near "no motion" -- a good prior for short horizons and stable early training.
"""
import torch.nn as nn


class Heads(nn.Module):
    def __init__(self, C=384):
        super().__init__()
        self.delta = nn.Linear(C, 3)                          # per-step displacement (standardized space)
        self.vis = nn.Linear(C, 1)                            # per-step visibility logit
        nn.init.trunc_normal_(self.delta.weight, std=1e-3)    # start near no-motion (§8)
        nn.init.zeros_(self.delta.bias)

    def forward(self, tok_obj):                               # tok_obj (B, L_pred, N, C)
        delta = self.delta(tok_obj)                           # (B, L_pred, N, 3) standardized
        vis_logit = self.vis(tok_obj).squeeze(-1)             # (B, L_pred, N)
        return delta, vis_logit
