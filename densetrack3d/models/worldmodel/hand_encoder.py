"""Sequential hand-action encoder (spec §5).

One ``C``-vector per hand pose. A pose at frame tau is a single configuration; the encoder
compresses that frame's ``d_q`` features to one ``C``-channel vector. There is NO per-keypoint
token axis -- keypoints / joint angles are just the ``d_q`` input columns.

A single sequential pass over the WHOLE hand sequence ``[q_hist ; q_future]`` makes each future
step's feature already integrate the history, so downstream needs no separate history injection
(§5, §7.2). Prefer a NON-CAUSAL encoder (BiGRU or a small transformer with a step positional
embedding): the whole hand plan is given up front, so every ``a_future[tau]`` should see the
entire history AND the entire future plan. Unidirectional GRU is the cheap fallback (``kind``).
"""
import torch
import torch.nn as nn


class HandEncoder(nn.Module):
    def __init__(self, d_q=26, hidden=384, kind="bigru", n_layers=1, max_len=64, nhead=6):
        super().__init__()
        self.kind = kind
        self.embed = nn.Linear(d_q, hidden)                   # per-frame embed: d_q -> C

        if kind in ("bigru", "gru"):
            bidir = kind == "bigru"
            self.rnn = nn.GRU(hidden, hidden if not bidir else hidden // 2,
                              num_layers=n_layers, batch_first=True, bidirectional=bidir)
        elif kind == "transformer":
            self.step_emb = nn.Parameter(torch.zeros(1, max_len, hidden))
            nn.init.trunc_normal_(self.step_emb, std=0.02)
            layer = nn.TransformerEncoderLayer(hidden, nhead, hidden * 4,
                                               batch_first=True, activation="gelu")
            self.rnn = nn.TransformerEncoder(layer, n_layers)
        else:
            raise ValueError(f"unknown hand encoder kind={kind}")

    def forward(self, q_hist, q_future):
        # q_hist (B, T_hist, d_q), q_future (B, L_pred, d_q)
        T_hist = q_hist.shape[1]
        q_seq = torch.cat([q_hist, q_future], dim=1)          # (B, L, d_q),  L = T_hist + L_pred
        e = self.embed(q_seq)                                 # (B, L, C)  one C per pose
        if self.kind in ("bigru", "gru"):
            seq, _ = self.rnn(e)                              # (B, L, C)
        else:
            seq = self.rnn(e + self.step_emb[:, :e.shape[1]])
        a_future = seq[:, T_hist:]                            # (B, L_pred, C) history-aware
        return dict(a_future=a_future)
