"""Shared structural types for the object-flow intent model.

Three dicts form the contract between the data, model, and loss layers. They live
here (not in intent_model.py or flow_window_dataset.py) so neither side has to import
the other just to name the other's payload:

  - ``FlowItem``     : one window emitted by ``FlowWindowDataset.__getitem__`` -- numpy.
  - ``IntentBatch``  : a collated mini-batch (``scripts/train_intent.collate``) -- torch.
  - ``IntentOutput`` : the raw head outputs of ``IntentModel.forward`` -- torch.

These are ``TypedDict``s, so ``batch["q_future"]`` is a checkable ``torch.Tensor`` and the
per-field shapes (which PEP 484 can't express) stay in the trailing comments. Symbols:
``B`` batch, ``P`` cloud points, ``N`` query points, ``T_hist`` history steps,
``L_pred`` predicted steps, ``d_q`` hand-feature width.
"""
from __future__ import annotations

from typing import Any, TypedDict

import numpy as np
import torch


class FlowItem(TypedDict, total=False):
    """One sampling window (numpy)."""
    cloud: np.ndarray        # (P+N, 3)      float32  present-frame cloud + query seeds (network input)
    x0: np.ndarray           # (N, 3)        float32  query seeds x_{n,t}, always METRIC
    target: np.ndarray       # (L_pred, N, 3) float32 camera-frame positions, always METRIC
    target_vis: np.ndarray   # (L_pred, N)   bool     per-step visibility of each query point
    q_hist: np.ndarray       # (T_hist, d_q) float32  hand pose cue (history)
    q_future: np.ndarray     # (L_pred, d_q) float32  hand pose action (future)
    K: np.ndarray            # (4,)          float32  fx, fy, cx, cy (episode intrinsics)
    frame_meta: dict[str, Any]                       # episode dir, present frame t, stride_hz, ...
    dxyz_mean: np.ndarray    # (3,)          float32  per-step displacement mean
    dxyz_std: np.ndarray     # (3,)          float32  per-step displacement std


class IntentBatch(TypedDict, total=False):
    """A collated mini-batch (torch). Float fields are stacked from ``FlowItem``"""
    cloud: torch.Tensor        # (B, P+N, 3)      float32
    x0: torch.Tensor           # (B, N, 3)        float32  METRIC
    target: torch.Tensor       # (B, L_pred, N, 3) float32 METRIC
    target_vis: torch.Tensor   # (B, L_pred, N)   bool
    q_hist: torch.Tensor       # (B, T_hist, d_q) float32
    q_future: torch.Tensor     # (B, L_pred, d_q) float32
    K: torch.Tensor            # (B, 4)           float32
    dxyz_mean: torch.Tensor    # (B, 3)           float32
    dxyz_std: torch.Tensor     # (B, 3)           float32
    frame_meta: list[dict[str, Any]]              # one per item, kept off-device


class IntentOutput(TypedDict):
    """Raw head outputs of ``IntentModel.forward`` (torch), in standardized delta space.

    ``IntentModel.predict_trajectory`` consumes these and returns the composed metric trajectory
    ``(x_pred, vis_prob)`` as a separate tuple -- it does not add keys to this dict."""
    delta: torch.Tensor      # (B, L_pred, N, 3) standardized per-step displacement
    vis_logit: torch.Tensor  # (B, L_pred, N)    pre-sigmoid visibility logit
