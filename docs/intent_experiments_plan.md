# Intent-model hyperparameter sweep — plan

**Goal:** find a good configuration for the object-flow intent model
(`x_{n,t:t+L} = f(P_t, q_hist, q_future)`), starting from the reference run below.
Autonomous sweep driven by `scripts/run_sweep.py`; live results in
[`intent_experiments.md`](intent_experiments.md).

## Protocol

- **Metric:** `val/ade_m` — metric-space mean displacement error (metres) over the
  reported horizon, measured on the **EMA** weights on the held-out episode split.
  Lower is better. Secondary: `val/fde_m` (final-step error), and `val/vis`/`val/reg`
  vs `train/reg` to read overfitting.
- **Budget:** 10 epochs per experiment (the reference run plateaued in `val/ade_m`
  by ~epoch 8, so 10 captures the achievable minimum without wasting GPU).
- **Isolation:** each experiment runs as its own `train_intent.py fit` subprocess
  under `logdirs/sweep/<name>/`, so failures don't cascade and GPU memory is released
  between runs. Metrics via CSVLogger (no W&B during the sweep).
- **One axis at a time**, all measured against the same `baseline` so deltas are
  attributable. Batch size (256), LR (4e-4), warmup (500), EMA (0.999) held fixed
  unless a size change demands otherwise.

## Reference run (established before the sweep)

Full config in `configs/intent.yaml`. Prior 15-epoch run: `val/ade_m` fell
9.1mm → **4.35mm** and plateaued by ~epoch 8, while `val/vis` **rose** 0.83 → 1.69
(the visibility BCE head overfits). Trajectory regression (`val/reg`) plateaued
healthily at ~0.85. On `epoch009`, EMA beat raw on every metric (ADE 4.72 vs 4.76mm,
vis 1.11 vs 1.52), per-step ADE grew cleanly 1.4mm → 8.0mm over the 8-step horizon.

**This directly motivates the loss-balancing experiments** — the composite `val/loss`
is dominated by a vis head that overfits, so `val/ade_m` (what we care about) may
improve, or at least the model may generalize better, by down-weighting `w_vis`.

## Axes explored

| axis | experiments | question |
|---|---|---|
| **loss balancing** | `wvis_0.5`, `wvis_0.25`, `reg_huber` | Does down-weighting the overfitting vis head help ADE? Is Huber more robust than MSE on Δ? |
| **horizon** | `hist_8`, `pred_16`, `pred_4` | Does more history help velocity estimation? How does ADE scale with the prediction horizon? |
| **capacity** | `small` (C256/d4/h4), `deep` (d8), `wide` (C512/h8) | Is the 34.7M baseline over/under-parameterized for ~108k windows? |
| **architecture** | `hand_transformer`, `query_feat` | Non-causal transformer hand encoder vs BiGRU; does feeding query point-features into the token seed help? |
| **hand representation** | `no_wrist`, `wrist_mat`, `node_pose` | Is the anchor-relative wrist rotation worth its 6 dims? 6D vs matrix? Ergonomics angles vs raw keypoints? |
| **normalization** | `no_norm` | Is per-channel Δ-standardization actually helping, or could raw-metre regression work? |
| **supervision density** | `nquery_32` | Do more query points per window (denser supervision) improve or just cost more? |

`d_q` (hand-encoder input width) is **derived** from `articulation`/`use_wrist`/`wrist_repr`
via a LightningCLI link, so the hand-representation experiments can't silently mis-shape
the encoder.

## Reading the results

The best single config may combine winners from independent axes (e.g. lower `w_vis`
+ a capacity tweak). After the one-axis sweep, the top 1–2 ideas get a combined
confirmation run. Nothing is committed automatically.
