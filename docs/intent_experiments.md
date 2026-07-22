# Intent-model sweep — results

Auto-appended by `scripts/run_sweep.py`. 10 epochs each, val/ade_m is the metric (metric-space mean displacement error over the reported horizon, EMA weights). `best epoch` = epoch of lowest val/ade_m; `train reg` is the last epoch's train regression loss (gap vs `val reg` = generalization).

| experiment | overrides | best ADE (mm) | FDE (mm) | best epoch | val reg | val vis | train reg | status |
|---|---|---|---|---|---|---|---|---|
| baseline | (none) | **4.34** | 7.57 | 8 | 0.850 | 1.119 | 0.110 | ok |
| wvis_0.5 | w_vis=0.5 | **4.33** | 7.56 | 9 | 0.846 | 1.103 | 0.095 | ok |
| wvis_0.25 | w_vis=0.25 | **4.33** | 7.61 | 9 | 0.844 | 0.959 | 0.090 | ok |
| reg_huber | reg=huber | **4.48** | 7.76 | 8 | 0.391 | 1.247 | 0.072 | ok |
| hist_8 | t_hist=8 | **4.38** | 7.62 | 8 | 0.864 | 1.117 | 0.110 | ok |
| pred_16 | t_pred=16 | **6.09** | 11.07 | 8 | 0.888 | 1.029 | 0.093 | ok |
| pred_4 | t_pred=4 | **2.96** | 4.80 | 8 | 0.910 | 1.145 | 0.128 | ok |
| deep | depth=8 | **4.33** | 7.53 | 8 | 0.845 | 1.106 | 0.113 | ok |
| wide | C=512, heads=8 | **4.38** | 7.66 | 8 | 0.858 | 1.239 | 0.101 | ok |
| hand_transformer | articulation=ergonomics, hand_kind=transformer | **4.39** | 7.58 | 8 | 0.865 | 1.120 | 0.134 | ok |
| query_feat | emit_query_feat=True | **4.31** | 7.54 | 9 | 0.837 | 1.332 | 0.118 | ok |
| no_wrist | use_wrist=False | **5.17** | 8.88 | 7 | 1.120 | 0.999 | 0.107 | ok |
| wrist_mat | wrist_repr=matrix | **4.26** | 7.48 | 8 | 0.821 | 1.088 | 0.116 | ok |
| node_pose | articulation=raw_node_pose | **4.29** | 7.44 | 8 | 0.832 | 1.082 | 0.134 | ok |
| no_norm | normalize=False | **9.18** | 16.08 | 9 | 0.000 | 1.026 | 0.000 | ok |
| nquery_32 | n_query=32 | **4.34** | 7.61 | 8 | 0.854 | 1.302 | 0.100 | ok |
| small | C=256, heads=4, depth=4 | **4.37** | 7.57 | 8 | 0.852 | 0.916 | 0.148 | ok |

## Analysis

All 8-step ADE numbers are directly comparable **except** `pred_4`/`pred_16` (different
horizons — compare their FDE at a fixed step instead) and `no_norm` (see below). Best-epoch
was 7–9 for every run, so 10 epochs was sufficient to capture the minimum.

### The one axis that moves ADE: the wrist frame

| config | ADE (mm) | vs baseline |
|---|---|---|
| `no_wrist` (drop wrist rotation) | 5.17 | **+19%** (much worse) |
| `6d` wrist (baseline) | 4.34 | — |
| `matrix` (9-D) wrist | **4.26** | **−2%** (best real config) |

Wrist orientation is **load-bearing** for object-motion prediction — unsurprising for
in-hand manipulation, where the object rides the wrist frame. Dropping it costs 19% and
raises `val/reg` to 1.12. Representing the rotation as a full 3×3 matrix beats the 6-D
Gram–Schmidt parameterization by a small but consistent margin (lowest `val/reg` 0.821 and
lowest FDE 7.48 in the whole sweep). **This is the sweep's headline recommendation:
`wrist_repr=matrix`.**

### Normalization is essential

`no_norm` = 9.18 mm (>2× worse). With `normalize=False` the Δ-target is raw metres
(~0.001–0.008), so its MSE is ~1e-5 (prints as `reg=0.000`) and produces almost no gradient
— the model barely trains. Per-channel Δ-standardization is what puts the target in a range
where MSE gives usable gradients. **Keep `normalize=True`.**

### Everything else is flat at ~4.3 mm (at the task/label floor)

- **Loss balancing** — `w_vis` 1.0→0.5→0.25: ADE unchanged (4.34/4.33/4.33). The visibility
  BCE head overfits (`val/vis` climbs) but is *decoupled* from the trajectory head, so
  down-weighting it cleans up `val/loss` without changing ADE. `reg=huber` is slightly worse
  (4.48); MSE wins. `w_vis=0.25` is a reasonable default (same ADE, less vis overfitting).
- **Horizon** — per-step error is roughly horizon-independent (FDE@4 ≈ 4.8, FDE@8 ≈ 7.6,
  FDE@16 ≈ 11.1; sub-linear growth, no bad error compounding). Horizon is a task choice, not
  an accuracy lever. `t_hist=8` doesn't beat `t_hist=4` — 4 frames already capture the motion cue.
- **Capacity** — the model is **over-parameterized ~3×**: `small` (10.6M) ties baseline
  (35M) at 4.37 mm, and `deep` (46M) / `wide` (61.5M) give nothing above baseline (4.33–4.38).
  If inference cost matters, `small` is a free 3.3× parameter reduction at equal accuracy.
- **Architecture** — transformer hand encoder (4.39) ≈ BiGRU; `query_feat` (4.31) is within
  batch-averaging noise and raises `val/vis`. Default architecture is fine.
- **Articulation encoding** — `node_pose` (raw 72-D keypoints, 4.29) ties ergonomics angles
  (20-D). Same information; ergonomics stays on efficiency grounds.
- **Supervision density** — `nquery_32` (4.34) = baseline. More query points just cost compute.

### Recommended config

Baseline **+ `wrist_repr=matrix`** (the only change that improves ADE), optionally
**+ `w_vis=0.25`** (free reduction of vis-head overfitting, no ADE cost). Everything else
stays at the baseline default (`ergonomics`, BiGRU, C=384/depth=6, `t_hist=4`, `t_pred=8`,
`n_query=16`, `normalize=True`, MSE). A combined confirmation run (`wrist_mat` + `wvis_0.25`)
is the natural next step before locking the config in.

### Code fix during the sweep

`small`/`wide` initially crashed: `IntentModelConfig` had a redundant `point_out` width field
(default 384) that `point_encoder.py` summed with the C-width positional code, so any `C≠384`
mismatched. Fixed by removing `point_out` and tying the scene-token width to `cfg.C` (it is
structurally forced to equal `C`). Added `_cfg_from_dict` so checkpoints saved with the old
`point_out` key still load. `wide` then completed; `small` was re-run after the fix and
also ties baseline (4.37 mm at 10.6M). **Uncommitted — pending review.**
| small | C=256, heads=4, depth=4 | **4.37** | 7.57 | 8 | 0.852 | 0.916 | 0.148 | ok |
| wvis_0 | w_vis=0.0 | **4.40** | 7.67 | 9 | 0.860 | 1.696 | 0.091 | ok |
| cam_hand | articulation=camera_node_pose | **4.19** | 7.40 | 9 | 0.814 | 1.166 | 0.136 | ok |
| ema_0.995 | decay=0.995 | **4.35** | 7.59 | 5 | 0.860 | 0.912 | 0.111 | ok |
| ema_0.99 | decay=0.99 | **4.36** | 7.58 | 3 | 0.862 | 0.811 | 0.111 | ok |
