# Intent-Model Repo Split — Design

**Date:** 2026-07-23
**Status:** approved (design), pending spec review

## Goal

Extract the object-flow **intent model** work out of `DenseTrack3Dv2` into a new
standalone repo at `/home/labeng/yanghan/code/intent-model/`, running on **Python 3.12**
(so it can `import rclpy` from ROS Jazzy for live inference, alongside `torch`).
`DenseTrack3Dv2` stays as the **data-processing pipeline** (mcap → RGB-D → segmentation →
tracking → smoothing → `object_flow.pkl` / `hand.pkl`), essentially unchanged.

## Why this split is possible (and why now)

The intent-model code imports only a **small, well-bounded** slice of the DenseTrack3D
package, and none of it pulls in the heavy tracker / UniDepth / Grounded-SAM stack. Those
hard-to-upgrade dependencies stay in `DenseTrack3Dv2`, which is exactly what frees the new
repo to move to py3.12. The runtime boundary between the two repos is a **file on disk**
(`object_flow.pkl` + `hand.pkl`), not a code import — so after the split there is **zero
runtime import** from the new repo back into `DenseTrack3Dv2`.

## Architecture: the two-repo boundary

```
DenseTrack3Dv2 (py3.10, data processing)          intent-model (py3.12, model)
─────────────────────────────────────            ─────────────────────────────────
mcap → RGB-D → seg → track → smooth               reads object_flow.pkl + hand.pkl
      → gen_flow_labels.py                  ═══►   → FlowWindowDataset → IntentModel
      → object_flow.pkl / hand.pkl          disk   → train / eval / viz / (rclpy infer)
                                          contract
```

**One-way, disk-mediated.** `preprocess/` never imports intent code (verified). The intent
code's only ties back into `DenseTrack3Dv2` are severed three ways below.

## Shared code — three classes, three treatments

The intent code imports exactly three things from the DenseTrack3D package plus one
`preprocess/` module. Each is handled so the new repo has **no** `densetrack3d.*` or
`preprocess.*` import:

### 1. `hand_frame_transforms.py` — MOVE canonical, VENDOR copy back
Numpy-only geometry (frame-P placement, wrist rotation, joint-jitter). It is the
**training-time hand→camera placement** (`placed_hand_camera` is the augmentation choke
point), so it is training-critical. It is imported by BOTH sides:
- intent side: `data/flow_window_dataset.py`, `data/viz_flow_window_item.py`
- data side: `preprocess/viz_hand_cloud_live.py` (stays in `DenseTrack3Dv2`)

**Treatment:** the **canonical** copy moves to the new repo (`intent-model/data/hand_frame_transforms.py`),
because training correctness lives there. `DenseTrack3Dv2` keeps a **vendored copy** at its
current path for `viz_hand_cloud_live.py`. Both copies get a header comment naming the other
and stating they must stay in sync. Drift risk is low (the transforms are frozen), but the
note makes the coupling explicit.

### 2. DenseTrack model primitives — COPY into `intent_model/modules/`
Shared with DenseTrack but NOT used on the data-processing side, so a plain copy (no
vendor-back). Exact source ranges in `DenseTrack3Dv2` today:
- `models/densetrack3d/blocks.py`: `Mlp` (L68–112), `Attention` (L462–553),
  `AttnBlock` (L934–965), `CrossAttnBlock` (L967–999) → `intent_model/modules/nn_blocks.py`
  (~210 lines, 4 classes). These use only each other; they do **not** use `bilinear_sampler`,
  so `model_utils` is not pulled in through this path. The `flex_attention` import in
  `blocks.py` is behind a `try/except` and is **not** referenced by `Mlp`/`Attention`
  (they use `torch.nn.functional.scaled_dot_product_attention` + `SDPBackend`), so it is
  omitted from the copy.
- `models/embeddings.py`: `get_3d_embedding` (L226–253, 29 lines, no local-func calls,
  no intra-repo deps) → `intent_model/modules/embeddings.py`
- `models/loss.py`: `balanced_bce_loss` (L75–98) + `models/model_utils.py`:
  `reduce_masked_mean` (L112–159, needed by balanced_bce_loss) → `intent_model/modules/losses.py`.
  Keep the `jaxtyping` `Float`/`Int64` annotations (jaxtyping is a py3.12-compatible dep).

### 3. Two 2D-viz helpers — COPY into `intent-model/scripts/viz_helpers_2d.py`
`viz_intent_predictions.py` imports `render_2d_overlay` + `rainbow_colors_by_position` from
`preprocess/track_windowed.py`. Those two functions are self-contained (33 + 22 lines, no
local-func calls, need only cv2/numpy/matplotlib), but `track_windowed.py` as a module pulls
in the whole DenseTrack tracker at import time. **Treatment:** copy just those two functions
into a new `scripts/viz_helpers_2d.py` in the new repo; `viz_intent_predictions.py` imports
from there.

## File map

### New repo: `/home/labeng/yanghan/code/intent-model/`

```
intent-model/
├── intent_model/
│   ├── __init__.py                 # exports IntentModel, IntentModelConfig, intent_loss
│   ├── intent_model.py             ← MOVE densetrack3d/models/worldmodel/intent_model.py
│   ├── backbone.py                 ← MOVE .../worldmodel/backbone.py
│   ├── hand_encoder.py             ← MOVE .../worldmodel/hand_encoder.py
│   ├── heads.py                    ← MOVE .../worldmodel/heads.py
│   ├── point_encoder.py            ← MOVE .../worldmodel/point_encoder.py
│   ├── types.py                    ← MOVE .../worldmodel/types.py
│   └── modules/                    # COPIED, vendored DenseTrack primitives
│       ├── __init__.py
│       ├── nn_blocks.py            ← COPY Mlp, Attention, AttnBlock, CrossAttnBlock
│       ├── embeddings.py           ← COPY get_3d_embedding
│       └── losses.py               ← COPY balanced_bce_loss + reduce_masked_mean
├── data/
│   ├── flow_window_dataset.py      ← MOVE data/flow_window_dataset.py
│   ├── viz_flow_window_item.py     ← MOVE data/viz_flow_window_item.py
│   ├── hand_frame_transforms.py    ← MOVE preprocess/hand_frame_transforms.py (CANONICAL)
│   └── flow_stats.npz              ← MOVE data/flow_stats.npz (3.3 KB, regenerable)
├── scripts/
│   ├── train_intent.py             ← MOVE
│   ├── compute_flow_stats.py       ← MOVE
│   ├── analyze_intent_ckpt.py      ← MOVE
│   ├── viz_intent_predictions.py   ← MOVE (imports viz_helpers_2d, not track_windowed)
│   ├── viz_helpers_2d.py           ← COPY render_2d_overlay + rainbow_colors_by_position
│   └── sweep_intent.sh             ← MOVE (update python path + cwd)
├── configs/
│   └── intent.yaml                 ← MOVE (update data_root, stats path)
├── docs/specs/
│   ├── 2026-07-17-object-flow-intent-model-design.md          ← MOVE
│   ├── 2026-07-20-object-flow-model-architecture-detail.md    ← MOVE
│   └── 2026-07-22-hand-joint-jitter-augmentation-design.md    ← MOVE
├── docs/
│   ├── 2026-07-22-hand-joint-jitter-augmentation.md  ← MOVE (plan for the jitter spec)
│   ├── intent_experiments.md       ← MOVE
│   └── intent_experiments_plan.md  ← MOVE
├── logdirs/                        # gitignored; baseline ckpt copied in AFTER it finishes
├── pyproject.toml                  # NEW (py3.12)
├── requirements.txt                # NEW
├── .gitignore                      # NEW
└── README.md                       # NEW
```

### This repo `DenseTrack3Dv2` — changes

- `preprocess/hand_frame_transforms.py` — KEEP as a vendored copy (add sync-note header).
- `scripts/gen_flow_labels.py` — KEEP (data-side: imports `preprocess.extract_mcap_rgbd`).
- Everything else (`preprocess/*`, `track_windowed.py`, demos, submodules) — unchanged.
- The moved intent files are **deleted** from `DenseTrack3Dv2` only after the new repo is
  verified working (kept during verification so nothing is lost mid-migration).
- `object_flow_models_plan.md` — STAYS (not moved).

## Import rewrites (mechanical core)

| Old (DenseTrack3Dv2)                                     | New (intent-model)                          |
|----------------------------------------------------------|---------------------------------------------|
| `densetrack3d.models.worldmodel.X`                       | `intent_model.X`                            |
| `densetrack3d.models.worldmodel` (the package)           | `intent_model`                              |
| `densetrack3d.models.densetrack3d.blocks`                | `intent_model.modules.nn_blocks`            |
| `densetrack3d.models.embeddings`                         | `intent_model.modules.embeddings`           |
| `densetrack3d.models.loss`                               | `intent_model.modules.losses`               |
| `preprocess.hand_frame_transforms`                       | `data.hand_frame_transforms`                |
| `preprocess.track_windowed` (2 helpers)                  | `scripts.viz_helpers_2d` (local copy)       |

`scripts/*` and `data/*` in the new repo import via the top-level package names
(`intent_model`, `data`, `scripts`); the new repo runs with its root on `sys.path`
(as `train_intent.py` already does via `sys.path.insert`).

## Environment (py3.12 + rclpy)

- New conda env `intent`, **Python 3.12**.
- `torch==2.5.1` (+ matching torchvision) — ships cp312 wheels.
- `lightning`, `einops`, `jaxtyping`, `numpy`, `wandb`, `viser`, `opencv-python`,
  `matplotlib`, `tqdm` — all have 3.12 wheels.
- **`rclpy`**: NOT pip-installed. Comes from `source /opt/ros/jazzy/setup.bash` (ROS Jazzy
  is built for py3.12, installed at `/opt/ros/jazzy` on this host, so its `rclpy` matches the
  env interpreter). Verified during implementation with a `python -c "import rclpy, torch"`
  smoke test under py3.12 with Jazzy sourced.
- No UniDepth / Grounded-SAM / DepthCrafter in the new repo — those stay in `DenseTrack3Dv2`,
  so their (py3.10-era) pins do not constrain the new py3.12 env.

## Git & checkpoints

- New repo: `git init`, `.gitignore` (logdirs/, wandb/, `__pycache__/`, `*.ckpt`, data root),
  single initial commit "Initial commit: object-flow intent model (split from DenseTrack3Dv2)".
  **No git history transfer** (intent commits were interleaved with data commits; a filtered
  history would be partial). Per-commit intent history stays recoverable in `DenseTrack3Dv2`.
- **Checkpoints: NOT moved now.** After the currently-training baseline sweep run finishes,
  copy that baseline `.ckpt` into `intent-model/logdirs/` (on disk, gitignored) so inference
  works immediately in the new repo.
- Standing rule: **no commits** in either repo until the user approves.

## Verification (no pytest in this project → standalone python checks)

1. `python -c "import ast; ast.parse(open(f).read())"` on every moved/created `.py`.
2. With ONLY the new repo root on `sys.path` (no DenseTrack3Dv2):
   `python -c "from intent_model import IntentModel, IntentModelConfig, intent_loss"` resolves.
3. `grep -rn "densetrack3d\|preprocess\.\|worldmodel" intent_model/ data/ scripts/` → **empty**
   (no stale cross-repo imports).
4. Behavior parity: load the baseline checkpoint (once copied) and run `predict_trajectory`
   on one batch in BOTH repos; assert identical output → proves the vendored primitives
   (`nn_blocks`, `embeddings`, `losses`) behave byte-identically to the DenseTrack originals.
5. `import rclpy, torch` under py3.12 with Jazzy sourced → both import in one interpreter.
6. `DenseTrack3Dv2` regression: `python -c "import ast; ..."` + import check on
   `preprocess/viz_hand_cloud_live.py` still resolves against its vendored `hand_frame_transforms.py`.

## Non-goals

- No change to `object_flow.pkl` / `hand.pkl` contents or frame conventions.
- No git history transfer.
- No move of the mcap/stitching/smoothing pipeline, `gen_flow_labels.py`, or
  `object_flow_models_plan.md`.
- No functional change to the intent model — this is a relocation + import rewrite +
  vendoring, not a rewrite.
- No commits until approved.
