# Object-Flow Intent Model — Design Spec

**Date:** 2026-07-17
**Status:** design, pending review
**Context:** clean rewrite of the earlier `object_flow_models_plan.md` / `2026-07-13` design after
the wrist-pose work, the single-cloud + hand-history reformulation, and the decision to keep the v1
target simple (per-step delta + visibility). **This is the current spec.**

One model: an **action-conditioned forward-dynamics / intent model** that predicts future 3D object
flow from (a) a **single** point-cloud observation of the object, (b) the recent **hand-pose
history** (which supplies the current-motion state), and (c) a planned future **hand-motion**
sequence (the action). Supervised by labels DenseTrack3Dv2 already produces (`object_flow.pkl`).

At the present time `t`, with a history horizon `T_hist` and a prediction horizon `T_pred`:

```
x_{n, t:t+T_pred} = f( P_t,  q_{t-T_hist : t},  q_{t : t+T_pred} )
```

— one cloud `P_t`, hand-pose history `q_{t-T_hist:t}` (current state / velocity cue), future hand
plan `q_{t:t+T_pred}` (action); query points `x_{n,t} ⊂ P_t`; predict their flow over the horizon,
jointly (non-causal), the whole hand plan given up front.

**Why hand history rather than a cloud time series.** There is no realtime segmentation model fast
enough to produce an `F`-frame cloud series at inference — one segmentation per prediction is
feasible, a time series is not. So the object's **current velocity cannot come from geometry**; the
recent **hand-pose history** (available at full rate from the glove, no segmentation) carries that
cue instead. Consequence: current object velocity is an **inferred** quantity, not an observed one —
consistent with the forward-dynamics framing (hand motion drives object motion) and bounded by the
same contact assumption below, so no new risk.

---

## 1. Scope & the two teleop contexts

Shared-autonomy teleop; the operator always wears the Manus glove. There are two deployment
situations, and **only the second is this repo's job:**

- **In-hand (out of scope).** The object is physically in the robot's hand. Its pose is tracked in
  realtime by **classical CV pose estimation** and fed straight to the RL policy as a goal. No
  learned model — a prediction model there would only add fragility to a solved problem.
- **Air-grasp (this repo).** The operator pantomimes a hand motion with **no object in hand**; the
  robot holds the object elsewhere. The model reads the hand-motion plan and predicts the object
  flow that plan is *intended* to induce. That predicted flow drives the RL policy.

**Load-bearing assumption.** Every training clip is a real, in-contact human manipulation. At
air-grasp inference there is no contact. We assume **the same hand-motion plan implies the same
intended object motion**, contact or not — this is what makes the learned action→flow map reusable
at deploy. (This is LUCID's premise; stated here so it is a known risk, not a hidden one.)

**Embodiment gap and how we close it.** At air-grasp inference the object is robot-held; every
training clip is human-held. So the **state** must not depend on human/robot appearance — we use the
object **point cloud only** (geometry, no pixels, no hand). The **action** is human hand pose at both
train and test, so there is no gap on the action side.

---

## 2. Data — what the labels actually are

Established by inspecting real result folders (`results/test_manip_data_cube/`, etc.).

### 2.1 `object_flow.pkl` — the flow labels
`coords (Tclip,N,3)` float32 camera-frame metres, `vis (Tclip,N)` bool, `colors (N,3)`. Produced by
the windowed tracker (`preprocess/track_windowed.py`): N is a few thousand (~1900 in these clips),
points are re-seeded every `stride=5` frames, so only a few percent of tracks are visible in any
single frame.

**Sufficient as a label source (verified):** over a ~1 s horizon (32 native frames), a median of
**82** points per window stay continuously visible forward, and only ~1% of windows have <16
continuously-visible points. Sampling N=16 query points per window is easily met **by slicing the
existing pkl** — no re-seeding, no re-running the tracker. Point disappearance is genuine occlusion
/ out-of-view, handled by a per-step visibility flag rather than forcing every point to stay alive.

### 2.2 `hand.pkl` — the action source
`ergonomics (T,20)` joint angles (deg); `raw_node_pose (T,25,7)` node poses `[px,py,pz,qx,qy,qz,qw]`
in a fixed hand-**local** frame; `raw_node_names`/`raw_node_parent` (skeleton topology);
**`wrist_quat (T,4)`** = Manus `raw_sensor_orientation` (per-session absolute wrist orientation);
`retarget_values (T,20)` (robot DOF — the downstream controller's input, **not used here**);
`frame_timestamps_ns (T,)`; `side`.

Two facts that shape the model:
- `raw_node_pose` carries **no wrist / global-placement / palm-in-camera pose** — all 5 MCP bases
  have temporal std = 0. On its own it is informationally identical to `ergonomics` (finger
  articulation only), just as Cartesian keypoints.
- `wrist_quat` adds wrist **orientation** over time (not placement — there is still no camera↔Manus
  extrinsic and no wrist translation). Used as anchor-relative rotation
  `M_rel(τ) = R(q_t)ᵀ R(q_τ)` (anchor = present frame t, so history and future are both relative to
  now), it cancels the arbitrary per-session Manus world frame and gives "which way the wrist is
  turning" — the dominant cue for intended object rotation. Shared math lives in
  `preprocess/hand_frame_transforms.py`.

### 2.3 Alignment guarantee
`hand.pkl` rows and `object_flow.pkl` frames share one index grid (both aligned to the camera frames
by `extract_hand.py`), so at window frame `t`: flow row `t` ↔ hand row `t` ↔ `wrist_quat` row `t`.

---

## 3. Coordinate frame — camera frame, and why it is *preferred* not just required

The object-flow **target is in the camera (world) frame.** This is the primary representation for
three independent reasons:

1. **Transfers human→robot for free.** Both hand mounts are fixed; the camera keeps the same
   relative pose to the hand in collection and deployment; the human wrist is held (near-)static
   during collection. So camera-frame object flow means the same thing on both embodiments — no
   reframing at deploy.
2. **It is the steadier, more learnable signal.** To rotate an object a given way, the wrist
   unintentionally counter-moves to hold the object steady in the world. In camera frame the object
   shows a clean, steady tendency (exactly what the intent model should learn). In a
   wrist-rotation-cancelled frame that same steady object becomes a back-and-forth trajectory — it
   tracks `R_wrist(t)ᵀ` applied to a near-constant world position. Cancelling wrist rotation
   *re-injects* the operator's steadying motion as apparent object noise. **This is why wrist frame
   is worse here, not better.**
3. **Target frame and action-encoding frame are independent.** The stored `wrist_quat` is **not**
   used to reframe the target. It feeds the **hand-action encoder** (as `M_rel`) and drives the
   stabilized visualization only.

Wrist-rotation-cancelled flow (`wrist_frame_flow` in `hand_frame_transforms.py`, anchor 0) is
derivable and kept as a **deprioritized optional ablation**, not the target.

---

## 4. Problem statement & notation

- `t` — present time. `N` — query points (default **16**).
- **`T_pred`** — prediction horizon, fixed ~1 s; derives from rate: 8 Hz → 8; ~15 Hz → 16.
- **`pred_pad`** — optional extra steps predicted beyond `T_pred` (config, default 0). Only the first
  `T_pred` steps are the reported flow; the pad lets the model predict further (useful later for a
  centered endpoint velocity, §4.2). Total predicted length `L_pred = T_pred + pred_pad`.
- **`T_hist`** — history horizon, ~0.5 s of frames (e.g. 4 @ 8 Hz), shorter than `T_pred`. Supplies
  the current-motion cue via hand pose (see below).
- `P_t` — a **single** object **point cloud** at t (camera frame), `(P,3)`; the read-only state.
  **No cloud history** (§1: segmentation isn't realtime enough for a series).
- `x_{n,t} ∈ ℝ³` — query point n at t (camera xyz); a member of `P_t`.
- `q_τ` — human hand pose at step τ. History `q_{t-T_hist:t}` = current state; future `q_{t:t+T_pred}`
  = action. Each frame: articulation (`ergonomics` or `raw_node_pose`) ⊕ optional wrist orientation
  (`M_rel(τ)`); see §6.3 for the representation matrix.
- **Output:** per-step displacement `Δ_{n,τ} ∈ ℝ³`, so `x_{n,τ} = x_{n,t} + Σ Δ`; `(T_pred,N,3)`
  camera frame. Plus per-step visibility `v_{n,τ} ∈ (T_pred,N)`.

**Design principle:** decouple **spatial grounding** (from the single cloud at t) from **temporal
grounding** (hand-pose history for current velocity, future hand plan for the action). Predict
**deltas**, not absolutes.

### 4.1 The target (v1): per-step delta displacement + visibility
Keep v1 simple. The model predicts, for each query point and each predicted step:
- **per-step displacement** `Δ_{n,τ} ∈ ℝ³` (recover `x_{n,τ} = x_{n,t} + Σ Δ`), and
- **per-step visibility** `v_{n,τ} ∈ {0,1}`.

That's it — no goal-pose head, no endpoint-velocity head (deferred, §4.2). The dense per-step signal
is exactly what we want to train on anyway: we are ~100× more data-starved than LUCID (tens–hundreds
of clips total), so dense supervision (T_pred vectors per point, not one) is the data-efficient choice.

Target = **xyz displacements**, trained with **MSE on per-channel-standardized targets** (each of
Δx,Δy,Δz rescaled to unit variance from train-set stats, so metric depth can't dominate the
gradient). This matches LUCID. A `uvd` (pixel-track + depth) split was considered and rejected —
per-channel standardization already fixes the axis-scale problem without two heads or eval-time
backprojection; kept only as a documented fallback ablation if depth noise empirically dominates.

Sanity viz for inspecting the flow labels already exists: `preprocess/viz_goal_flow.py`
(dot-now + arrow-to-future) and `preprocess/viz_velocity_field.py` (projected 3D velocity).

### 4.2 Deferred to a future version: goal-pose + endpoint-velocity target
A more policy-friendly target is a **goal pose** (where each point ends up after ~1 s) plus **endpoint
velocity** (whether it is still moving, and which way). Sketch for later, **not built in v1:**
- goal pose `xT_n = x_{n,t} + Σ_τ Δ_{n,τ}`, endpoint velocity `vT_n` = a **smoothed** central
  difference at the endpoint over `±dt` (~1/3 s), not the noisy one-sample last delta;
- centering needs frames past the endpoint → predict beyond `T_pred` (the v1 `pred_pad` config, §5,
  already supports predicting extra steps) and read `vT` off the model's own trajectory;
- full trajectory recoverable from `{x_t, v_t, xT, vT}` by **cubic Hermite** for eval;
- loss variants then become a switch (dense / dense+endpoint-velocity / endpoint+velocity only).
`viz_goal_flow.py` (goal-survival fraction) and `viz_velocity_field.py --dt 10` are the sanity tools
for this target when it is built.

---

## 5. Label generation — hybrid offline precompute + online windowing

**Dataset layout (on disk).** `inhand_manipulation/<clip>/episode_<k>/` — each episode is one
manipulation with `color.mp4`, `depth.mkv`, `hand.pkl`, `intrinsics.txt`, `object_flow.pkl`, and
`seg/mask/{f:05d}.png` (+`.json`). **Masks are dense — one per frame, not just tracker keyframes**
(verified: episode_1 has 1138 frames and 1138 mask PNGs). `object_flow.pkl`: `coords (Tclip,N,3)`
float32 camera-frame metres, `vis (Tclip,N)`, `colors (N,3)`; N is a few thousand (~1900 in these
clips).

**Every frame can be a sample.** Because masks are dense, any frame `t` is a valid window
present-frame — no restriction to a seg-mask stride grid. This multiplies usable windows ~`stride`×
over the keyframe-only assumption.

**Why hybrid (not fully offline, not fully online).** Rough scale: ~120 episodes × ~40 s × 30 fps ≈
1200 frames/episode. The `object_flow.pkl` labels are tens of MB/episode → a few GB total, and they
(plus `color.mp4` / `depth.mkv` / `hand.pkl`) already exist on disk, so raw storage is not the
concern. The two things to avoid:
- **Fully-offline materialized window cache** — adjacent windows overlap by `T_pred−stride_win`
  frames, so every target frame is re-stored ~`T_pred/stride_win`× *and* the cache bakes in N /
  stride / sampling choices we want to sweep. Wasteful and rigid.
- **Fully-online backprojection** — building the object cloud (decode `depth.mkv` → apply mask →
  back-project → subsample) re-run every epoch would re-decode video repeatedly. Slow.

**The split:**
- **Offline, once per episode (`gen_flow_labels.py`):** precompute only the expensive part — the
  object point cloud at **every** frame → a compact per-episode artifact `clouds.npz`
  (`(Tclip, P, 3)`; e.g. 1138 frames × 512 × 3 × 4 B ≈ **7 MB/episode → ~1 GB total**). Everything
  else stays as the existing `object_flow.pkl` / `hand.pkl`.
- **Online, in the `Dataset` (`flow_window_dataset.py`):** slice each window on the fly — cheap array
  indexing into `coords`/`vis` (flow window) and `hand.pkl` (hand window), plus loading the one
  precomputed cloud. Rate downsample, query sampling, `M_rel` derivation, and augmentation all happen
  here, so they are free to vary without regenerating anything.

### 5.1 Offline precompute (per episode)
For **every** frame `t` (masks are dense): back-project the **object-masked** depth at `t` to
camera-frame 3D via `intrinsics.txt` (mask: `seg/mask/{t:05d}.png`), subsample to fixed `P` (e.g.
512) by FPS or random → row `t` of `clouds.npz`. **One cloud per frame — no series.** Frames whose
mask is empty (object fully occluded) get an all-NaN row and are skipped as present-frames.

### 5.2 Online window sampling (per training item)
At present frame `t` (any frame with a non-empty cloud), rate `stride_hz`:
1. **Rate grid.** 4 → 8 Hz → T_pred=8; 2 → ~15 Hz → T_pred=16. `T_pred` ~1 s, `T_hist` ~0.5 s;
   optionally predict `pred_pad` steps beyond `T_pred` (§4).
2. **Cloud `P_t`.** Load the precomputed cloud for `t` → `(P,3)`.
3. **Query points (concat into the cloud, not a subset).** Among tracks visible at `t`, sample N
   (=16); their positions at `t` are `x_{n,t}` (from `coords`). We do **not** require `x_{n,t} ∈ P_t`
   — the query points and the cloud are sampled two different ways (tracker uv+depth lift vs.
   back-projected masked depth) from the same surface, so exact membership isn't guaranteed and isn't
   needed. Instead **concatenate the N query points into the cloud** → `(P+N, 3)`, guaranteeing every
   query point exists as a scene token for cross-attention. Prefer points with the most future
   coverage but do **not** require full-horizon continuity — the vis head handles occlusion,
   invisible steps are masked in the loss. Drop the rare window that can't supply N points visible
   at `t`.
4. **Targets.** `x_{n, t : t+L_pred}` (from `coords`) and visibility (from `vis`), `L_pred =
   T_pred+pred_pad`; store absolutes, the loss consumes deltas. All target frames exist at label time.
5. **Hand pose (history + action).** From `hand.pkl`, take the downsampled frames spanning
   `t-T_hist : t+L_pred`: articulation (`ergonomics` or `raw_node_pose`, config) and `wrist_quat`.
   Derive `M_rel(τ) = R(q_t)ᵀR(q_τ)` (anchor = present frame `t`; `wrist_M_rel` in
   `hand_frame_transforms.py`) and assemble the chosen representation (§6.3). Split into
   `q_hist (T_hist,·)` and `q_future (L_pred,·)`.

Item dict returned by the Dataset:
```
{ cloud (P+N, 3) xyz,                        # present-frame object cloud with query points concatenated
  x0 (N,3),                                  # query seeds x_{n,t} (also the last N rows of cloud)
  target (L_pred, N, 3) xyz,  target_vis (L_pred, N),
  q_hist (T_hist, ·),  q_future (L_pred, ·), # articulation ⊕ M_rel
  K (3,3),  frame_meta }
```
(Cache uses `x0` for the notation's `x_{n,t}`.) Every frame is a candidate present-frame; windows
stride the episode every `stride_win` frames (default 1 — every frame a sample).

**Verification:** visualize a few windows with `viz_goal_flow.py` / `viz_velocity_field.py`; confirm
shapes, that the concatenated query points overlay the object cloud, and ≥N tracks visible at `t`.

---

## 6. Model architecture

Shared backbone: this repo's `EfficientUpdateFormer`
(`densetrack3d/models/densetrack3d/update_transformer.py`) with its `(b,t,n,c)` factorized
time/space + virtual-track attention.

### 6.1 Object-cloud state encoder (the divergence from LUCID)
LUCID uses a full RGB-D DINOv3 encoder (hand pixels included) and defeats the human→robot gap with
aggressive human-region pixel augmentation. We reach the same goal **structurally** by excluding
appearance entirely:

- `PointNet++` (SA layers) over the **single** cloud `P_t` (with the N query points concatenated in,
  §5.2) → **scene point tokens** `S ∈ (B,M,C)`. PointNet++ over point-transformer as the default for
  speed on a small dataset; encoder is a config switch. (No cloud history — the current-motion cue
  comes from the hand-pose history, §1/§6.3, not
  from geometry.)
- Read-only; never predicted. Geometry only — no appearance, no hand.
- **Occlusion/point augmentation** at train time (dropout, patch removal, jitter) so the model
  doesn't depend on *which* region is visible (human vs robot hands occlude differently). This is
  our analog of LUCID's human-region augmentation, and the main state-side gap mitigation.

### 6.2 Object (query) tokens
Token grid `(B, L_pred, N, C)` (`L_pred = T_pred + pred_pad`). Token `(τ,n)` init:
```
tok[τ,n] = proj(sinusoid(x_{n,t}))   # (C,) spatial seed, same for all τ
         + step_emb(τ)               # (C,) which future step
         + ctx                       # (C,) hand-history context, same for all (τ,n)
```
`ctx (B,C)` is the pooled hand-history embedding from §6.3 — this is *how the history's current-motion
cue enters*: every object token is seeded with what the hand was just doing. Optional + track feature.
(LUCID inits object tokens from the projected query point the same way; `ctx` is our addition for the
single-cloud / inferred-velocity design.)

### 6.3 Hand-action encoder — concrete shapes

**One sequential encoder over the whole hand sequence.** Concatenate history and future along time:
`q_seq = [q_hist ; q_future]`, length `L = T_hist + L_pred`. Each frame is a raw vector `d_q` (e.g.
`ergonomics` 20 + `M_rel` as 6D = 26). Then:
```
e   = Linear(d_q -> C)(q_seq)        # (B, L, C)  per-frame embed
seq = TemporalEncoder(e)             # (B, L, C)  GRU or small transformer over L
h_hist   = seq[:, :T_hist]           # (B, T_hist, C)
a_future = seq[:, T_hist:]           # (B, L_pred, C)   <- per predicted step
ctx      = pool(h_hist)              # (B, C)   GRU last hidden, or mean over T_hist
```
Because the encoder is **sequential**, each future step in `a_future` has already attended over the
history — so history informs the action without a separate fusion module. `ctx` is the pooled
current-state summary consumed by the query-token init (§6.2). `a_future` is the per-step action
signal injected into the backbone (below). For `raw_node_pose` the per-frame vector is the flattened
24 keypoints (drop node0) `⊕ M_rel`; the "25 tokens" option (below) expands the keypoint axis instead.

**Injecting `a_future` into the backbone (both):**
- **FiLM / additive** — broadcast over the N query tokens, add before temporal attn:
  `tok = tok + a_future[:, :, None, :]` → stays `(B, L_pred, N, C)`.
- **Token concat** — append hand token(s) on the token axis:
  `hand_tok = a_future[:, :, None, :]` is `(B, L_pred, K, C)` with **K=1** (pooled per-step embedding);
  `tok = cat([tok, hand_tok], dim=2)` → `(B, L_pred, N+K, C)`. Spatial self-attn then runs over N+K.
  With the `raw_node_pose` "25 tokens" variant, **K=25** (one token per keypoint) so spatial attn can
  in principle weight individual fingers. Heads read only the object tokens `tok[..., :N, :]`.

**Representation config switch** — the per-frame `d_q` varies along two independent axes:

- **What info:** articulation only, vs. articulation + wrist orientation.
- **What frame:** hand-local / wrist frame, vs. camera frame.

The organizing fact: **articulation (joint angles / hand-local keypoints) is frame-invariant, so
wrist orientation is the *only* frame-dependent piece of hand information.** Hence the consistency
rule — *the action's frame should match the target's frame, and wrist orientation is skipped only
when the target is the wrist frame* (there the flow already carries it):

| Target frame | Action representation | Wrist orientation |
|---|---|---|
| **camera** (default) | `ergonomics` **or** hand-local `raw_node_pose`, ⊕ `M_rel(τ)` | explicit (6D or 3×3) |
| **camera** | `raw_node_pose` **transformed into camera frame** | baked into coordinates |
| **wrist** (ablation) | `raw_node_pose` hand-local, **no** `M_rel` | not needed (in the flow) |
| either | raw `ergonomics` / hand-local raw, **no** wrist term | **omitted** — weak baseline |

Notes:
- The two camera-frame rows carry the **same information** — one keeps wrist orientation as a
  separate feature, the other bakes it into the keypoint coordinates (that transform *uses* `M_rel`
  via the fixed hand-local→camera placement; it is not a no-`M_rel` option).
- The **no-wrist** row is a deliberately weak baseline for a camera-frame target: it drops the
  wrist-rotation cue, which (§3) is the *dominant* driver of intended object rotation; articulation
  alone only captures finger-gaiting reorientation. Keep it to quantify how much the wrist cue buys.
- Default = `ergonomics` ⊕ `M_rel` (compact, frame-consistent with the camera-frame target),
  giving `d_q = 26`.

**No metric hand↔object coupling by 3D proximity.** No camera-frame hand translation exists, and in
air-grasp the hand isn't even co-located with the object — so object tokens can't attend to
fingertips by distance. The model learns articulation + wrist-rotation → object motion abstractly.
(This is why the hand token is injected as an abstract per-step signal, not placed geometrically.)

### 6.4 Backbone blocks (per layer)
1. **Scene cross-attention** — query/hand tokens attend into scene point tokens `S` (spatial
   grounding at t; replaces the 4D-correlation lookup). `CrossAttnBlock`, `blocks.py`.
2. **Spatial self-attention** — across N (+ hand) tokens within a step (rigidity/coherence; keep the
   virtual-track trick for larger N).
3. **Temporal self-attention** — across the `L_pred` predicted steps per token, **non-causal**
   (whole horizon given up front); receding-horizon re-runs for closed loop.

**Dropped from the tracking path:** `Corr4DCNN`, `get_*corr*`, sliding-window chaining — no
correlation against future frames.

### 6.5 Heads & losses
- **Object head** `Linear(C→3)` per step → `Δ_{n,τ}`; recover `x_{n,τ} = x_{n,t} + Σ Δ`.
- **Visibility head** `Linear(C→1)` + `balanced_bce_loss` (our addition over LUCID; needed because
  windowed labels have genuine occlusion).
- **Loss** = MSE(Δ, per-channel-standardized xyz) + `w_vis · BCE`, masked by target visibility,
  averaged over `L_pred`. (Huber is a drop-in robustness swap. Goal-pose / endpoint-velocity loss
  terms are §4.2, deferred.)
- AdamW, cosine 1e-4→1e-6, 5 warmup epochs, wd 0.01, grad-clip 1.0, EMA 0.999 for eval.
- **Model size — start small.** LUCID: hidden 768 / depth 12 / 12 heads (~85M) on 20k clips/task. We
  have ~100× less data → **start hidden 384 / depth 6**, scale toward 768/12 only if underfitting
  persists with augmentation. Batch as memory allows.

**No palm-pose output head.** LUCID regresses future palm SE(3) because in full autonomy no human is
present at deploy. Our framing removes the need (the hand is a full-horizon *input*), and we have no
camera-frame palm pose anyway. Dropped for v1; could return as an auxiliary task later.

---

## 7. Repo mapping

| Piece | Action | Location |
|---|---|---|
| `EfficientUpdateFormer`, `CrossAttnBlock`, `AttnBlock`, `Mlp` | keep/reuse | `models/densetrack3d/{update_transformer,blocks}.py` |
| Sinusoidal embeds | reuse | `models/embeddings.py` |
| Losses (`huber_loss`, `balanced_bce_loss`, `track_loss`) | reuse | `models/loss.py` |
| `Corr4DCNN`, `get_*corr*`, sliding-window chaining | drop | `densetrack3dv2.py`, `corr4d_blocks.py` |
| RGB-D image scene encoder (old plan) | drop / replaced | — |
| PointNet++ object-cloud encoder | **add** | new `models/worldmodel/point_encoder.py` |
| Scene(point) cross-attn backbone + FiLM | add | new `models/worldmodel/backbone.py` |
| Hand encoder (articulation ⊕ `M_rel`) | add | new `models/worldmodel/hand_encoder.py` |
| object (xyz-disp) + vis heads | add (thin) | new `models/worldmodel/heads.py` |
| Shared frame transforms (`M_rel`, cloud→P, etc.) | reuse | `preprocess/hand_frame_transforms.py` |

New files:
```
densetrack3d/models/worldmodel/
  __init__.py  point_encoder.py  hand_encoder.py  backbone.py  heads.py
  intent_model.py     # full assembly (action-conditioned)
scripts/
  gen_flow_labels.py  train_intent.py
data/
  flow_window_dataset.py
```

Forward sketch (`backbone.py`):
```
# --- state + hand (§6.1, §6.3) ---
S       = point_encoder(cloud)                  # (B, M, C)   cloud=(P+N) query pts concatenated, read-only xyz
seq     = hand_encoder([q_hist ; q_future])     # (B, T_hist+L_pred, C)  one sequential pass
h_hist  = seq[:, :T_hist]                        # (B, T_hist, C)
a_fut   = seq[:, T_hist:]                        # (B, L_pred, C)  per predicted step
ctx     = pool(h_hist)                           # (B, C)  history state / velocity cue

# --- object tokens (§6.2) ---
tok = init_query_tokens(x0, L_pred, ctx)         # (B, L_pred, N, C)  = sinusoid(x0)+step_emb+ctx
tok = tok + a_fut[:, :, None, :]                 # FiLM/additive inject (broadcast over N)
tok = cat([tok, a_fut[:, :, None, :]], dim=2)    # + hand token(s): (B, L_pred, N+K, C), K=1 (or 25)

# --- backbone (§6.4) ---
for blk in blocks:
    tok = cross_attn(tok, S)                     # ground against the single object cloud
    tok = spatial_self_attn(tok)                 # within-step, over N+K
    tok = temporal_self_attn(tok)                # across-step, non-causal

# --- heads (§6.5); read only object tokens ---
delta  = obj_head(tok[..., :N, :])               # (B, L_pred, N, 3)
vis    = vis_head(tok[..., :N, :])               # (B, L_pred, N)
x_pred = x0[:, None] + delta.cumsum(1)           # camera-frame xyz; report [:, :T_pred]
```

---

## 8. One model across objects
Because the "state" is the object **point cloud** (geometry observed as an *input*, not baked into
weights), one model can in principle generalize across objects — unlike LUCID's per-task models.
**Plan for one multi-object model** (pool all clips); fall back to per-object only if generalization
is poor.

---

## 9. Build order

The hand stream is built in from the start (it's the deliverable); the no-hand baseline is an
ablation at the end.

1. **Offline `gen_flow_labels.py`** — precompute per-episode `clouds.npz` (object cloud at every
   frame, §5.1). **`flow_window_dataset.py`** — online windowing (§5.2): `cloud` (+concatenated query
   points), x0, dense target + vis, `q_hist`/`q_future` articulation ⊕ `M_rel`, K. Verify a few
   windows with `viz_goal_flow.py` / `viz_velocity_field.py`. (No tracker re-run.)
2. **Model skeleton** — point encoder, query tokens (+ `ctx`), sequential hand encoder
   (`ergonomics`/`raw_node_pose` ⊕ `M_rel`) + FiLM/hand-token, `EfficientUpdateFormer` + scene
   cross-attn, per-step disp + vis heads. **Overfit one clip** → sanity.
3. **Train at scale** — eval dense 3D ADE/FDE + visibility. Add mixture/diffusion head only if
   mode-averaging (blurry mean motion) shows up.
4. **Ablations** — no-hand baseline; `ergonomics` vs `raw_node_pose`; 8 Hz vs 15 Hz; camera-frame
   vs wrist-frame target (expect camera-frame steadier, §3); FiLM vs hand-token injection.
5. **Future (§4.2)** — goal-pose + endpoint-velocity target and its loss variants, once v1 works.

---

## 10. Non-goals (v1)
- No in-hand model (CV pose-tracking handles that context).
- No image/RGB or hand-pixel observation (deliberate, for transfer).
- No palm-pose output head (§6.5).
- No metric wrist→object placement (no data; not meaningful for air-grasp).
- No re-running the tracker for labels (existing pkl suffices).
- No robot-DOF (`retarget_values`) conditioning (downstream controller's input).
- No closed-loop rollout harness yet (receding-horizon is a later concern).

## 11. Downstream interface note (design around, not build)
LUCID's intent runs at 8 Hz over 1 s while the RL policy consumes that `T=8` lookahead at 50 Hz via
a **segment-progress scalar** interpolating between reference steps. When we build the control side,
replicate this (reference steps + progress scalar in the observation/reward) rather than forcing the
model to predict at the control rate. Parked so rate/T stay decoupled from the controller.

## 12. Open parameters to sweep (not blockers)
`N` (16→64), `P` cloud size, `T_hist` (~0.5 s), `pred_pad`, point encoder (PointNet++ vs
point-transformer), hand `TemporalEncoder` (GRU vs transformer) + `ctx` pooling, backbone
hidden/depth, rate (8/15 Hz), articulation encoding, `M_rel` as 6D vs 3×3, hand-token K (1 vs 25),
FiLM-vs-token injection, MSE-vs-Huber, one-model-vs-per-object.
