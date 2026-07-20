# Object-Flow World Models — Design Plan

Two models that predict **future 3D object flow** (per-point 3D trajectories), trained on
labels produced by **DenseTrack3Dv2** (this repo, used unchanged as an offline label engine):

- **Model A — Intent model** (unconditioned): predict object flow from *past observation only*.
  The model must *imagine* plausible object motion. (LUCID-style, arXiv:2606.11628.)
- **Model B — Forward dynamics model** (action-conditioned): predict object flow *given a
  planned hand-pose sequence* over the prediction horizon. Better-posed: the action removes
  most of the future's ambiguity. This is `s_{t+1..t+T} = f(s_t, a_{t:t+T})`.

Both share one backbone (`EfficientUpdateFormer`, already in this repo) and one label pipeline.
Model B = Model A + a hand-pose conditioning stream. Build A first; B is an additive delta.

---

## 0. Notation & problem statement

- `t`            : current time (the "present").
- `F`            : number of past RGB-D frames used as context (default **2**).
- `T`            : number of future steps predicted (default **8**, ~1 s horizon).
- `N`            : number of object query points (default **16–64**).
- `I_{≤t}`       : RGB-D history, shape `(F, 4, H, W)` (RGB + depth), resized to e.g. 256×256.
- `x_{n,0} ∈ ℝ³` : 3D position (camera frame) of query point `n` at time `t`.
- `q_τ`          : hand pose at future step `τ` (Model B only). Either
                   joint angles `(20,)` or FK keypoints `(21, 3)`. Full plan: `(T, 20)` or `(T, 21, 3)`.
- Output         : `{ x_{n,τ} ∈ ℝ³ }` for `n=1..N`, `τ=1..T`, i.e. `(T, N, 3)` in **camera frame**.
  Optional per-point visibility `(T, N)`.

**Design principle (the crux):** decouple *spatial grounding* from *temporal grounding*.
- Spatial grounding comes only from the RGB-D history at `t` → a **read-only scene memory**.
- Temporal grounding of each future step comes from the **step index** (Model A) or the
  **time-aligned hand pose `q_τ`** (Model B) — *not* from a future image.
- Object tokens are seeded in space by `t` and carried through time by attention over steps
  (and, in B, by the action).

Both models predict **per-step displacements** `Δ_{n,τ}` and recover `x_{n,τ} = x_{n,0} + Σ` (or
`x_{n,0} + Δ_{n,τ}` per step); predicting deltas is far better-conditioned than absolutes.

---

## 1. Shared component: label generation with DenseTrack3Dv2

DenseTrack3Dv2 is used **as-is, offline** (no forward-pass changes) to produce ground-truth
future 3D tracks. This mirrors LUCID's supervision pipeline (their Fig. 8).

Per training clip:
1. **Depth**: metric depth per frame (UniDepth already wired in `demo.py`; or ViPE/ZED depth).
2. **Object mask** at the query frame (SAM/Grounded-SAM-2 in `submodules/`, or manual).
   Restricts query points to the object (mask plumbing already exists: `--mask_path` in
   `demo.py`/`demo_sparse.py`, `segm_mask` in `predictor.py`).
3. **3D tracks**: run `DensePredictor3D` / `Predictor3D` on the *whole* clip (past + future
   frames — at label time we have all frames) to get `trajs_3d_dict`:
   - `coords (T_full, N, 3)`, `vis (T_full, N)`, `colors (N, 3)` in camera frame.
   See `convert_trajs_uvd_to_trajs_3d` in `densetrack3d/models/model_utils.py`.
4. **Windowing**: slice into training samples `(I_{≤t}, x_{·,0}, [q_{0:T}], target x_{·,1:T})`.
   The query points `x_{·,0}` are the tracks' positions at the window's present frame `t`;
   targets are their positions at `t+1..t+T`.
5. **(Model B only) Hand pose**: extract `q_τ` per frame (WiLoR / MediaPipe / dataset labels).
   If joint angles → also precompute FK keypoints. Store both; decide representation at train time.

Output: a cached dataset of tensors (npz/pkl per window). **The tracker never runs at
inference time for either model** — it only makes labels.

> Camera-frame note: object flow is defined in **camera frame** (user requirement). If the
> capture camera moves, camera-frame flow entangles object motion with camera motion. Two
> options: (a) accept it (fine for static-camera captures), or (b) store camera extrinsics per
> frame and additionally supervise/evaluate in a world frame. Keep extrinsics in the cache
> even if unused at first — cheap insurance. (See §6.)

---

## 2. Shared component: backbone & I/O

### 2.1 Scene encoder (spatial memory, both models)

- **Frozen DINOv3 ViT-B/16** (LUCID's choice) OR reuse this repo's `BasicEncoder` (`blocks.py`)
  on the `F` history frames to avoid an external dependency. Start with `BasicEncoder` for a
  fast first iteration; swap to DINOv3 if semantic grounding is weak.
- **Depth fusion**: zero-initialized residual adapter (Conv2d → LayerNorm → Linear) that adds
  depth features into the patch-token space. Zero-init ⇒ training starts as RGB-only, depth
  learned in.
- Output: scene patch tokens `S ∈ (B, M, C)` where `M = F · (H/16)·(W/16)`, `C = hidden`.
  Read-only; never predicted.

### 2.2 Object tokens (both models)

- Token grid `(B, T, N, C)`. Token `(τ, n)` initialized as:
  `proj(sinusoid(x_{n,0}))  +  step_emb(τ)`  (same spatial seed replicated across all `τ`).
- Optional: add the point's color / initial track feature (available from the label pipeline).

### 2.3 Backbone: `EfficientUpdateFormer` (kept from this repo)

`densetrack3d/models/densetrack3d/update_transformer.py`. Its `(b, t, n, c)` token layout with
factorized attention and virtual tracks is exactly right. Per block, run **three** attentions:

1. **Scene cross-attention** — object/hand tokens (query) attend into frozen scene tokens `S`.
   *This replaces CoTracker/DenseTrack's 4D correlation lookup.* Use a `CrossAttnBlock`
   (`blocks.py`) per layer. Gives spatial grounding at `t`.
2. **Spatial self-attention** — across the `N` (+ hand, in B) tokens *within* a step. Models
   object rigidity/coherence. Keep the **virtual-track** trick for large `N` (already in
   `EfficientUpdateFormer`, `add_space_attn=True`).
3. **Temporal self-attention** — across the `T` steps for each token. Where dynamics propagate.

**Temporal attention is NON-causal / bidirectional.** We provide the whole horizon up front
(Model A: all `T` query steps; Model B: the full action plan `q_{0:T}`), so predicting the
whole rollout jointly is simpler and more accurate than autoregression. For closed-loop use,
re-run in receding-horizon fashion rather than making attention causal.

**Discard** from the tracking path: `fnet` correlation usage is replaced by scene cross-attn;
`Corr4DCNN` / `get_4dcorr_features` / `get_single_corr_depth` / sliding-window chaining in
`densetrack3dv2.py` are **not used** (we don't correlate against future frames).

### 2.4 Heads & losses (both models)

- Per-step **displacement head**: `Linear(C → 3)` → `Δ_{n,τ}`; recover `x_{n,τ}`.
- Optional **visibility head**: `Linear(C → 1)` + BCE (`balanced_bce_loss` in `loss.py`).
- **Loss**: MSE (or Huber, `huber_loss` in `loss.py`) on 3D positions, averaged over `T`.
  Per-channel standardize targets using train-set stats. Optional iteration-weighting if you
  keep the refine-loop (`track_loss` already supports `weight_offset`).
- AdamW, cosine LR 1e-4→1e-6, warmup, grad-clip 1.0, EMA(0.999) weights for eval (LUCID Table 2).

---

## 3. Model A — Intent model (unconditioned)

**Signature**
```
f_θ( I_{≤t}, {x_{n,0}}_{n=1..N} )  →  {x_{n,τ}}_{n=1..N, τ=1..T}
```

**Structure** = §2 exactly, with no hand stream:
- Scene memory from `I_{≤t}` (§2.1).
- Object tokens seeded from `x_{n,0}` + step embedding (§2.2).
- `EfficientUpdateFormer` with scene cross-attn / spatial / temporal blocks (§2.3).
- Displacement heads (§2.4).

**What it learns**: the *prior* over how this object tends to move given the scene — i.e.,
"intent". Ambiguous by nature; expect multimodality. If mode-averaging (blurry mean motion)
is a problem, later upgrade the head to a small mixture / diffusion / VAE over trajectories.
Not needed for v1.

**Use `F=2`, not 1**: with a single frame the object's initial velocity is unobservable and
static vs. moving objects are indistinguishable. Two frames give a motion cue.

---

## 4. Model B — Forward dynamics model (action-conditioned)

**Signature**
```
g_θ( I_{≤t}, {x_{n,0}}_{n=1..N}, {q_τ}_{τ=0..T} )  →  {x_{n,τ}}_{n=1..N, τ=1..T}
```

= Model A **plus** a hand-pose conditioning stream. This is the only structural delta.

### 4.1 Hand encoder

Support both representations behind one interface (user will try both):

- **Joint angles**: `q_τ ∈ (20,)` → `MLP(20 → C)`.
- **FK keypoints**: `q_τ ∈ (21, 3)` → flatten `(63,)` → `MLP` **OR** treat 21 keypoints as
  21 tokens with sinusoidal 3D position encoding (richer; lets object points attend to
  individual fingertips).

**Recommendation**: prefer **FK keypoints in 3D**. They live in the *same metric space* as the
object points, so hand↔object spatial attention becomes geometrically meaningful (token
distance ≈ contact proximity) instead of the net having to learn kinematics. Joint angles are
a fallback / ablation.

Produce a per-step hand embedding `a_τ ∈ (B, T, C)` (or per-step hand token set).

### 4.2 Injecting the action (do both)

1. **As token(s)** at each step `τ`: append the hand embedding as an extra token alongside the
   `N` object tokens at that step, so **spatial self-attention** (§2.3-2) models contact/coupling
   geometry (which fingers near which points). With FK keypoints as 21 tokens, this is direct.
2. **FiLM / additive** conditioning: add (or FiLM-modulate) `a_τ` into every object token at
   step `τ` *before temporal attention*, so the action strongly gates per-step dynamics.

### 4.3 Coordinate frame for the hand (the accuracy caveat)

Object flow is in **camera frame**. Hand keypoints come in hand/wrist frame and need a
hand→camera transform that is **estimated and noisy** (user's constraint). Options, in order of
robustness to that noise:

- **B0 (most robust)**: feed hand as **joint angles** (frame-invariant) *plus* a single noisy
  wrist-in-camera pose token. The network fuses the reliable articulation with the unreliable
  global placement, and can down-weight the latter. Least sensitive to bad extrinsics.
- **B1**: feed **FK keypoints transformed into camera frame**. Cleanest geometric coupling, but
  directly injects the transform's error into every keypoint. Good if the transform is decent.
- **B2**: feed FK keypoints in **wrist-relative frame** + wrist pose token separately. Decouples
  articulation (accurate) from global placement (noisy) — a middle ground.

Make the frame/representation a **config switch** and ablate B0/B1/B2. Expect B0 or B2 to win
when the transform is poor; B1 when it's good.

### 4.4 Why B should outperform A

The action collapses the future's multimodality: given the actual hand plan, object motion is
largely determined (contact + push/grasp). Same backbone, much sharper targets. B is the
"world model / dynamics" you can roll out for planning; A is the "what will probably happen"
prior.

---

## 5. Repo mapping — keep / add / replace

| Piece | Action | Location |
|---|---|---|
| `EfficientUpdateFormer` (time/space/virtual-track) | **Keep** | `models/densetrack3d/update_transformer.py` |
| `CrossAttnBlock`, `AttnBlock`, `Mlp`, `Attention` | **Keep / reuse** | `models/densetrack3d/blocks.py` |
| `BasicEncoder` (history encoder, v1) | **Reuse** | `models/densetrack3d/blocks.py` |
| Sinusoidal embeds (`get_1d/2d_sincos...`) | **Reuse** | `models/embeddings.py` |
| Losses (`huber_loss`, `track_loss`, `balanced_bce_loss`) | **Reuse** | `models/loss.py` |
| `Corr4DCNN`, `get_4dcorr_features`, `get_single_corr_depth` | **Drop** (no future-frame correlation) | `densetrack3dv2.py`, `corr4d_blocks.py` |
| Sliding-window chaining in `DenseTrack3DV2.forward` | **Drop** | `densetrack3dv2.py` |
| DINOv3 encoder + depth adapter | **Add** (optional, phase 2) | new `models/worldmodel/encoder.py` |
| Scene cross-attention wiring | **Add** | new `models/worldmodel/backbone.py` |
| Hand encoder + FiLM/token injection (Model B) | **Add** | new `models/worldmodel/hand_encoder.py` |
| Displacement / vis heads | **Add** (thin) | new `models/worldmodel/heads.py` |
| DenseTrack3D label generator (offline) | **Reuse as-is** | `demo.py` / `predictor.py` |

### Proposed new files
```
densetrack3d/models/worldmodel/
  __init__.py
  encoder.py        # history RGB-D -> scene patch tokens (BasicEncoder v1, DINOv3 v2)
  hand_encoder.py   # (T,20) or (T,21,3) -> per-step embedding/tokens (Model B)
  backbone.py       # EfficientUpdateFormer + scene cross-attn + (optional) action FiLM
  heads.py          # per-step displacement (+vis) heads
  intent_model.py   # Model A assembly
  dynamics_model.py # Model B assembly (A + hand stream)
scripts/
  gen_flow_labels.py   # run DenseTrack3D over clips -> cached (I,x0,[q],target) windows
  train_intent.py      # train Model A
  train_dynamics.py    # train Model B
data/
  flow_window_dataset.py  # loads cached windows; returns dict of tensors
```

### Forward-pass sketch (both, `backbone.py`)
```
scene = encoder(I_hist)                     # (B, M, C)   read-only
tok   = init_object_tokens(x0, T)           # (B, T, N, C)
if dynamics:
    a   = hand_encoder(q)                   # (B, T, C) and/or hand tokens
    tok = tok + film(a)[:, :, None, :]      # FiLM inject
    tok = concat_hand_tokens(tok, a)        # optional per-step hand token(s)
for blk in blocks:
    tok = cross_attn(tok, scene)            # spatial grounding @ t
    tok = spatial_self_attn(tok)            # within-step (virtual tracks)
    tok = temporal_self_attn(tok)           # across-step (non-causal)
delta = disp_head(tok[..., :N, :])          # (B, T, N, 3)
x_pred = x0[:, None] + delta                # (B, T, N, 3), camera frame
```

---

## 6. Coordinate-frame decision (applies to both)

- **Targets**: camera frame (fixed requirement).
- **Static camera**: camera-frame flow == object motion. Proceed directly.
- **Moving camera**: store per-frame extrinsics in the label cache. Either (a) keep supervising
  camera-frame flow and accept camera-motion entanglement, or (b) additionally lift to a world
  frame for a cleaner dynamics signal and transform back for eval. Decision can be deferred, but
  **cache extrinsics now**.
- **Hand frame (Model B only)**: config switch B0/B1/B2 (§4.3); ablate against the noisy
  hand→camera transform.

---

## 7. Defaults (from LUCID Tables 1–2, adjust to data)

- Input 256×256 RGB-D; `F=2`; `N=16` (raise to 64 for denser flow); `T=8` (~1 s).
- Backbone hidden 768, depth 12, heads 12 (or start smaller: hidden 384/depth 6 as in this
  repo's config, scale up if underfitting).
- Object head `Linear(C→3)` per step; (B) hand joint MLP or 21-keypoint tokens.
- AdamW, lr 1e-4 (cosine → 1e-6), warmup 5 ep, wd 0.01, grad-clip 1.0, EMA 0.999, ~100 ep,
  batch 16. Per-channel standardized targets. Heavy augmentation of RGB-D (and, if human in
  frame, of human pixels) so the model reads object motion, not demonstrator appearance.

---

## 8. Build order (milestones)

1. **Label pipeline** (`gen_flow_labels.py`): DenseTrack3D + mask → cached windows with
   `(I_hist, x0, target, [q, extrinsics])`. Verify by visualizing tracks (viser tooling exists).
2. **Dataset + Model A skeleton**: `BasicEncoder` scene memory, object tokens,
   `EfficientUpdateFormer` + scene cross-attn, displacement head. Overfit one clip → sanity.
3. **Train Model A** at scale; eval 3D endpoint error / ADE-FDE; add vis head if needed.
4. **Model B**: add hand encoder + FiLM + hand token; ablate joint-angle vs FK-keypoint and
   frame B0/B1/B2. Compare B vs A (B should be sharper, lower error).
5. **(Optional) upgrades**: DINOv3 encoder swap; mixture/diffusion head for A's multimodality;
   world-frame supervision for moving-camera data; receding-horizon closed-loop rollout for B.

---

## 9. Key references

- **LUCID** (arXiv:2606.11628): intent model adapts CoTracker3's `EfficientUpdateFormer`,
  runs forward-in-time, replaces correlation with scene cross-attention, uses DINOv3+depth
  adapter, joint object+palm tokens, MSE on displacements. Uses **DenseTrack3Dv2 (this repo)**
  as its offline 3D-flow label generator (their Fig. 8). Our Model A ≈ LUCID intent model;
  our Model B adds explicit action (hand-plan) conditioning → forward dynamics.
- **CoTracker3 / DenseTrack3Dv2**: source of the `EfficientUpdateFormer` backbone and the
  factorized time/space + virtual-track attention we keep.
```
