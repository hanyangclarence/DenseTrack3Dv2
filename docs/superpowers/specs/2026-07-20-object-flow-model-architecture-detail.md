# Object-Flow Intent Model — Detailed Model Architecture

**Date:** 2026-07-20
**Status:** design detail, pending review
**Parent spec:** `2026-07-17-object-flow-intent-model-design.md` (§6 architecture, §7 repo map).
This document does **not** re-argue the design; it turns §6/§7 into concrete module signatures,
tensor shapes, and a line-by-line reuse map against the code that already exists in this repo, so
that implementation (build-order step 2) is a matter of typing, not deciding.

Everything here is grounded in files that exist today:
`densetrack3d/models/densetrack3d/{update_transformer,blocks}.py`, `models/embeddings.py`,
`models/loss.py`, and the finished data layer `data/flow_window_dataset.py`.

---

## 0. What is already fixed (do not re-open)

From the parent spec and the shipped data layer:

- **Inputs per item** are frozen by `FlowWindowDataset.__getitem__` (§2 below is a copy of its
  contract). The model consumes exactly those arrays; the model must not reach back to disk.
- **Target** = per-step displacement `Δ` (recover `x = x0 + Σ Δ`) + per-step visibility, camera frame.
- **Loss** = per-channel standardized MSE on `Δ` (stats ride along on the item as `dxyz_mean/std`) +
  `w_vis · balanced_bce`, masked by `target_vis`.
- **State = geometry only** (point cloud, no pixels, no hand geometry). **Action = hand pose**
  (`ergonomics ⊕ M_rel`, default `d_q = 26`).
- **Start small:** `C = 384`, `depth = 6`. Scale to 768/12 only if underfitting persists.

---

## 1. Notation & default config

| Symbol | Meaning | Default |
|---|---|---|
| `B` | batch | 16 |
| `P` | cloud points per frame (from `clouds.npz`) | 512 |
| `N` | query points | 16 |
| `M` | scene tokens out of the point encoder | 64 (final SA-level `npoint`) |
| `T_hist` | history steps | 4 (~0.5 s @ 8 Hz) |
| `T_pred` | reported prediction steps | 8 (~1 s @ 8 Hz) |
| `pred_pad` | extra predicted steps | 0 |
| `L_pred` | predicted steps `= T_pred + pred_pad` | 8 |
| `L` | full hand sequence `= T_hist + L_pred` | 12 |
| `d_q` | per-frame hand feature dim | 26 (`ergonomics` 20 ⊕ `M_rel` 6D) |
| `C` | hidden width | 384 |
| `depth` | backbone layers | 6 |
| `H` | attention heads | 6 (`C/H = 64`) |
| `w_vis` | visibility loss weight | 1.0 (sweep) |

The hand pose is always **one `C`-vector per step** (§5) — no hand-token axis. `raw_node_pose`
articulation instead of `ergonomics` only changes `d_q` (`72 + 6 = 78`); the encoding is unchanged.
Default path is `ergonomics ⊕ M_rel-6D` (`d_q = 26`).

---

## 2. Data contract (input to the model)

`FlowWindowDataset` emits this dict per item; `torch.utils.data.default_collate` stacks the array
fields into a leading batch axis, and `frame_meta` stays a list of dicts.

| Field | Shape (batched) | Dtype | Meaning |
|---|---|---|---|
| `cloud` | `(B, P+N, 3)` | float32 | present-frame object cloud, **query seeds are the last N rows**; **globally centered + isotropically scaled** (network input, §2.1) |
| `x0` | `(B, N, 3)` | float32 | query seeds `x_{n,t}`, **metric** (loss/recon anchor; `== cloud[:, -N:]` only in metric space) |
| `target` | `(B, L_pred, N, 3)` | float32 | camera-frame future positions, **metric** |
| `target_vis` | `(B, L_pred, N)` | bool | per-step visibility (loss mask) |
| `q_hist` | `(B, T_hist, d_q)` | float32 | hand cue (normalized articulation ⊕ raw `M_rel`) |
| `q_future` | `(B, L_pred, d_q)` | float32 | hand action (same layout) |
| `K` | `(B, 4)` | float32 | `fx, fy, cx, cy` (intrinsics; **not** needed by v1 model, kept for eval/backprojection) |
| `dxyz_mean` | `(B, 3)` | float32 | per-channel target-displacement mean (loss standardization) |
| `dxyz_std` | `(B, 3)` | float32 | per-channel target-displacement std |
| `frame_meta` | list of dict | — | provenance (`episode`, `t`, `stride_hz`, `query_idx`, …) |

Notes that shape the model:
- **No padding anywhere.** `P`, `N`, `T_hist`, `L_pred` are fixed per config, so every attention set
  is dense — the backbone's `attn_mask` is all-ones (we keep the argument for API compatibility with
  the reused blocks, but pass a trivial mask). The only real mask is `target_vis`, applied in the loss.
- `dxyz_mean/std` are identical across the batch (dataset-level stats broadcast per item); the loss
  can use `dxyz_mean[0]`. They are already the **per-step** displacement stats (`Δ_τ = x_τ − x_{τ−1}`,
  `x_0 = x0`) that `scripts/compute_flow_stats.py` fits over visible steps.
- The hand articulation block in `q_hist`/`q_future` is **already normalized** by the dataset; the
  `M_rel` rotation block is **raw** (bounded). The model does no further hand normalization.

### 2.1 Normalization scheme — positions vs. displacements (the tricky part)

Cloud/x0/target are all born in the same camera metre-space, but they play three roles and get three
treatments. The organizing fact: **positions and per-step displacements are different quantities,
standardized by different stats, consumed in different places — they never have to be reconciled.**

| Quantity | Role | Normalization | Stats | Where applied |
|---|---|---|---|---|
| `cloud` (incl. its last N query rows) | network **input** (spatial tokens, scene memory) | global center + **isotropic** scale | `cloud_mean` (3,) + `cloud_scale` (scalar) | **Dataset** (`normalize=True`) |
| `x0` | metric **anchor** for reconstruction + loss deltas | none (metric) | — | — |
| `target` | supervision, consumed as **deltas** | **per-channel** standardize the *delta*, not the position | `dxyz_mean/std` | **loss** (§9) |

The two normalizations are deliberately different operators, because positions and displacements need
different things:
- **Cloud (positions): isotropic scale, or you warp the shape.** `cloud = (cloud - cloud_mean) /
  cloud_scale`, where `cloud_scale` is a **single scalar** = `sqrt(mean(cloud_std²))` (RMS of the
  per-axis stds). Dividing xyz by *different* per-axis stds (`cloud_std ≈ [0.020, 0.021, 0.038]`, z ~2×
  x/y) would anisotropically distort the object — spheres → ellipsoids — which is exactly wrong for a
  point encoder whose job is local surface geometry (corner vs. edge vs. face). One scalar preserves
  shape and only removes overall scale (verified: normalized per-axis std ratio z/x = 1.86 matches the
  metric 1.84 — proportions intact). `cloud_std` is retained in the stats file for provenance only.
- **`cloud_mean` is GLOBAL, not per-item.** We do **not** center each cloud on its own centroid: the
  object's camera-frame placement encodes its pose *relative to the fixed hand-mounted camera*, a real
  signal the model should keep. Global centering removes only the dataset-wide offset, preserving
  placement variation.
- **Target (displacements): per-channel, because the axes really do differ.** `dxyz_std ≈ [0.0020,
  0.0016, 0.0015]` per-step motion is **~10–25× smaller** than the object extent, and here per-channel
  is correct — a displacement is not a shape, so whitening each axis to unit variance is the LUCID
  recipe (metric depth can't dominate the gradient) without any shape to distort.

Three consequences that make the split consistent:
1. **Cloud and x0 share ONE transform.** x0 *is* `cloud[-N:]` (same surface, two samplings), so they
   take the same center+isotropic-scale. After normalization `cloud[-N:] == normalize(x0)` exactly
   (verified: max abs diff 0.0) — the model reads the normalized seed for free, while the *unnormalized*
   `x0` stays in the item as the metric anchor. Do **not** normalize x0/target with cloud stats.
2. **The target is never cloud-normalized.** The loss forms `Δ = diff([x0 ; target])` in metric space,
   then standardizes `Δ` by `dxyz_mean/std`. Because `dxyz` was *fit on metric deltas* and GT deltas are
   *computed from metric x0/target*, the standardization is exact (verified: standardized Δ over 2000
   items → mean ~0, std ~1.00). This is why `x0`/`target` must stay metric.
3. **The two stats are genuinely different quantities** — a position scale (isotropic, for shape-preserving
   input normalization) and a displacement scale (per-channel, for a balanced regression target). They
   are never reconciled because they normalize different tensors consumed in different places.

The model consumes the **normalized** cloud (and hence normalized query seeds via `cloud[-N:]`), predicts
`Δ` in **standardized space**, and de-standardizes with `dxyz_std/mean` only for reconstruction/eval (§8).

---

## 3. Module map (new files under `densetrack3d/models/worldmodel/`)

```
densetrack3d/models/worldmodel/
  __init__.py        # exports IntentModel, IntentModelConfig
  point_encoder.py   # cloud (B,P+N,3) -> scene tokens S (B,M,C) [+ per-query feats]
  hand_encoder.py    # [q_hist;q_future] (B,L,d_q) -> a_future (B,L_pred,C), history-aware
  backbone.py        # scene-cross-attn + spatial + temporal, per layer
  heads.py           # delta head (C->3), vis head (C->1)
  intent_model.py    # full assembly + loss
scripts/train_intent.py   # training loop (separate doc / later)
```

Reuse (do **not** re-implement):

| Need | Reuse | From |
|---|---|---|
| within-step & across-step self-attention block | `AttnBlock` (attn_class=`Attention`) | `densetrack3d/models/densetrack3d/blocks.py` |
| query→scene cross-attention block | `CrossAttnBlock` | same |
| attention primitive (masking, flash flag) | `Attention` | same |
| MLP | `Mlp` | same |
| sinusoidal xyz embedding for query seeds | `get_3d_embedding(xyz, C, cat_coords=False)` | `models/embeddings.py` |
| visibility loss | `balanced_bce_loss` | `models/loss.py` |
| robust regression swap (optional) | `huber_loss` | `models/loss.py` |
| virtual-track spatial attention (only if N grows large) | `EfficientUpdateFormer` internals | `update_transformer.py` |

**Backbone decision — compose from `blocks.py`, do not call `EfficientUpdateFormer.forward`.**
`EfficientUpdateFormer` bundles two things we don't want here: (a) a dense `H×W` **local attention**
path (`local_attention`, `dH/dW`, `F.unfold`) that assumes tokens lie on an image grid — our N query
points do not; and (b) it has **no scene cross-attention** into an external point-token set, which is
exactly the addition §6.4 calls for. Building `backbone.py` directly from `CrossAttnBlock` + `AttnBlock`
gives the three-attention layer the spec wants with less machinery. The virtual-track trick inside
`EfficientUpdateFormer` stays available as a drop-in for the spatial stage if we later push `N` to
64–256 and dense `N×N` spatial attention gets expensive.

---

## 4. `point_encoder.py` — object-cloud state encoder (§6.1)

**Role:** turn the single cloud `(B, P+N, 3)` into `M` **position-aware** scene tokens `S ∈ (B, M, C)`
that the backbone cross-attends into. (An optional per-query FP feature exists but is off by default;
see point 4 below.)

**Recommended implementation: dependency-free PointNet++ set abstraction.** Clouds here are tiny
(`P+N ≈ 528`), so the classic bottleneck (CUDA ball-query kernels) is irrelevant — a pure-PyTorch SA
layer using `torch.cdist` for grouping is fast enough and avoids the CUDA-extension build pain this
env already fights (see memory `env-cuda-build-setup`). Keep the encoder behind a config switch so a
`point-transformer` or a compiled `pointnet2_ops` backend can replace it later without touching callers.

```python
class PosEnc(nn.Module):
    # ONE shared positional code for xyz -> C, used on BOTH sides of cross-attention (see below).
    # get_3d_embedding(xyz*scale, emb_C) [fixed sinusoid] -> Linear -> C. `scale` (=0.02) SHRINKS
    # unit-scale coords out of the ramp's aliasing regime into a smooth-decay band (§11 gate 2b).
    # Owned by IntentModel and
    # passed to both the point encoder and the query-token init so the SAME weights encode both.
    def forward(self, xyz): return self.proj(get_3d_embedding(xyz * self.scale, self.emb_C, cat_coords=False))

class SetAbstraction(nn.Module):
    # FPS to `npoint` centroids -> ball-query / kNN group (torch.cdist) -> shared MLP -> max-pool
    def forward(self, xyz, feat) -> (new_xyz, new_feat): ...   # new_xyz (B,npoint,3), new_feat (B,npoint,c)

class FeaturePropagation(nn.Module):
    # 3-NN inverse-distance interpolation of centroid features back to arbitrary query xyz
    def forward(self, query_xyz, centroid_xyz, centroid_feat) -> query_feat: ...  # (B, Nq, c)

class PointEncoder(nn.Module):
    # sa_cfg = ((npoint, radius, nsample, mlp), ...). RADII ARE IN NORMALIZED UNITS, not metres:
    # the cloud enters already centered + isotropically scaled (§2.1), where a typical object has
    # ~0.75 mean radius from its centroid (verified). Metric radii like 0.05 m would be ~1-2 mm in
    # this space -> near-empty ball queries. Grouping is torch.cdist kNN/ball in normalized coords.
    def __init__(self, pos_enc, out_dim=384, sa_cfg=(( 256, 0.2, 32, (32,32,64)),
                                           ( 128, 0.4, 32, (64,64,128)),
                                           (  64, 0.8, 32, (128,128,256))),
                 emit_query_feat=False):       # OFF by default -- cross-attn into S supplies context (§6)
        self.pos_enc = pos_enc                 # SHARED positional code (same instance as query init)
        # last-level channels projected to out_dim=C
    def forward(self, cloud, x0_norm):    # cloud (B, P+N, 3) normalized; x0_norm (B, N, 3) = cloud[:, -N:]
        xyz, feat = cloud, None
        centroids = []
        for sa in self.sa_layers:
            xyz, feat = sa(xyz, feat)          # FPS downsample + group + pool; xyz (B, npoint, 3)
            centroids.append((xyz, feat))
        cxyz, cfeat = centroids[-1]            # (B, M, 3), (B, M, c)   M = last SA npoint (e.g. 64)
        S = self.proj(cfeat) + self.pos_enc(cxyz)   # (B, M, C) SCENE TOKENS, position via the SHARED code
        query_feat = None
        if self.emit_query_feat:               # optional ablation: interpolate S back to query positions
            query_feat = self.proj(self.fp(x0_norm, cxyz, cfeat))     # (B, N, C) via 3-NN FP
        return S, query_feat
```

**Point 4 — how `S` and `query_feat` are separated, and why `query_feat` is OFF by default.**
PointNet++ SA layers **FPS-downsample** to M centroids; the N query points generally do **not** survive
as centroids, so you cannot "gather features at the query indices." The correct mechanism:
- **`S` = the M final-level centroid tokens, made position-aware** by adding a centroid positional
  embedding `pos_enc(cxyz)`. This mirrors DenseTrack3D's `pos_emb` and is what lets the query token do a
  spatially-grounded lookup via cross-attention. **Crucially, `pos_enc` is the SAME shared module used to
  position the query tokens (§6) — same function AND same weights** (see the "unified positional code"
  box below): both sides of cross-attention read position through one code, so Q·K can match a query to
  nearby centroids directly instead of first learning a translation between two coordinate bases.
- **`query_feat` = a Feature-Propagation interpolation of centroid features back to the N query xyz**
  (3-NN inverse-distance, `query_feat[n] = Σ_k w_k · cfeat[nn_k(x0_n)]`). **Off by default:** with a
  position-aware `S`, scene **cross-attention already supplies each query's local surface context**
  (§6, point 3), so the FP branch is redundant. Keep it as an ablation to *measure* whether
  cross-attention localizes well enough; turn on only if not.

Details:
- **Normalization is already done in the Dataset (§2.1).** The cloud arrives globally centered
  (`cloud_mean`) and isotropically scaled (`cloud_scale`, one scalar — shape preserved), so the encoder
  does **not** re-normalize. `x0_norm = cloud[:, -N:]` is the normalized query position handed to
  `pos_enc`/FP; the metric `x0` (for the loss/reconstruction) stays separate. The SA radii above live in
  this same normalized space.
- **Train-time point augmentation (§6.1, our analog of LUCID human-region aug):** random point
  dropout, small Gaussian jitter, and patch removal (drop a ball of points). **Place these in the
  Dataset/collate**, not the encoder, so eval is clean and augmentation is a data concern.
  *(Not built in the data layer yet — a follow-up; the encoder must be robust to variable effective
  point count, which the SA + max-pool already is.)*

---

## 5. `hand_encoder.py` — sequential hand-action encoder (§6.3)

**One `C`-vector per hand pose.** A hand pose at frame τ is a single configuration; the encoder's job
is to compress that frame's `d_q` features to one `C`-channel vector. There is **no per-keypoint token
axis** — the keypoints (or joint angles) are just the `d_q` input columns. Output is `a_future (B,
L_pred, C)`: one action feature per predicted step.

One sequential pass over the **whole** hand sequence `[q_hist ; q_future]`, so each future step's
feature has **already integrated the history** — history informs the action inside the encoder, with no
separate fusion module or separate history injection downstream (see §6/§7).

```python
class HandEncoder(nn.Module):
    def __init__(self, d_q=26, hidden=384, kind="bigru", n_layers=1):
        self.embed = nn.Linear(d_q, hidden)                 # per-frame embed: d_q -> C
        self.temporal = BiGRU(hidden) | TransformerEncoder  # non-causal over L (see note)
    def forward(self, q_hist, q_future):
        # q_hist (B, T_hist, d_q), q_future (B, L_pred, d_q)
        q_seq = cat([q_hist, q_future], dim=1)              # (B, L, d_q),  L = T_hist + L_pred
        e     = self.embed(q_seq)                           # (B, L, C)     one C per frame/pose
        seq   = self.temporal(e)                            # (B, L, C)
        a_future = seq[:, T_hist:]                          # (B, L_pred, C)  history-aware per-step action
        return dict(a_future=a_future)                      # the sole hand signal (see §7)
```

- **`a_future` is the only hand output the backbone needs.** Because the encoder is sequential over
  `[hist ; future]`, `a_future[τ]` already carries the recent-motion / velocity cue from the history —
  so we do **not** separately inject history tokens or a pooled `ctx` (that was redundant; see §6). The
  history's job is to shape `a_future`, and it does that inside the encoder.
- **Prefer a non-causal encoder** (bidirectional GRU or a small transformer with a step positional
  embedding over `L`): the whole hand plan is given up front, so every `a_future[τ]` should see the
  entire history *and* the entire future plan, not just frames up to τ. Unidirectional GRU is the cheap
  fallback. Config switch `kind`.
- **`d_q` per articulation choice:** `ergonomics` 20, or `raw_node_pose` = 25 keypoints × 3 = 75 raw
  (the dataset drops node0, the wrist-at-origin, → **72**), each optionally ⊕ `M_rel` (6D=6 or 3×3=9).
  Default `ergonomics ⊕ M_rel-6D → d_q = 26`; `raw_node_pose ⊕ M_rel-6D → d_q = 78`. In every case the
  frame is flattened into `d_q` and encoded to one `C` — nothing here changes but the input width.
- The encoder is **frame/representation-agnostic**: it consumes whatever `q_hist/q_future` columns the
  dataset produced; only `d_q` changes.

---

## 6. Query-token initialization (§6.2) — x0 is the seed; nothing step-invariant is added

**How the query point's initial pose `x0` enters the model — one place, one mechanism.** The token for
query point `n` is seeded from the **shared positional code** `pos_enc(x0_norm[n])` (the same module,
same weights, that positions the scene tokens `S` in §4), broadcast across the `L_pred` predicted steps.
That seed is the token's **identity and position**: it tells the model *which* point this token predicts
and *where on the object it starts*. Using the identical `pos_enc` on both the query side here and the
scene side in §4 is the point of the "unified positional code" box below — cross-attention can then match
a query to nearby centroids directly, with no learned change-of-basis.

```python
def init_query_tokens(x0_norm, pos_enc, query_feat=None):
    # x0_norm (B, N, 3) normalized query xyz (= cloud[:, -N:]); pos_enc SHARED with the point encoder (§4)
    seed = pos_enc(x0_norm)                                        # (B, N, C) position/identity (SAME code as S)
    if query_feat is not None:                                    # query_feat OFF by default (see below)
        seed = seed + query_feat                                  # content add (already projected to C by FP)
    tok  = seed[:, None].expand(-1, L_pred, -1, -1).clone()        # (B, L_pred, N, C) same seed every step
    tok  = tok + self.step_emb[None, :, None]                      # (B, L_pred, N, C) learned per-step embed
    return tok
```

**What is NOT added here (points 2 & 3):**
- **No history / `ctx`.** The history reaches the model through the **history-aware `a_future`** (the
  hand encoder is sequential over `[hist ; future]`, §5), which is FiLM-injected in the backbone (§7).
  Adding a separate pooled `ctx` or prepending history tokens is **redundant** — the same information is
  already in `a_future`. (It was also incoherent: a pooled history is *one* timestep, so "concat along
  time" would add a single token, not a real temporal memory.) Dropped entirely.
- **`step_emb`** (learned `(L_pred, C)`) is the only additive term, and it is *positional* (which future
  step), exactly the kind of signal DenseTrack3D adds (`time_emb`). It is added last so nothing washes it out.

**Is `query_feat` from the point encoder necessary? No — default OFF.** A query token needs two things
from geometry: (1) **position/identity**, given by `pos_enc(x0_norm)` above; (2) **local surface context**
(is this point on a corner, an edge, a face?). (2) is supplied by the **scene cross-attention** into the
point tokens `S` (§7) — provided `S` is position-aware, which we ensure by adding the shared positional
code to `S` in the encoder (§4). With positioned `S`, the query token and the scene tokens live in one
spatial frame, and cross-attention *is* the spatially-grounded lookup that gathers local context — making
a separate FP-interpolated `query_feat` branch redundant. So `query_feat` is an **optional ablation, off
by default**; turn it on only if cross-attention proves too weak to localize (measure it).

> **Unified positional code (both sides of cross-attention share it).** Cross-attention grounds a query
> token by matching it against scene tokens via Q·K. If the two sides encoded position with *different*
> learned codes (query: one `Linear∘sinusoid`; scene: a separate `pos_mlp`), the attention projections
> would first have to learn a translation between those two coordinate bases before any spatial matching
> could happen — capacity we can't spare on our data budget. So we use **one shared `PosEnc` instance**
> (§4 `PosEnc`), owned by `IntentModel` and passed to *both* the point encoder (for `S`'s centroid
> positions) and `init_query_tokens` (for the query seed). Same function *and* same weights → the query
> and the centroids are placed in a single coordinate code, and cross-attention matches positions
> directly. Since §7.2 makes position-aware `S` load-bearing, this is the default, not an aside.

Notes:
- `PosEnc` wraps `get_3d_embedding` (embeddings.py) → `(B, ·, 3·emb_C)` → `Linear → C`, with the input
  **down-scaled** (`scale=0.02`) out of the ramp's aliasing regime into a smooth-decay band (§11 gate 2b).
  Pick `emb_C` for frequency *count* (e.g. 128); `scale` sets the *range*. The *same* instance is used for
  `S` (§4) and the query seed here.
- `step_emb` learned table; a sincos step embedding is a drop-in.

---

## 7. `backbone.py` — three-attention update (§6.4)

Per layer, in order: **scene cross-attention → spatial self-attention → temporal self-attention.**
Non-causal in time (whole horizon given up front). The only conditioning input is the history-aware
action `a_future` (§5), injected per-step by **FiLM**. There is no separate history token stream and no
hand-token axis (see the two "removed" notes below).

```python
class IntentBackbone(nn.Module):
    def __init__(self, C=384, depth=6, heads=6, mlp_ratio=4.0):
        self.cross = ModuleList(CrossAttnBlock(C, C, heads, mlp_ratio) for _ in range(depth))
        self.space = ModuleList(AttnBlock(C, heads, mlp_ratio=mlp_ratio)  for _ in range(depth))
        self.time  = ModuleList(AttnBlock(C, heads, mlp_ratio=mlp_ratio)  for _ in range(depth))
        self.film  = ModuleList(nn.Linear(C, 2 * C) for _ in range(depth))   # per-step (gamma, beta)
        for f in self.film: nn.init.zeros_(f.weight); nn.init.zeros_(f.bias)  # start (1+gamma)=1, beta=0

    def forward(self, tok, S, a_future):
        # tok (B, L_pred, N, C), S (B, M, C) [position-aware, §4], a_future (B, L_pred, C)
        B, Lp, N, C = tok.shape
        for l in range(depth):
            # per-step FiLM inject the action; affine, differs across tau -> no fight with step_emb
            gamma, beta = self.film[l](a_future).chunk(2, -1)             # (B, L_pred, C) each
            tok = (1 + gamma[:, :, None, :]) * tok + beta[:, :, None, :]

            # 1. scene cross-attn: every (tau, token) attends into the M scene tokens S
            x = rearrange(tok, "b l n c -> (b l) n c")
            x = self.cross[l](x, repeat(S, "b m c -> (b l) m c", l=Lp))   # CrossAttnBlock(query, context)
            # 2. spatial self-attn: within a step, across the N tokens -- inter-point coherence
            x = self.space[l](x)
            tok = rearrange(x, "(b l) n c -> b l n c", b=B)
            # 3. temporal self-attn: across the L_pred predicted steps, per token, non-causal
            y = rearrange(tok, "b l n c -> (b n) l c")
            y = self.time[l](y)
            tok = rearrange(y, "(b n) l c -> b l n c", b=B)
        return tok                                                       # (B, L_pred, N, C)
```

### 7.1 Why FiLM (channel C) is the sole action injection

A hand pose is one `C`-vector per step (§5), so `a_future` is a **per-step global conditioning signal**
over the object tokens. **FiLM** — a per-step affine `(1+γ)⊙tok + β`, `γ,β = Linear(a_future)` — is the
standard operator for that: it *modulates every channel* of every object token by the action, is
**per-step** (γ,β differ across τ, so it never smears like a step-invariant bias would), and costs one
small `Linear(C→2C)` per layer. Plain addition (`tok += a_future`) is the weak special case (γ=0);
FiLM strictly generalizes it. **Zero-init the FiLM Linear** so `(1+γ)=1, β=0` at start — the model
begins effectively unconditioned (no noise-scale modulation forced into every layer on step one),
matching the near-zero delta-head init; the action's influence grows as training moves the FiLM weights
off zero. This is a real fix, not cosmetic: a randomly-initialized FiLM multiplies every token by
noise-scale γ from the first step, fighting the "start near no-motion" prior the head init establishes.

**Concat-along-N is removed.** The earlier draft kept a "K=25 keypoint tokens" path that appended hand
tokens on the spatial axis. That is gone: a hand pose is a single `C`-vector, not 25 positioned tokens
(§5, point 1), so there is nothing to place among the N query points, and the spatial axis stays exactly
the N object tokens (its job is inter-point rigidity, not hand↔object geometry — which doesn't exist
here, §6.3 of the parent spec). Injection is FiLM, full stop.

### 7.2 Other notes
- **History injection is removed** (point 2). The recent-motion cue is already inside `a_future` because
  the hand encoder runs sequentially over `[hist ; future]` (§5); prepending a separate history token
  stream would re-inject the same information. The temporal stage attends only over the `L_pred`
  predicted steps.
- **`S` must be position-aware, via the SAME `pos_enc` as the query seed.** Cross-attention is the query
  token's *only* route to local surface context (§6, point 3), so the point encoder adds `pos_enc(cxyz)`
  to `S` (§4) — the identical shared module that positions the query tokens, so Q·K matches positions with
  no learned change-of-basis. Without a position-aware `S`, cross-attention degrades to a content-only pool
  and query localization suffers.
- `CrossAttnBlock.forward(x, context, mask=None)` and `AttnBlock.forward(x, mask=None)` are used exactly
  as in `update_transformer.py`; sets are dense here so we pass `mask=None`.
- **Scaling knob:** if `N` grows to 64–256 and spatial `N×N` attention dominates, swap the `space` stage
  for the virtual-track spatial attention from `EfficientUpdateFormer` (real↔virtual cross-attn) — same
  `blocks.py` primitives, sub-quadratic in N.

---

## 8. `heads.py` — displacement + visibility (§6.5)

```python
class Heads(nn.Module):
    def __init__(self, C=384):
        self.delta = nn.Linear(C, 3)     # per-step displacement, zero-ish init (trunc_normal std 1e-3)
        self.vis   = nn.Linear(C, 1)     # per-step visibility logit
    def forward(self, tok_obj):          # tok_obj (B, L_pred, N, C) = backbone output (all object tokens)
        delta = self.delta(tok_obj)                       # (B, L_pred, N, 3)  standardized-space
        vis_logit = self.vis(tok_obj).squeeze(-1)         # (B, L_pred, N)
        return delta, vis_logit
```

- **`delta` is predicted in standardized space** (unit-variance per channel). De-standardize with the
  item's `dxyz_std/mean` before composing the trajectory:
  `Δ_metric = delta · dxyz_std + dxyz_mean`; `x_pred = x0[:, None] + Δ_metric.cumsum(dim=1)`.
  The loss (§9) compares in standardized space, so this de-standardization is only for eval/reporting.
- **Init the delta head small** (as `EfficientUpdateFormer.initialize_weights` does for its flow head:
  `trunc_normal_(weight, std=0.001)`) so the model starts near "no motion" — a good prior for short
  horizons and stable early training.

---

## 9. `intent_model.py` — assembly, forward, loss

```python
@dataclass
class IntentModelConfig:
    C=384; depth=6; heads=6; mlp_ratio=4.0
    articulation="ergonomics"; d_q=26; hand_kind="bigru"   # non-causal encoder (§5)
    point_out=384; sa_cfg=<default>; emit_query_feat=False  # cross-attn into position-aware S (§4/§6)
    emb_C=128; pos_scale=0.02; w_vis=1.0; reg="mse"         # pos_scale=0.02 from §11 gate 2b (anti-alias); reg or "huber"

class IntentModel(nn.Module):
    def __init__(self, cfg):
        self.pos_enc = PosEnc(emb_C=cfg.emb_C, C=cfg.C, scale=cfg.pos_scale)   # SHARED positional code
        self.pt   = PointEncoder(self.pos_enc, ...)          # uses pos_enc for S's centroid positions (§4)
        self.hand = HandEncoder(...); self.backbone = IntentBackbone(...); self.heads = Heads(...)
    def forward(self, batch):
        cloud = batch["cloud"]                                    # (B, P+N, 3) normalized (dataset)
        x0_norm = cloud[:, -self.N:]                              # normalized query seeds (== norm(x0))
        S, qfeat = self.pt(cloud, x0_norm)                        # S (B,M,C) position-aware; qfeat None by default
        a_future = self.hand(batch["q_hist"], batch["q_future"])["a_future"]  # (B,L_pred,C) history-aware
        tok = init_query_tokens(x0_norm, self.pos_enc, qfeat)     # (B,L_pred,N,C) seed = pos_enc(x0) (+qfeat)
        tok = self.backbone(tok, S, a_future)                     # (B,L_pred,N,C) FiLM(a_future)+cross/space/time
        delta, vis_logit = self.heads(tok)                        # standardized delta, vis logit
        return dict(delta=delta, vis_logit=vis_logit)
```

The forward reflects the corrections: query seeds are read from the **normalized** cloud
(`cloud[:, -N:]`, §2.1); the query seed and the scene tokens `S` share **one `pos_enc` instance** so
cross-attention matches positions directly (§4/§6), with `query_feat` **off** by default; the sole hand
signal is the **history-aware `a_future`** (no separate history stream, §5/point 2); and the backbone
conditions purely by per-step **FiLM** (no hand-token axis, §7/point 1).

### Loss

Per-channel standardized MSE on the **per-step displacement**, masked by visibility, plus balanced BCE
on visibility:

```python
def intent_loss(out, batch, w_vis=1.0, reg="mse"):
    # ground-truth per-step displacement in metres
    traj  = cat([batch["x0"][:, None], batch["target"]], dim=1)   # (B, L_pred+1, N, 3)
    d_gt  = diff(traj, dim=1)                                     # (B, L_pred, N, 3) metres
    std, mean = batch["dxyz_std"][:, None, None], batch["dxyz_mean"][:, None, None]
    d_gt_std  = (d_gt - mean) / std                              # standardized target
    d_pred    = out["delta"]                                     # already standardized space

    vis  = batch["target_vis"]                                   # (B, L_pred, N) bool
    # TWO-SIDED delta mask: Delta_tau = x_tau - x_{tau-1} needs BOTH endpoints valid. The
    # predecessor of step 0 is the query point at t (visible by construction).
    prev_vis = cat([ones_like(vis[:, :1]), vis[:, :-1]], dim=1)   # (B, L_pred, N)
    m    = (vis & prev_vis)[..., None].float()
    # Occluded steps carry NaN coords in `target` (hence in d_gt_std). Scrub the NaN and
    # zero the masked targets BEFORE the squared error -- otherwise backprop yields NaN
    # grads (2*(d_pred - NaN)*0 = NaN) even though the forward sum looks masked-out.
    # SAFETY: scrubbing is only correct if every NaN is masked; assert it (a NaN on a
    # visible step would otherwise become a silent 0-target trained at full weight).
    assert not (~isfinite(d_gt) & (m > 0)).any()   # NaN(target) subset of masked steps
    d_gt_std = nan_to_num(d_gt_std) * m
    reg_l = (huber_loss if reg=="huber" else mse)(d_pred * m, d_gt_std)   # (B,L_pred,N,3)
    reg_l = (reg_l * m).sum() / m.sum().clamp_min(1)             # both-endpoints-visible steps only

    vis_l = balanced_bce_loss(out["vis_logit"][..., None], vis.float()[..., None])   # loss.py, reused
    return reg_l + w_vis * vis_l, {"reg": reg_l, "vis": vis_l}
```

- **Standardize the target, predict in standardized space, mask by `vis`** — this is the exact LUCID
  recipe and the reason `compute_flow_stats.py` produced `dxyz_mean/std`.
- `balanced_bce_loss` from `models/loss.py` is reused verbatim (it expects `(b,s,n,c)`; add a trailing
  channel). It internally balances pos/neg, so genuine occlusion imbalance is handled.
- **Masking subtlety (two-sided).** Occluded steps carry **NaN** coords (the source `object_flow.pkl`
  writes NaN for occluded points, and the dataset copies coords verbatim — *verified on real data:
  ~33% of target point-steps are NaN and every one sits on `vis=0`; `x0` is NaN-free*). So a delta must
  be trained only when *both* its endpoints are valid: `mask_τ = vis_τ & vis_{τ-1}` (predecessor of step
  0 is the query point at t, visible by construction). A one-sided `vis_τ` mask would train the
  reappearance step `Δ_τ = x_τ − x_{τ-1}(NaN)` on a corrupted delta. **In this dataset visibility is
  monotonic** (once a point disappears it stays gone), so `vis_τ ⟹ vis_{τ-1}` and the two-sided mask
  *equals* the one-sided one — meaning `compute_flow_stats.py`'s "visible steps only" dxyz fit is **not**
  contaminated by reappearance jumps. We still write the mask two-sided: correct in general, free under
  monotonicity, and robust if the labeling ever changes. (A cheap `assert (vis[:,1:] <= vis[:,:-1]).all()`
  documents the assumption at load.)
- **NaN scrub + safety assert.** Because occluded targets are NaN (not stale-but-finite), the loss scrubs
  them *before* the squared error — `NaN·0 = NaN` would otherwise poison both the forward sum and the
  backward pass (`2·(d_pred − NaN)·0 = NaN`). The scrub `nan_to_num(d_gt_std) · m` is only safe while
  every NaN is masked; a NaN on a **visible** step (bad depth on a tracked point) would be silently turned
  into a 0 target (= mean displacement) and trained at full weight. The loss therefore asserts
  `NaN(d_gt) ⊆ (m==0)`, converting that silent data-corruption mode into a loud failure.
- Optimization (from spec §6.5): AdamW, cosine `1e-4 → 1e-6`, 5 warmup epochs, wd 0.01, grad-clip 1.0,
  EMA 0.999 for eval.

---

## 10. End-to-end shape trace (defaults)

```
cloud        (B, 528, 3)  normalized   # P=512 + N=16;  x0_norm = cloud[:, -16:]
  └─PointEncoder(cloud, x0_norm)→ S (B, 64, 384); S position = pos_enc(cxyz);  query_feat=None (default)
q_hist       (B, 4, 26)
q_future     (B, 8, 26)
  └─HandEncoder([hist;future])→ a_future (B, 8, 384)   # history-aware; one C per pose; no ctx/h_hist out
x0_norm      (B, 16, 3)
  └─init_query_tokens(x0_norm, pos_enc)→ tok (B, 8, 16, 384)   # seed = pos_enc(x0) + step_emb; SAME pos_enc as S
backbone(tok, S, a_future) [FiLM]:
   per layer:  FiLM(a_future)→ cross-attn(S)→ spatial(over N)→ temporal(over L_pred=8, non-causal)
  → tok (B, 8, 16, 384)
heads→ delta (B, 8, 16, 3)  [standardized];  vis_logit (B, 8, 16)
de-standardize + cumsum, anchored on METRIC x0→ x_pred (B, 8, 16, 3) camera metres; report x_pred[:, :T_pred]
```

**Parameter count** at `C=384, depth=6` — **~35 M**, backbone-dominated (measured, not estimated):

| Component | Params |
|---|---|
| backbone (18 blocks) | **33.7 M** — cross 10.6 + spatial 10.6 + temporal 10.6 + FiLM 1.8 |
| hand encoder | 0.68 M |
| point encoder | 0.33 M (the 1×1 conv stacks are tiny — <0.35 M total) |
| shared `PosEnc` | 0.15 M |
| heads | ~0 M |

Each `AttnBlock`/`CrossAttnBlock` is ~1.77 M (`4C²` attention + `8C²` MLP at `mlp_ratio=4`), and there
are **3 families × 6 layers = 18** of them → ~32 M. (An earlier draft of this line wrote "~1.8 M each"
then "≈15–20 M" — an arithmetic slip; 18 × 1.8 ≈ 32, not 20. The point encoder is **not** a material
contributor — attributing the size to "pointnet MLPs" was backwards.)

At ~35 M this is still well under LUCID's ~85 M, consistent with the "start small, ~100× less data"
stance (§0 also keeps the escape hatch to scale *up* to 768/12 if underfitting persists). If a smaller
model is wanted instead, the measured landing points are: `depth=4` → **~23.5 M**, `C=256, depth=6` →
**~15.6 M** (both one-line `IntentModelConfig` changes).

---

## 11. Sanity & build gates (build-order step 2)

Before training at scale, in order:

1. **Shape test.** Feed one collated batch (from the existing dataset benchmark path) through
   `IntentModel`; assert every shape in §10. Model **outputs** must be NaN-free — but the **targets are
   NOT**: occluded steps carry NaN coords (§9), so the check is `NaN(target) ⊆ (target_vis==0)` and
   `x0`/outputs finite, not "no NaNs anywhere." The loss enforces exactly this subset with an assert.
2. **Loss finiteness.** `intent_loss` returns finite scalars; `vis` mask sum > 0.
2b. **Sinusoid is a positional code, not a hash.** `get_3d_embedding` comes from the tracker, where
   inputs are pixel-scale (hundreds); at `emb_C=128` its frequency ramp spans **0–~984 rad/unit**. On our
   ~unit-scale normalized coords the band is far too *fine*, not too coarse: at `pos_scale=1.0` a point-pair
   0.1 apart sees up to ~98 rad of phase delta, deep in the **aliasing** regime, so nearby points get
   **near-orthogonal** codes (cos ≈ 0.02 at *every* distance — measured). That is a hash: cross-attention
   gets no "closer centroid → higher Q·K" prior, and localization must be learned entirely by the `Linear`
   picking the few usable low-frequency channels from random init — a bet we can't afford on this data
   budget. **The real risk is cos → 0, not cos → 1.** So the gate does **not** merely assert distinguishability
   (`cos < 0.99`, which only catches "all tokens alike"); it sweeps distance and asserts **smooth monotone
   decay** over the object's ~0.1–1.5 range with a mid-range value that is neither ~1 nor ~0. Fix by
   **lowering** `pos_scale` (the fixed input multiplier), not raising it — `emb_C` sets the frequency *count*,
   `pos_scale` sets the *range*. Verified landing: **`pos_scale = 0.02`** gives cos 0.95→0.81→**0.48**(@0.19,
   the median NN gap)→0.18→0.09→0.06 across d = 0.05…1.5 — locality preserved, still distinguishable. Tune
   **once**; because `PosEnc` is shared (§4/§6) this fixes resolution for both `S` and the query seed at once.
3. **Overfit one clip.** Single episode, `split="all"`, tiny LR schedule; the model must drive
   `reg` → ~0 and visibility BCE down. If it cannot overfit 1 clip, the architecture is wrong, not the
   data. Visualize predicted flow with the existing `data/viz_flow_window_item.py` (feed `x_pred` in
   place of `target`) or `preprocess/viz_goal_flow.py`.
4. **Gradient reach.** Confirm gradients flow to the point encoder (`S`), the shared `PosEnc` (it should
   receive gradient from *both* the `S` side and the query-seed side), and the hand encoder (`a_future`,
   via the FiLM γ/β) — no accidentally-detached branch. A common bug is a conditioning signal (FiLM γ/β,
   `S`) that is added/attended but sits off the autograd graph, so its encoder never trains.

---

## 12. Deferred / config-switch surface (not v1 blockers)

- Point-augmentation hooks (jitter, dropout, patch removal) in the Dataset/collate — promised in
  §6.1, not yet built.
- `query_feat` FP branch (`emit_query_feat=True`) — ablation to test if cross-attention localizes
  well enough without it (§4/§6).
- `raw_node_pose` (`d_q=78`) and `wrist_repr="matrix"` articulation paths (still one `C` per pose, §5).
- Virtual-track spatial attention swap for large `N`.
- `pred_pad` > 0 to support the deferred goal-pose + endpoint-velocity target (§4.2 of parent spec).
- Huber regression swap; mixture/diffusion head if mode-averaging (blurry mean flow) appears (§9.3).
- No-hand baseline (drop the FiLM action injection) — an ablation, wired by config, at the end.

All of the above are switches on `IntentModelConfig` or the Dataset config, so none require touching
module internals.

---

## 13. Training-loop hazards (bake into `train_intent.py`, do NOT let a refactor drop these)

These are correctness requirements for the *next* file, not optional polish. Written here because the
training loop doesn't exist yet and these are exactly the kind of thing that gets "simplified" away.

- **EMA must copy BUFFERS, not just parameters.** `SetAbstraction` uses `nn.BatchNorm2d`, whose
  `running_mean`/`running_var`/`num_batches_tracked` are **buffers, not parameters** (verified: 27 such
  buffers in the default model). The spec calls for EMA-for-eval (§9 / parent §6.5). A naive EMA that
  iterates `model.parameters()` copies the weights but leaves the EMA model's BN running stats at
  random init → **eval silently produces garbage** while training metrics look fine. The EMA shadow
  MUST also track buffers. Two safe implementations:
    1. EMA the params, but **copy buffers verbatim** from the live model each update
       (`ema.buffers() ← model.buffers()`), since running stats are already a moving average; or
    2. use `torch.optim.swa_utils.AveragedModel(..., use_buffers=True)`, which averages both.
  A one-line `assert len(list(ema.buffers())) == len(list(model.buffers()))` in the EMA setup makes a
  future parameters-only "simplification" fail loudly. **This comment must survive into the code.**
- **Alternative (a design choice, not done here): swap BatchNorm → GroupNorm in `SetAbstraction`.**
  GroupNorm has **no running-stats buffers**, so it sidesteps the EMA hazard *and* BN's train/eval
  distribution shift on our **small, variable** effective point counts (occlusion + the planned point-
  dropout aug make per-item point count non-constant, which BN handles poorly). This changes the model,
  so it's deferred to a deliberate decision — but if we hit BN-related eval instability, this is the first
  lever. (No CUDA/build implication; pure-PyTorch either way.)
- **BN in eval needs warm running stats.** Because BN uses batch stats in train and running stats in
  eval, run enough training steps (or the gate-3 overfit) before trusting any eval number; a cold-start
  eval right after init reflects init BN stats, not the model.
