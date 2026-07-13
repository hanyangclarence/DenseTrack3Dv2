# Object-motion smoothing for windowed 3D tracks

**Date:** 2026-07-11
**Status:** Approved (ready for implementation plan)
**New file:** `postprocess/smooth_object_motion.py`

## Goal

Turn the tracked 3D object flow into a signal that reflects the **overall object
motion** (the deliberate rotation/translation the hand is producing) rather than the
**finger-reset jitter** — the back-and-forth (+20° then −10°) that a human hand adds
because fingers must reset between rotation strokes. The user wants that jitter
**attenuated, not removed**: genuine, sustained direction changes (the hand rotates
one way for a few seconds, then deliberately the other way) must survive.

Input is an existing `dense_3d_track.pkl` (from `preprocess/track_windowed.py`).
Output is a smoothed `dense_3d_track.pkl` plus a `tracks_2d.mp4` overlay, so the
result stays drop-in for training and the existing visualizers, and raw-vs-smoothed
can be eyeballed side by side.

## What the data shows (measured on `results/sphere_2/dense_3d_track.pkl`)

- **The object moves rigidly.** Frame-to-frame Kabsch/Procrustes fit on the ~66
  points visible in each adjacent frame pair has median residual **0.7 mm**
  (p95 7.8 mm). So one rigid pose per frame explains the motion — we do not smooth
  2769 point tracks independently.
- **Jitter and intent are frequency-separated.** Signed angular velocity about the
  local dominant axis reverses sign every **~6 frames** (finger jitter); 80–90% of
  the oscillation energy sits at **2–4 frame periods**. Genuine direction changes
  are seconds apart (~1 s smoothing leaves ~71 sign changes over 1600 frames). A
  low-pass with a cutoff between these scales separates them cleanly.
- **Efficiency ratio ≈ 36%**: only ~1/3 of rotational travel is net progress; ~2/3
  is oscillation. Confirms the problem is real and worth correcting.

## Why smoothing must happen in motion space, not position space

The back-and-forth is a **real, low-frequency rigid motion**, not high-frequency
per-point noise. Low-pass filtering the 3D point positions directly would (a) fail
to separate the slow back-and-forth from the slow net spin, and (b) destroy the
rigid geometry (points would drift independently). Instead we recover the object's
rigid motion, filter **that** 6-DOF signal, and re-apply it to the points — rigidity
is preserved by construction, and the filter acts on exactly the quantity that
oscillates.

**No global-axis / net-direction assumption.** All estimation is on *local*
frame-to-frame motion, so a deliberate multi-second reversal is itself a
low-frequency component that passes through the filter. The method is **shape-
agnostic** (Kabsch assumes rigidity only, never geometry — works for sphere, box,
stick) and handles **translation + rotation** together (full 6-DOF twist).

## Algorithm

Let `coords (T,N,3)`, `vis (T,N)`, `colors (N,3)` be the input pkl arrays.

### 1. Per-frame incremental rigid motion (robust)
For each adjacent pair `(f, f+1)`, take points visible in **both** with valid
(finite, nonzero-depth) coords. Estimate the incremental rigid transform
`T_f ∈ SE(3)` mapping frame `f`'s points to frame `f+1`'s, using **RANSAC + Kabsch**
(reject tracks with large residual — occlusion/depth spikes) rather than plain
least squares. Represent `T_f` as a body twist `ξ_f = (ω_f, v_f) ∈ ℝ⁶`
(`ω` = rotation vector = axis·angle, `v` = translation).

- Pairs with fewer than `--min-corr` (default 8) valid correspondences: mark the
  twist missing and fill by linear interpolation of the twist components (the pose
  is coasted through short gaps). Report how many frames were gap-filled.

### 2. Low-pass the twist sequence
Filter each of the 6 twist channels over time with a zero-phase smoother
(Gaussian / moving-average; zero-phase so motion isn't time-shifted). One knob:
**`--smooth-window`** (frames, default derived from the jitter scale, ≈ 9). Larger =
more attenuation. The fast ±jitter averages toward zero; sustained drift (including
genuine multi-second reversals) passes through.

- `--smooth-window 1` (or `0`) is a **no-op passthrough** (identity), so the script
  can regenerate an equivalent pkl for A/B and as a self-test.

### 3. Re-integrate to absolute poses
Compose the smoothed incremental twists on SE(3) into an absolute pose trajectory
`G_t` (with `G_0 = I`): `G_{t+1} = G_t · exp(ξ̂_t^smooth)`. `G_t` is the cleaned
object pose at frame `t` relative to frame 0.

### 4. Re-synthesize point flow
Each track is re-placed by the **cleaned** object motion from its own birth frame,
anchored at its real observed birth position (so absolute scale/placement is kept and
error doesn't accumulate from frame 0 for late-born points):

```
t0 = first visible frame of track i
coords_smooth[t, i] = (G_t · G_{t0}^{-1}) · coords[t0, i]     for t in track i's visible span
                    = NaN                                       elsewhere
```

`vis` and `colors` are copied unchanged. Output arrays have identical shape/dtype to
the input pkl.

### 5. Outputs
- **`<out>/dense_3d_track.pkl`** — `{coords: smoothed, colors: unchanged, vis:
  unchanged}`, byte-compatible schema with the tracker's output.
- **`<out>/tracks_2d.mp4`** — 2D overlay of the smoothed tracks, reusing
  `track_windowed.py`'s `render_2d_overlay` + `rainbow_colors_by_position`.
  Pixel coords are obtained by **reprojecting** the smoothed 3D with the camera
  intrinsics: `u = fx·X/Z + cx`, `v = fy·Y/Z + cy` (points with `Z ≤ 0` or NaN are
  not drawn; verified 95.5% of frame-0 points reproject in-bounds).
  - Background: if `--video color.mp4` (or a frame folder) is provided, overlay on
    those RGB frames (matches the tracker's video). If omitted (e.g. `sphere_2/` has
    no `color.mp4`), draw on a **black canvas** at `--image-size W,H` (default
    1280×720). The pkl has no image dimensions, so size comes from the video when
    present else the flag.

## CLI

```
smooth_object_motion.py
  --pkl PATH              input dense_3d_track.pkl (required)
  --output-path DIR       output dir for smoothed pkl + mp4 (required)
  --smooth-window N       low-pass window in frames (default ~9; 1 = passthrough)
  --intrinsics fx,fy,cx,cy   default 771.59,771.365,645.555,349.653 (ZED native)
  --video PATH            optional color.mp4 / frame folder for the overlay background
  --image-size W,H        canvas size when --video is absent (default 1280,720)
  --min-corr N            min correspondences to fit a pair's pose (default 8)
  --ransac-thresh M       RANSAC inlier residual in metres (default 0.01)
  --fps F                 output mp4 fps (default 10, matching the tracker)
```

## Edge cases

- **Sparse/occluded frame pair** (< `--min-corr` inliers): twist gap-filled by
  interpolation; logged.
- **Degenerate geometry for rotation** (e.g. a stick — points near-collinear):
  Kabsch still returns the best rigid fit; the unresolved DOF (spin about the stick's
  axis) is simply not observable and stays near-identity — acceptable, not an error.
- **Track born mid-clip:** anchored at its own birth frame via `G_t · G_{t0}^{-1}`,
  so it's placed correctly regardless of when it appears.
- **`Z ≤ 0` after smoothing:** point not drawn in the 2D overlay (kept in the pkl as
  computed; such points are already `vis=False` in practice).
- **`--smooth-window 1`:** exact passthrough — smoothed pkl equals input (self-test).

## Testing / verification

1. **Unit — SE(3) round-trip:** `exp`/`log` and pose compose/invert are inverses to
   float tolerance; `--smooth-window 1` reproduces input coords bit-for-bit (identity
   path).
2. **Unit — synthetic rigid motion:** generate points under a known
   oscillating-plus-drift rotation; assert the recovered per-frame twist matches
   ground truth (residual ~0), and that smoothing reduces the **oscillation energy**
   while preserving **net rotation** (efficiency ratio rises toward 100%).
3. **Integration on `sphere_2`:** run with the default window; assert the smoothed
   signed-angular-velocity oscillation power drops and net rotation is preserved
   (within a few %), the pkl loads with identical shape/`vis`/`colors`, and
   `tracks_2d.mp4` is written. Sanity-view the overlay.
4. **Genuine-reversal preservation:** on a clip (or synthetic) with a real
   multi-second direction change, confirm the change survives smoothing (the slow
   trend is not flattened) — guards against over-smoothing intent.

## Out of scope

- Re-running the tracker or any change to `track_windowed.py` (this is a pure
  post-process reading its output).
- Per-part / articulated motion (assumes one rigid body per clip, per the capture
  setup).
- Auto-selecting `--smooth-window` (manual knob; the diagnostics make it easy to
  tune, and auto-tuning can come later if wanted).
- Saving the pose trajectory `G_t` as a separate artifact (the user asked for pkl +
  mp4 only; can be added later if the flow model wants it).
