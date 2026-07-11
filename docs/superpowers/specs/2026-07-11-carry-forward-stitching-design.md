# Seam-free carry-forward stitching for windowed 3D tracking

**Date:** 2026-07-11
**Status:** Approved (ready for implementation plan)
**Target file:** `preprocess/track_windowed.py`

## Goal

Produce **long-horizon 3D object trajectories** from the windowed sparse tracker,
so the output reflects the object's overall motion instead of per-window fragments
that reset every `stride` frames. These trajectories are training data for an
object-flow model (`object_flow_models_plan.md`), so a stitched identity must be
the *same physical surface point* across handoffs — a wrong or approximate link
would inject fake motion, which is worse than a short honest track.

## Problem with the current behavior

`track_windowed.py` today runs each overlapping window as an independent predictor
call (seed a `grid_size × grid_size` grid ∩ the window-start mask, track forward),
then merges by **pure union along the track axis** (`np.concatenate`, NaN-padded
outside each window). There is no identity linking: a physical point seen in
windows 0, 1, 2 becomes three separate ≤`win`-frame tracks. Long-horizon motion is
therefore never represented — only short shaking segments are.

## Approach: carry-forward seeding (query injection)

Every window still re-seeds fresh, so the immediately-previous window always holds
the least-drifted copy of each currently-visible object point. We exploit that:

- **Continue** an identity by injecting *its own* seam-frame position as an explicit
  query in the next window. The predictor overwrites its prediction at the query
  frame with the exact query position (`predictor.py:190-192`), so the continued
  segment starts **exactly** where the previous one was — **seam-free**, no matching
  guesswork.
- **Add** genuinely new surface (revealed as the object rotates) by seeding fresh
  grid∩mask points that are not near any carried-forward point.

Because each frame's value still comes from a window seeded ≤`win` frames earlier,
this yields long identities **while preserving the bounded-drift property** that
motivated the original no-stitch design. Identities are **unbounded**: a point
tracked from frame 0 to the end stays a single identity.

### Why this is low-risk for training data

There is **no cross-track nearest-neighbor matching** in the identity path. We do
not guess which new grid point corresponds to an old track; we continue each
previous point by injecting its own coordinates. The only proximity decision is
deduping *new* grid points against carried points — this is cosmetic (avoids
redundant tracks) and never affects an existing identity's trajectory.

## Enabling facts (verified in the current code)

- `Predictor3D.forward` accepts explicit `queries` of shape `(B, N, 3)` =
  `(frame, x_native, y_native)` (`predictor.py:112-121`); it samples each query's
  depth internally (`:147-155`) and overwrites the query-frame prediction with the
  query position (`:190-192`).
- Seeding all queries at local frame 0 means per-track **colors** are sampled from
  that frame (`predictor.py:204`, `model_utils.py:668`) — same as today.
- The grid∩mask construction we replicate for fresh points is
  `get_points_on_a_grid(grid_size, interp_shape)` intersected with the interpolated
  mask (`predictor.py:124-131`). We build the equivalent at native resolution and
  pass explicit queries, so the fresh-point set matches today's seeding.
- 3D coords are camera-centric per frame, so placing a segment at absolute frame
  indices is still just indexing — no cross-window transform needed.

## Algorithm

Windows are planned exactly as today (`starts` by `stride`, final window clamped to
cover the tail). Process windows in order, maintaining a table of **active
identities**. Each identity holds: a persistent integer id, its seed color, and its
per-frame `coords`/`uv`/`vis` filled in as windows extend it.

For window `w` with local range `[s, e)` and absolute start frame `f = abs_frames[s]`:

1. **Carry-forward queries.** From the **immediately previous window's** tracks,
   select those that are **visible at frame `f`** AND have **valid depth** at `f`
   in this window's depth stack (depth > 0). For each, inject query
   `(0, x_f, y_f)` at this window's local frame 0, retaining its persistent id.
   (Chaining: these carried points already transitively represent all older
   identities still alive.)
2. **Fresh queries.** Build `grid_size × grid_size` grid ∩ this window's start-mask
   (native resolution, matching `predictor.py:124-131`). Drop any grid point within
   `--merge-radius` px of a carried-forward point. Survivors get **new** persistent
   ids — this is the new-surface source.
3. **Predict.** Concatenate `queries = [carried ; fresh]` (all at local frame 0) and
   call `predictor(vid_w, dep_w, queries=queries, segm_mask=None, grid_query_frame=0,
   backward_tracking=False, predefined_intrs=K)`. Autocast/dtype context unchanged.
4. **Split & assign.** The first `len(carried)` output columns **extend** existing
   identities (their `coords/uv/vis` over `[s,e)` are written, overwriting the
   previous window's values on the overlap — the fresher re-anchor wins). The
   remaining columns create **new** identities. Apply the existing first-occlusion
   cutoff (`np.logical_and.accumulate`, unless `--keep-reappearing`) per segment
   before writing.

### Stitching into final trajectories

Because a later window's write overwrites the overlap, each identity's value at
frame `t` naturally comes from the **freshest window with seed ≤ `t` that tracked
it**. After all windows, each identity is one continuous trajectory. Assemble:

- `coords (T, N_ids, 3)` float32 — NaN where an identity is inactive
- `colors (N_ids, 3)` float32 — 0-255 RGB from each identity's *first* seed frame
- `vis (T, N_ids)` bool — True only where model-visible (post-cutoff)
- `uv (T, N_ids, 2)` — for the 2D overlay (internal; not in the pkl)

This is **drop-in identical** in shape and semantics to today's pkl
(`coords`/`colors`/`vis`), so both visualizers and downstream training keep working
unchanged — only `N_ids` (few long identities) replaces the old large fragment
count. `tracks_2d.mp4` rendering (`render_2d_overlay`, `rainbow_colors_by_position`)
is unchanged.

## Edge cases

- **Carried point invalid at seam** (invisible or depth ≤ 0): not carried; the
  identity ends cleanly at the previous window's last visible frame.
- **First window:** no previous window → no carries, all fresh. Identical to
  current behavior.
- **Window with zero queries** (empty mask AND no carries): skipped with a warning,
  as today (`track_windowed.py:351-353`).
- **`--no-stitch`:** bypass carry-forward entirely and reproduce today's pure-union
  output exactly (for A/B comparison and regression safety).

## CLI additions

- `--stitch` / `--no-stitch` — stitching **on by default** (the file's purpose is
  long-horizon flow); `--no-stitch` restores pure-union behavior.
- `--merge-radius` (float px, default = one grid-cell spacing `≈ W / grid_size`) —
  dedup distance for new grid points vs carried points.

No other CLI or output changes. `--win`, `--stride`, `--grid-size`, `--intrinsics`,
`--depth-scale`, frame-range args, and `--keep-reappearing` keep their meaning.

## Testing / verification

1. **Unit — identity bookkeeping:** a synthetic 3-window case (hand-built predictor
   outputs) asserting: carried ids persist across windows, new ids appear for
   un-deduped fresh points, and each id's final trajectory takes values from the
   freshest window covering each frame.
2. **Integration:** run on the sample capture; assert `N_ids` ≪ the pure-union track
   count, and that carried segments are **seam-continuous** (an identity's position
   at a seam frame matches across the handoff to within depth-sampling error, since
   the query position is injected exactly). Confirm the pkl loads in the visualizer.
3. **A/B regression:** `--no-stitch` reproduces the current output (same track count
   and values) — guards the refactor.

## Out of scope

- Cross-track nearest-neighbor identity matching (deliberately avoided; carry-forward
  injects a point's own position instead of guessing correspondences).
- Length-capping / periodic re-id of identities (identities are unbounded by choice).
- Any change to the mcap→RGBD→segmentation pipeline or the predictor/model internals.
