# End-to-end mcap → object 3D tracking pipeline

**Date:** 2026-07-10
**Status:** Approved (ready for implementation plan)

## Goal

One command that takes a robot-capture `.mcap` file (ZED RGB-D on ROS 2 topics)
and produces object 3D tracking results plus a 2D visualization, replacing the
manual multi-repo workflow in `workflow.md`. Also produce compact, training-ready
RGB-D artifacts and minimize disk usage.

## Input

An mcap recording, e.g.
`/home/labeng/data/default_task/20260710_104834/episode_1/mcaps/recording_2026-07-10-14-49-28/recording_2026-07-10-14-49-28_0.mcap`

Relevant topics (verified by probing the file):

| Topic | ROS type | Encoding | Notes |
|---|---|---|---|
| `/dag/zed/compressed` | `sensor_msgs/msg/CompressedImage` | JPEG | 1280×720, 331 msgs, ~7.5 fps |
| `/dag/zed/depth` | `sensor_msgs/msg/Image` | `32FC1` | 1280×720, 331 msgs, **float32 metres**, invalid = **NaN** (~18%), range 0.12–4.77 m |

RGB and depth are 1:1 and time-aligned (nearest-depth offset ≤ 9 ms).

**Camera intrinsics** (ZED, native 1280×720): `fx,fy,cx,cy = 771.59, 771.365, 645.555, 349.653`
— identical to `track_windowed.py`'s existing default.

## Reused, unchanged components

- **`Track4World/scripts/run_dino_sam2.py`** — Grounded-DINO + SAM2 segmentation.
  Takes `--video-path` (mp4 or frame folder) and `--text-prompt`, writes
  `<out>/mask/{00000..}.png` and `<out>/vis/...`. Runs in the **track4world** env.
- **`preprocess/track_windowed.py`** — sparse windowed 3D tracker. Already consumes
  a color mp4, a depth source, and `{frame:05d}.png` masks, with ZED intrinsics /
  depth-scale as defaults. Writes `dense_3d_track.pkl` and `tracks_2d.mp4`. Runs in
  the **densetrack3d** env. Needs one small additive patch (depth-video loader).

## Environments

- **track4world** (`/home/labeng/miniconda3/envs/track4world`): has `rosbags`
  (pure-Python mcap reader, no ROS install needed) and OpenCV. Runs Stage 1 + 2.
- **densetrack3d** (`/home/labeng/miniconda3/envs/densetrack3d`): runs the model.
  Runs Stage 3. See memory `env-cuda-build-setup`.
- **ffmpeg 6.1.1** on system PATH, with `ffv1` encoder and `gray16le` pixfmt —
  required for lossless 16-bit depth video.

## Architecture & data flow

```
recording_*.mcap
   │  STAGE 1  preprocess/extract_mcap_rgbd.py        [track4world: rosbags + ffmpeg]
   ▼
<work>/color.mp4        RGB, JPEG-decoded → mp4              ┐ durable (later training)
<work>/depth.mkv        FFV1 gray16le, 16-bit mm, lossless  ┘
   │  STAGE 2  Track4World/scripts/run_dino_sam2.py  (UNCHANGED)  [track4world]
   │           reads color.mp4, --text-prompt <object>
   ▼
<work>/seg/mask/{00000..}.png   per-frame masks (transient — deleted after Stage 3)
   │  STAGE 3  preprocess/track_windowed.py  (PATCHED)  [densetrack3d]
   │           reads color.mp4 + depth.mkv + seg/mask
   ▼
<work>/track/dense_3d_track.pkl   tracking result (coords/colors/vis)
<work>/track/tracks_2d.mp4        2D overlay visualization
```

**No per-frame depth files are ever written.** Depth goes mcap → `depth.mkv` →
decoded in-memory by the tracker.

## Space optimization

Verified by round-trip test: 16-bit depth stored as **FFV1 / `gray16le`** video is
**bit-exact** on decode. imageio's high-level API and `cv2.VideoWriter`/`VideoCapture`
silently truncate 16-bit → 8-bit, so both write and read MUST go through a raw ffmpeg
pipe (`-f rawvideo -pix_fmt gray16le`).

Final disk footprint per capture:
- `color.mp4` — compact RGB (kept for training).
- `depth.mkv` — lossless 16-bit mm depth, ~10–30 MB (vs ~100–165 MB as a PNG folder).
- `seg/` (masks + vis) — **deleted** unconditionally after Stage 3 succeeds.
- `track/dense_3d_track.pkl` + `track/tracks_2d.mp4` — results.

Masks are deleted after tracking (not a CLI option). Re-running tracking after mask
deletion requires re-running Stage 2, which is acceptable.

## Components

### Stage 1 — `preprocess/extract_mcap_rgbd.py` (NEW, track4world)

Responsibility: mcap → `color.mp4` + `depth.mkv`.

1. Open the mcap's parent dir with `rosbags.highlevel.AnyReader`.
2. Collect `(timestamp, raw)` for the RGB topic and the depth topic.
3. Pair each RGB message with its **nearest-timestamp** depth message; iterate in
   RGB timestamp order → 0-based frame index `i`.
4. RGB: `cv2.imdecode(JPEG)` (BGR) → append to `color.mp4`.
5. Depth: `np.frombuffer(data, '<f4').reshape(H, W)` (metres) → `NaN`/`inf` → 0
   → `× depth_scale` (1000) → clip to uint16 → `uint16` mm → write raw `gray16le`
   bytes to an `ffmpeg -c:v ffv1 -level 3 -pix_fmt gray16le` subprocess → `depth.mkv`.
6. CLI: `--mcap` (required), `--output-dir` (required),
   `--rgb-topic` (default `/dag/zed/compressed`),
   `--depth-topic` (default `/dag/zed/depth`),
   `--depth-scale` (default 1000.0), `--fps` (default 7.5).
7. Print: frame count, resolution, valid-depth %, output sizes.

Notes:
- `color.mp4` is a lossy re-encode of already-lossy JPEG. Acceptable: RGB feeds only
  detection/segmentation and point coloring, not geometry. Documented in the docstring.
- Depth stays lossless (FFV1). This is the geometry source.
- The two topic streams have equal counts here; nearest-timestamp pairing is robust
  to unequal counts / drift. If counts differ, iterate over the RGB stream and match.

### Stage 3 patch — `preprocess/track_windowed.py` (densetrack3d)

Add an additive depth loader; do not change tracking/merge/viz logic.

- New behavior in the depth-loading section: if `--depth` is a file ending in
  `.mkv`/`.mp4`, decode it via an ffmpeg raw pipe
  (`ffmpeg -i <file> -f rawvideo -pix_fmt gray16le pipe:1`) into a `(T,H,W)` uint16
  array, then apply the existing `/ depth_scale` → metres conversion (zeros stay 0
  = invalid). If `--depth` is a directory, keep the current PNG-folder path.
- Frame indexing (`--start-frame`, `--num-frames`) applies to the decoded stack
  exactly as it does to the PNG list today.
- Everything else (window plan, predictor calls, NaN-padded merge, pkl + 2D mp4
  output) is untouched.

### Driver — `run_pipeline.sh` (NEW, repo root)

Single bash entry point.

- Required: `--mcap`, `--text-prompt`.
- Always processes the whole capture from the first frame — the driver does NOT
  expose `--start-frame`/`--num-frames`. (`track_windowed.py` keeps those args for
  standalone use; the driver just omits them so they default to full-range.)
- Passthrough to Stage 3: `--win`, `--stride`, `--grid-size` (forwarded verbatim).
- `--output-dir` (default under `results/`).
- Config variables at top: env names, Track4World repo path, SAM2/DINO checkpoints,
  intrinsics, depth-scale, fps.
- `set -euo pipefail`. Per-stage banners.
- Skip-if-exists guards keyed on each stage's primary output
  (`color.mp4`+`depth.mkv`, then masks, then pkl); `--force` redoes everything.
- Stage invocation:
  - Stage 1: `conda run -n track4world python preprocess/extract_mcap_rgbd.py ...`
  - Stage 2: `conda run -n track4world python <Track4World>/scripts/run_dino_sam2.py
    --video-path <work>/color.mp4 --text-prompt "<prompt>" --output-dir <work>/seg ...`
  - Stage 3: `conda run -n densetrack3d python preprocess/track_windowed.py
    --video <work>/color.mp4 --depth <work>/depth.mkv --mask-dir <work>/seg/mask
    --output-path <work>/track ...`
- After Stage 3 succeeds: `rm -rf <work>/seg` (delete the whole segmentation output —
  both `mask/` and `vis/`, which are only intermediates). This removes the masks as
  required and reclaims the visualization frames too.

## Error handling

- `set -euo pipefail`: any stage's nonzero exit aborts the pipeline with the failing
  stage's output visible.
- Empty detections: `run_dino_sam2.py` already writes empty masks; `track_windowed.py`
  already warns and skips windows whose grid points miss the mask. No new handling.
- Mask deletion happens ONLY after Stage 3 exits 0, so a tracking failure leaves masks
  in place for debugging / rerun.

## Testing / verification

1. Stage 1 standalone: run on the sample mcap; assert 331 frames in `color.mp4`,
   `depth.mkv` decodes to `(331,720,1280)` uint16, valid-depth % ≈ 82, sizes sane.
2. Depth round-trip: decode `depth.mkv` and confirm values match the source metres
   (mm-quantized) on a sampled frame.
3. Full pipeline: run `run_pipeline.sh` on the sample mcap with a text prompt;
   confirm `dense_3d_track.pkl` + `tracks_2d.mp4` exist and masks are gone.
4. Tracker patch parity: for a short clip, confirm the video-depth path yields the
   same tracks as the equivalent PNG-folder depth (bit-exact depth → identical result).

## Out of scope

- Cross-window identity stitching (intentionally absent; see `track_windowed.py` docstring).
- Multi-mcap batch orchestration (single file per run).
- Reading non-ZED topics (glove/hand/control topics ignored).
