#!/usr/bin/env bash
# End-to-end: ZED mcap -> object 3D tracking + compact RGB-D artifacts.
#
#   ./run_pipeline.sh --mcap <file.mcap> --text-prompt "red tape measure." \
#                     --output-dir results/measure [--win 15 --stride 5 --grid-size 80] \
#                     [--smooth-window 50] [--crop-width 640] [--side left] [--no-viz] [--force]
#
# Turns one mcap into a self-contained training folder (mcap no longer needed):
#   color.mp4  depth.mkv  intrinsics.txt  hand.pkl  object_flow.pkl  [object_flow_2d.mp4]
#
# Stage 1 (track4world): extract color.mp4 + lossless depth.mkv from the mcap,
#   horizontally center-cropped to --crop-width (keeps the interaction region only);
#   the crop-adjusted intrinsics are written to intrinsics.txt and used downstream.
# Stage 2 (track4world): extract the Manus glove readout + retargeted hand joints
#   (active --side, default left), aligned to the camera frames -> hand.pkl.
# Stage 3 (track4world): Grounded-DINO + SAM2 segmentation of the prompted object.
#   Only every STRIDE-th frame is segmented -- the windowed tracker seeds masks
#   only at its window starts (multiples of STRIDE), so intermediate masks would
#   be discarded. This cuts the pipeline's slowest stage ~STRIDE-fold. Only the
#   single highest-confidence detection is kept per frame (--max-detections 1),
#   since each clip tracks exactly one goal object. "hand. glove." are given as
#   negative prompts (--exclude-prompt) so the manipulator matches its own label
#   and is dropped -- otherwise an occluded object can seed tracks on the hand.
# Stage 4 (densetrack3d): windowed 3D tracking, reading depth straight from depth.mkv.
# Stage 5 (densetrack3d): trajectory smoothing (attenuate hand-jitter) -> object_flow.pkl.
# Intermediates (seg/ masks, track/ unsmoothed flow) are deleted after smoothing.
set -euo pipefail

# --- fixed config -----------------------------------------------------------
REPO="/home/labeng/yanghan/code/vision/DenseTrack3Dv2"
TRACK4WORLD_REPO="/home/labeng/yanghan/code/vision/Track4World"
SAM2_CKPT="checkpoints/sam2.1_hiera_large.pt"          # relative to TRACK4WORLD_REPO
INTRINSICS="771.59,771.365,645.555,349.653"            # ZED @ 1280x720 (source)
DEPTH_SCALE=1000.0
FPS=30.0

# --- args -------------------------------------------------------------------
# Defaults tuned for 30 fps ZED capture (validated on the 8cm-sphere clips):
# win 15 / stride 5 / grid 80 track cleanly; smooth-window 50 for object-flow.
# crop-width 640 (centered half of 1280) keeps the interaction region; 0 = no crop.
MCAP=""; PROMPT=""; OUTDIR=""; WIN=15; STRIDE=5; GRID=40; SMOOTH_WIN=50; CROP_WIDTH=640
SIDE=left; NOVIZ=0; FORCE=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --mcap)          MCAP="$2"; shift 2;;
    --text-prompt)   PROMPT="$2"; shift 2;;
    --output-dir)    OUTDIR="$2"; shift 2;;
    --win)           WIN="$2"; shift 2;;
    --stride)        STRIDE="$2"; shift 2;;
    --grid-size)     GRID="$2"; shift 2;;
    --smooth-window) SMOOTH_WIN="$2"; shift 2;;
    --crop-width)    CROP_WIDTH="$2"; shift 2;;
    --side)          SIDE="$2"; shift 2;;
    --no-viz)        NOVIZ=1; shift;;
    --force)         FORCE=1; shift;;
    *) echo "Unknown arg: $1" >&2; exit 2;;
  esac
done
[[ -z "$MCAP"   ]] && { echo "ERROR: --mcap required" >&2; exit 2; }
[[ -z "$PROMPT" ]] && { echo "ERROR: --text-prompt required" >&2; exit 2; }
[[ -z "$OUTDIR" ]] && { echo "ERROR: --output-dir required" >&2; exit 2; }
mkdir -p "$OUTDIR"
OUTDIR="$(cd "$OUTDIR" && pwd)"   # absolute: Stage 2 runs from a different cwd

banner() { echo; echo "======== $* ========"; }

# --- Stage 1: extract RGB-D -------------------------------------------------
banner "STAGE 1/5  extract RGB-D from mcap"
if [[ $FORCE -eq 1 || ! -f "$OUTDIR/color.mp4" || ! -f "$OUTDIR/depth.mkv" || ! -f "$OUTDIR/intrinsics.txt" ]]; then
  conda run -n track4world --no-capture-output python "$REPO/preprocess/extract_mcap_rgbd.py" \
    --mcap "$MCAP" --output-dir "$OUTDIR" --depth-scale "$DEPTH_SCALE" --fps "$FPS" \
    --intrinsics "$INTRINSICS" --crop-width "$CROP_WIDTH"
else
  echo "skip: color.mp4 + depth.mkv + intrinsics.txt already exist (use --force to redo)"
fi

# Effective intrinsics for the (cropped) frames -- written by the extractor. Fall
# back to the source value for older uncropped runs that predate intrinsics.txt.
INTRINSICS_EFF="$(cat "$OUTDIR/intrinsics.txt" 2>/dev/null || echo "$INTRINSICS")"
echo "using intrinsics: $INTRINSICS_EFF"

# --- Stage 2: extract hand data ---------------------------------------------
# Manus glove readout + retargeted hand joints, nearest-timestamp aligned to the
# camera frames (hand.pkl rows are index-aligned with the object flow).
banner "STAGE 2/5  extract hand data (side=$SIDE)"
if [[ $FORCE -eq 1 || ! -f "$OUTDIR/hand.pkl" ]]; then
  conda run -n track4world --no-capture-output python "$REPO/preprocess/extract_hand.py" \
    --mcap "$MCAP" --output-dir "$OUTDIR" --side "$SIDE"
else
  echo "skip: hand.pkl already exists (use --force to redo)"
fi

# --- Stage 3: segmentation --------------------------------------------------
banner "STAGE 3/5  segmentation ('$PROMPT')"
if [[ $FORCE -eq 1 || ! -d "$OUTDIR/seg/mask" ]]; then
  ( cd "$TRACK4WORLD_REPO" && conda run -n track4world --no-capture-output python scripts/run_dino_sam2.py \
      --video-path "$OUTDIR/color.mp4" \
      --text-prompt "$PROMPT" \
      --sam2-checkpoint "$SAM2_CKPT" \
      --output-dir "$OUTDIR/seg" \
      --frame-stride 1 \
      --max-detections 1 \
      --exclude-prompt "hand. glove." )
else
  echo "skip: $OUTDIR/seg/mask already exists (use --force to redo)"
fi

# --- Stage 4: tracking ------------------------------------------------------
# Writes the unsmoothed flow to the intermediate track/ dir; always --no-viz since
# that raw flow (and its overlay) is deleted after smoothing.
banner "STAGE 4/5  windowed 3D tracking"
if [[ $FORCE -eq 1 || ! -f "$OUTDIR/track/dense_3d_track.pkl" ]]; then
  conda run -n densetrack3d --no-capture-output python "$REPO/preprocess/track_windowed.py" \
    --video "$OUTDIR/color.mp4" \
    --depth "$OUTDIR/depth.mkv" \
    --mask-dir "$OUTDIR/seg/mask" \
    --output-path "$OUTDIR/track" \
    --intrinsics "$INTRINSICS_EFF" \
    --depth-scale "$DEPTH_SCALE" \
    --num-frames -1 \
    --win "$WIN" --stride "$STRIDE" --grid-size "$GRID" \
    --no-viz
else
  echo "skip: track/dense_3d_track.pkl already exists (use --force to redo)"
fi

# --- Stage 5: trajectory smoothing ------------------------------------------
# Attenuate hand finger-jitter in the object flow by low-passing each point's 3D
# trajectory. Writes object_flow.pkl at the top level (+ object_flow_2d.mp4 unless
# --no-viz). --fps drives the output video only.
banner "STAGE 5/5  trajectory smoothing (window $SMOOTH_WIN)"
if [[ $FORCE -eq 1 || ! -f "$OUTDIR/object_flow.pkl" ]]; then
  VIZ_ARG=(); [[ $NOVIZ -eq 1 ]] && VIZ_ARG=(--no-viz)
  conda run -n densetrack3d --no-capture-output python "$REPO/postprocess/smooth_trajectory.py" \
    --pkl "$OUTDIR/track/dense_3d_track.pkl" \
    --output-path "$OUTDIR" \
    --output-name object_flow \
    --video "$OUTDIR/color.mp4" \
    --intrinsics "$INTRINSICS_EFF" \
    --smooth-window "$SMOOTH_WIN" \
    --fps "${FPS%.*}" \
    "${VIZ_ARG[@]}"
else
  echo "skip: object_flow.pkl already exists (use --force to redo)"
fi

# --- cleanup: unsmoothed flow is an intermediate ----------------------------
# Keep seg/ (object masks): the world-model label loader reads them to build the
# object point cloud, so re-segmenting at load time is avoided.
banner "cleanup: removing intermediates (track/)"
rm -rf "$OUTDIR/track"

banner "DONE"
echo "Artifacts in $OUTDIR:"
echo "  color.mp4  depth.mkv  intrinsics.txt"
echo "  hand.pkl                          (glove readout + retargeted joints, side=$SIDE)"
echo "  object_flow.pkl                   (smoothed object 3D flow, win $SMOOTH_WIN)"
[[ $NOVIZ -eq 1 ]] || echo "  object_flow_2d.mp4                (2D overlay)"
