#!/usr/bin/env bash
# End-to-end: ZED mcap -> object 3D tracking + compact RGB-D artifacts.
#
#   ./run_pipeline.sh --mcap <file.mcap> --text-prompt "red tape measure." \
#                     --output-dir results/measure [--win 15 --stride 5 --grid-size 80] \
#                     [--smooth-window 50] [--crop-width 428] [--force]
#
# Stage 1 (track4world): extract color.mp4 + lossless depth.mkv from the mcap,
#   horizontally center-cropped to --crop-width (keeps the interaction region only);
#   the crop-adjusted intrinsics are written to intrinsics.txt and used downstream.
# Stage 2 (track4world): Grounded-DINO + SAM2 segmentation of the prompted object.
#   Only every STRIDE-th frame is segmented -- the windowed tracker seeds masks
#   only at its window starts (multiples of STRIDE), so intermediate masks would
#   be discarded. This cuts the pipeline's slowest stage ~STRIDE-fold.
# Stage 3 (densetrack3d): windowed 3D tracking, reading depth straight from depth.mkv.
# Stage 4 (densetrack3d): trajectory smoothing (attenuate hand-jitter in object flow).
# Masks (results/<name>/seg) are deleted after tracking succeeds.
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
# crop-width 428 (~1/3 of 1280) keeps the centered interaction region; 0 = no crop.
MCAP=""; PROMPT=""; OUTDIR=""; WIN=15; STRIDE=5; GRID=40; SMOOTH_WIN=50; CROP_WIDTH=428; FORCE=0
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
banner "STAGE 1/4  extract RGB-D from mcap"
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

# --- Stage 2: segmentation --------------------------------------------------
banner "STAGE 2/4  segmentation ('$PROMPT')"
if [[ $FORCE -eq 1 || ! -d "$OUTDIR/seg/mask" ]]; then
  ( cd "$TRACK4WORLD_REPO" && conda run -n track4world --no-capture-output python scripts/run_dino_sam2.py \
      --video-path "$OUTDIR/color.mp4" \
      --text-prompt "$PROMPT" \
      --sam2-checkpoint "$SAM2_CKPT" \
      --output-dir "$OUTDIR/seg" \
      --frame-stride "$STRIDE" )
else
  echo "skip: $OUTDIR/seg/mask already exists (use --force to redo)"
fi

# --- Stage 3: tracking ------------------------------------------------------
banner "STAGE 3/4  windowed 3D tracking"
if [[ $FORCE -eq 1 || ! -f "$OUTDIR/track/dense_3d_track.pkl" ]]; then
  conda run -n densetrack3d --no-capture-output python "$REPO/preprocess/track_windowed.py" \
    --video "$OUTDIR/color.mp4" \
    --depth "$OUTDIR/depth.mkv" \
    --mask-dir "$OUTDIR/seg/mask" \
    --output-path "$OUTDIR/track" \
    --intrinsics "$INTRINSICS_EFF" \
    --depth-scale "$DEPTH_SCALE" \
    --num-frames -1 \
    --win "$WIN" --stride "$STRIDE" --grid-size "$GRID"
else
  echo "skip: track/dense_3d_track.pkl already exists (use --force to redo)"
fi

# --- Stage 4: trajectory smoothing ------------------------------------------
# Attenuate hand finger-jitter in the object flow by low-passing each point's 3D
# trajectory. Overlays on color.mp4; --fps drives the output video only.
banner "STAGE 4/4  trajectory smoothing (window $SMOOTH_WIN)"
if [[ $FORCE -eq 1 || ! -f "$OUTDIR/track_smoothed/dense_3d_track.pkl" ]]; then
  conda run -n densetrack3d --no-capture-output python "$REPO/postprocess/smooth_trajectory.py" \
    --pkl "$OUTDIR/track/dense_3d_track.pkl" \
    --output-path "$OUTDIR/track_smoothed" \
    --video "$OUTDIR/color.mp4" \
    --intrinsics "$INTRINSICS_EFF" \
    --smooth-window "$SMOOTH_WIN" \
    --fps "${FPS%.*}"
else
  echo "skip: track_smoothed/dense_3d_track.pkl already exists (use --force to redo)"
fi

# --- cleanup: masks/vis are intermediates ----------------------------------
banner "cleanup: removing segmentation intermediates"
rm -rf "$OUTDIR/seg"

banner "DONE"
echo "Artifacts in $OUTDIR:"
echo "  color.mp4  depth.mkv"
echo "  track/dense_3d_track.pkl           track/tracks_2d.mp4            (raw tracks)"
echo "  track_smoothed/dense_3d_track.pkl  track_smoothed/tracks_2d.mp4  (smoothed, win $SMOOTH_WIN)"
