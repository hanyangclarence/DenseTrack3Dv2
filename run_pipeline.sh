#!/usr/bin/env bash
# End-to-end: ZED mcap -> object 3D tracking + compact RGB-D artifacts.
#
#   ./run_pipeline.sh --mcap <file.mcap> --text-prompt "red tape measure." \
#                     --output-dir results/measure [--win 20 --stride 5 --grid-size 40] [--force]
#
# Stage 1 (track4world): extract color.mp4 + lossless depth.mkv from the mcap.
# Stage 2 (track4world): Grounded-DINO + SAM2 segmentation of the prompted object.
# Stage 3 (densetrack3d): windowed 3D tracking, reading depth straight from depth.mkv.
# Masks (results/<name>/seg) are deleted after tracking succeeds.
set -euo pipefail

# --- fixed config -----------------------------------------------------------
REPO="/home/labeng/yanghan/code/vision/DenseTrack3Dv2"
TRACK4WORLD_REPO="/home/labeng/yanghan/code/vision/Track4World"
SAM2_CKPT="checkpoints/sam2.1_hiera_large.pt"          # relative to TRACK4WORLD_REPO
INTRINSICS="771.59,771.365,645.555,349.653"            # ZED @ 1280x720
DEPTH_SCALE=1000.0
FPS=7.5

# --- args -------------------------------------------------------------------
MCAP=""; PROMPT=""; OUTDIR=""; WIN=20; STRIDE=5; GRID=40; FORCE=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --mcap)        MCAP="$2"; shift 2;;
    --text-prompt) PROMPT="$2"; shift 2;;
    --output-dir)  OUTDIR="$2"; shift 2;;
    --win)         WIN="$2"; shift 2;;
    --stride)      STRIDE="$2"; shift 2;;
    --grid-size)   GRID="$2"; shift 2;;
    --force)       FORCE=1; shift;;
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
banner "STAGE 1/3  extract RGB-D from mcap"
if [[ $FORCE -eq 1 || ! -f "$OUTDIR/color.mp4" || ! -f "$OUTDIR/depth.mkv" ]]; then
  conda run -n track4world --no-capture-output python "$REPO/preprocess/extract_mcap_rgbd.py" \
    --mcap "$MCAP" --output-dir "$OUTDIR" --depth-scale "$DEPTH_SCALE" --fps "$FPS"
else
  echo "skip: color.mp4 + depth.mkv already exist (use --force to redo)"
fi

# --- Stage 2: segmentation --------------------------------------------------
banner "STAGE 2/3  segmentation ('$PROMPT')"
if [[ $FORCE -eq 1 || ! -d "$OUTDIR/seg/mask" ]]; then
  ( cd "$TRACK4WORLD_REPO" && conda run -n track4world --no-capture-output python scripts/run_dino_sam2.py \
      --video-path "$OUTDIR/color.mp4" \
      --text-prompt "$PROMPT" \
      --sam2-checkpoint "$SAM2_CKPT" \
      --output-dir "$OUTDIR/seg" )
else
  echo "skip: $OUTDIR/seg/mask already exists (use --force to redo)"
fi

# --- Stage 3: tracking ------------------------------------------------------
banner "STAGE 3/3  windowed 3D tracking"
if [[ $FORCE -eq 1 || ! -f "$OUTDIR/track/dense_3d_track.pkl" ]]; then
  conda run -n densetrack3d --no-capture-output python "$REPO/preprocess/track_windowed.py" \
    --video "$OUTDIR/color.mp4" \
    --depth "$OUTDIR/depth.mkv" \
    --mask-dir "$OUTDIR/seg/mask" \
    --output-path "$OUTDIR/track" \
    --intrinsics "$INTRINSICS" \
    --depth-scale "$DEPTH_SCALE" \
    --num-frames -1 \
    --win "$WIN" --stride "$STRIDE" --grid-size "$GRID"
else
  echo "skip: track/dense_3d_track.pkl already exists (use --force to redo)"
fi

# --- cleanup: masks/vis are intermediates ----------------------------------
banner "cleanup: removing segmentation intermediates"
rm -rf "$OUTDIR/seg"

banner "DONE"
echo "Artifacts in $OUTDIR:"
echo "  color.mp4  depth.mkv  track/dense_3d_track.pkl  track/tracks_2d.mp4"
