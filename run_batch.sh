#!/usr/bin/env bash
# Batch driver: run run_pipeline.sh over every episode in one data folder.
#
#   ./run_batch.sh --data-dir ~/data/inhand_manipulation/0718_cube_small \
#                  --text-prompt "A blue and orange checkered cube." \
#                  --output-dir results/0718_cube_small [pipeline flags...]
#
#   ./run_batch.sh --data-dir ~/data/inhand_manipulation/0718_sphere_small \
#                  --text-prompt "A blue and orange checkered sphere." \
#                  --output-dir results/0718_sphere_small
#
# Layout assumed under --data-dir:
#   episode_<N>/mcaps/recording_.../recording_..._0.mcap
# Per-episode artifacts land in <output-dir>/episode_<N>/ (same set run_pipeline.sh
# produces: color.mp4, depth.mkv, intrinsics.txt, hand.pkl, object_flow.pkl, ...).
#
# One prompt per folder (all episodes share the same target object). Failures are
# logged and skipped so the batch runs unattended; a summary is printed at the end.
# Episodes whose object_flow.pkl already exists are skipped (pass --force to redo).
# Any flag other than the three below is forwarded verbatim to run_pipeline.sh
# (--win --stride --grid-size --smooth-window --crop-width --side --no-viz --force).
set -euo pipefail

REPO="/home/labeng/yanghan/code/vision/DenseTrack3Dv2"

# --- args -------------------------------------------------------------------
DATA_DIR=""; PROMPT=""; OUTDIR=""; FORCE=0
PASS_THRU=()   # extra flags forwarded to run_pipeline.sh
while [[ $# -gt 0 ]]; do
  case "$1" in
    --data-dir)    DATA_DIR="$2"; shift 2;;
    --text-prompt) PROMPT="$2"; shift 2;;
    --output-dir)  OUTDIR="$2"; shift 2;;
    --force)       FORCE=1; PASS_THRU+=("$1"); shift;;          # forwarded AND honored here
    --no-viz)      PASS_THRU+=("$1"); shift;;                   # valueless flags
    --win|--stride|--grid-size|--smooth-window|--crop-width|--side)
                   PASS_THRU+=("$1" "$2"); shift 2;;            # value flags
    *) echo "Unknown arg: $1" >&2; exit 2;;
  esac
done
[[ -z "$DATA_DIR" ]] && { echo "ERROR: --data-dir required" >&2; exit 2; }
[[ -z "$PROMPT"   ]] && { echo "ERROR: --text-prompt required" >&2; exit 2; }
[[ -z "$OUTDIR"   ]] && { echo "ERROR: --output-dir required" >&2; exit 2; }
[[ -d "$DATA_DIR" ]] || { echo "ERROR: --data-dir not found: $DATA_DIR" >&2; exit 2; }
mkdir -p "$OUTDIR"
OUTDIR="$(cd "$OUTDIR" && pwd)"   # absolute
LOG="$OUTDIR/batch.log"

banner() { echo; echo "########## $* ##########"; }
log() { echo "$*" | tee -a "$LOG"; }

# --- discover episodes ------------------------------------------------------
# episode_<N> dirs sorted numerically by <N> (episode_1, episode_2, ... episode_10).
mapfile -t EP_DIRS < <(
  find "$DATA_DIR" -maxdepth 1 -mindepth 1 -type d -name 'episode_*' -printf '%f\n' \
    | sort -t_ -k2,2n
)
[[ ${#EP_DIRS[@]} -eq 0 ]] && { echo "ERROR: no episode_* dirs under $DATA_DIR" >&2; exit 2; }

banner "BATCH  ${#EP_DIRS[@]} episodes  |  prompt: '$PROMPT'"
log "== batch start: $(date '+%Y-%m-%d %H:%M:%S') =="
log "data-dir:   $DATA_DIR"
log "output-dir: $OUTDIR"
log "prompt:     $PROMPT"
log "extra:      ${PASS_THRU[*]:-(none)}"

PROCESSED=(); SKIPPED=(); FAILED=(); NOMCAP=()

for ep in "${EP_DIRS[@]}"; do
  ep_out="$OUTDIR/$ep"

  # locate the single mcap for this episode
  mapfile -t mcaps < <(find "$DATA_DIR/$ep/mcaps" -name '*.mcap' 2>/dev/null | sort)
  if [[ ${#mcaps[@]} -eq 0 ]]; then
    log "SKIP  $ep  (no .mcap found)"; NOMCAP+=("$ep"); continue
  fi
  if [[ ${#mcaps[@]} -gt 1 ]]; then
    log "WARN  $ep  (${#mcaps[@]} mcaps found; using first: ${mcaps[0]##*/})"
  fi
  mcap="${mcaps[0]}"

  # resume: skip finished episodes unless --force
  if [[ $FORCE -eq 0 && -f "$ep_out/object_flow.pkl" ]]; then
    log "SKIP  $ep  (object_flow.pkl exists)"; SKIPPED+=("$ep"); continue
  fi

  banner "EPISODE $ep"
  log "-- $ep  ->  $ep_out  ($(date '+%H:%M:%S'))"
  if "$REPO/run_pipeline.sh" \
        --mcap "$mcap" \
        --text-prompt "$PROMPT" \
        --output-dir "$ep_out" \
        "${PASS_THRU[@]}"; then
    log "OK    $ep"; PROCESSED+=("$ep")
  else
    log "FAILED $ep  (exit $?)"; FAILED+=("$ep")
  fi
done

# --- summary ----------------------------------------------------------------
banner "BATCH SUMMARY"
log "processed: ${#PROCESSED[@]}   skipped: ${#SKIPPED[@]}   no-mcap: ${#NOMCAP[@]}   failed: ${#FAILED[@]}"
[[ ${#FAILED[@]} -gt 0 ]] && log "failed episodes: ${FAILED[*]}"
[[ ${#NOMCAP[@]} -gt 0 ]] && log "no-mcap episodes: ${NOMCAP[*]}"
log "== batch end: $(date '+%Y-%m-%d %H:%M:%S') =="

# non-zero exit if anything failed, so callers/CI notice
[[ ${#FAILED[@]} -gt 0 ]] && exit 1 || exit 0
