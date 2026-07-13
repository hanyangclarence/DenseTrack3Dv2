#!/usr/bin/env python3
"""Attenuate hand-jitter by low-passing each point's 3D trajectory directly.

Sibling of smooth_object_motion.py (which smooths the recovered rigid OBJECT POSE
and re-grounds points). That pose-chain route proved fragile: the raw and smoothed
absolute-pose chains (composed over ~1300 frames) drift apart, so the per-frame
correction R_sm @ R_raw^-1 is itself shaky and can ADD rotational jitter.

This script takes the simpler, robust route: filter each tracked point's world-space
3D trajectory over time with a zero-phase temporal Gaussian, using the SAME kernel
for every point. Because the smoothing is a shared linear operator:
  - high-frequency finger-jitter (fast curvature in every point's path) is removed;
  - the deliberate low-frequency rotation/translation survives, and a genuine
    multi-second direction change (also low-frequency) is preserved;
  - the object stays ~rigid -- a slow rigid motion traces smooth arcs, and low-
    passing an arc barely changes point-to-point distances (measured ~1% on a
    handheld sphere clip). It is APPROXIMATELY rigid, not exact, which is the one
    trade vs the pose-based method.

There is no pose integration, so there is no accumulated drift, no camera-vs-centroid
lever-arm error, and no birth-frame transport -- the failure modes that plagued the
pose route simply don't exist here.

Bad-depth tracks (which would fly) are rejected first with the same shape-agnostic
speed / rigidity-residual criteria as smooth_object_motion.py.

Output is a drop-in dense_3d_track.pkl (smoothed coords, same colors, vis of kept
tracks) plus a tracks_2d.mp4 overlay.
"""
import argparse
import os
import pickle
import sys

import cv2  # noqa: F401  (kept for parity / potential frame IO)
import mediapy as media
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# reuse the shared, tested helpers -- do not duplicate them
from postprocess.smooth_object_motion import (
    gaussian_kernel,
    smooth_zero_phase,
    recover_and_smooth,
    reject_bad_tracks,
    reproject,
    load_background,
)
from preprocess.track_windowed import render_2d_overlay, rainbow_colors_by_position


def smooth_trajectories(coords, vis, window, keep=None):
    """Low-pass each point's 3D world trajectory with a zero-phase Gaussian.

    coords (T,N,3), vis (T,N) bool. For each track, the smoothing runs over its
    contiguous visible span; interior occlusion gaps are linearly interpolated so
    the filter sees a continuous signal, then only originally-visible frames are
    written back (gap frames stay NaN / invisible). A 1-frame track is passed
    through unchanged. `keep` (N,) optionally drops outlier tracks (left NaN).

    Returns smoothed coords (T,N,3), NaN where a point is inactive / dropped.
    """
    T, N, _ = coords.shape
    kernel = gaussian_kernel(window)
    out = np.full_like(coords, np.nan)
    for i in range(N):
        if keep is not None and not keep[i]:
            continue
        fr = np.where(vis[:, i] & np.isfinite(coords[:, i]).all(axis=1))[0]
        if fr.size == 0:
            continue
        if fr.size == 1:
            out[fr, i] = coords[fr, i]
            continue
        span = np.arange(fr[0], fr[-1] + 1)                       # contiguous timeline
        traj = np.stack([np.interp(span, fr, coords[fr, i, j]) for j in range(3)], axis=1)
        sm = np.stack([smooth_zero_phase(traj[:, j], kernel) for j in range(3)], axis=1)
        mask = np.isin(span, fr)                                  # keep only real observations
        out[span[mask], i] = sm[mask]
    return out


def _rotational_travel(coords, vis):
    """Total frame-to-frame object rotation (deg) via Kabsch on common points --
    the shake metric. Lower = smoother. Imported lazily to avoid a hard dep here."""
    from postprocess.smooth_object_motion import kabsch
    T = coords.shape[0]
    tot = 0.0
    for f in range(T - 1):
        idx = np.where(vis[f] & vis[f + 1]
                       & np.isfinite(coords[f]).all(1) & np.isfinite(coords[f + 1]).all(1))[0]
        if idx.size < 8:
            continue
        R, _ = kabsch(coords[f, idx], coords[f + 1, idx])
        tot += np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1.0, 1.0)))
    return tot


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--pkl", required=True, help="input dense_3d_track.pkl")
    p.add_argument("--output-path", required=True, help="output dir for smoothed pkl + mp4")
    p.add_argument("--smooth-window", type=int, default=21,
                   help="temporal Gaussian window in frames (larger = smoother; 1 = passthrough)")
    p.add_argument("--intrinsics", default="771.59,771.365,645.555,349.653", help="fx,fy,cx,cy")
    p.add_argument("--video", default=None, help="optional color.mp4 / frame folder for overlay background")
    p.add_argument("--image-size", default="1280,720", help="canvas W,H when --video absent")
    p.add_argument("--min-corr", type=int, default=8, help="min correspondences for pose fit (rejection only)")
    p.add_argument("--ransac-thresh", type=float, default=0.01, help="RANSAC inlier residual (metres)")
    p.add_argument("--fps", type=int, default=10, help="output mp4 fps")
    p.add_argument("--reject-speed-k", type=float, default=6.0,
                   help="drop tracks whose p95 inter-frame speed exceeds median + k*MAD (shape-agnostic)")
    p.add_argument("--reject-resid-k", type=float, default=6.0,
                   help="drop tracks whose rigid-motion residual exceeds median + k*MAD (shape-agnostic)")
    p.add_argument("--no-reject", action="store_true", help="keep all tracks (skip outlier rejection)")
    return p.parse_args()


def main():
    args = parse_args()
    with open(args.pkl, "rb") as f:
        data = pickle.load(f)
    coords = data["coords"].astype(np.float64)
    vis = data["vis"].astype(bool)
    colors = data["colors"]
    T, N, _ = coords.shape
    print(f"Loaded {args.pkl}: coords {coords.shape}, {N} tracks over {T} frames")

    K = tuple(float(x) for x in args.intrinsics.split(","))

    # Shape-agnostic outlier rejection (needs the raw integrated pose only for the
    # rigidity residual; the smoothing itself does NOT use any pose).
    if args.no_reject:
        keep = np.ones(N, dtype=bool)
    else:
        _, diag = recover_and_smooth(coords, vis, args.smooth_window, args.min_corr, args.ransac_thresh)
        keep, rstats = reject_bad_tracks(coords, vis, diag["G_raw"],
                                         args.reject_speed_k, args.reject_resid_k)
        print(f"Outlier rejection: kept {rstats['n_keep']}/{rstats['n_total']} tracks "
              f"(dropped {rstats['n_drop']}, {100 * rstats['n_drop'] / rstats['n_total']:.0f}%; "
              f"speed>{rstats['speed_cut']:.3f} m/f or resid>{rstats['resid_cut']:.3f} m)")
    vis_keep = vis & keep[None, :]

    coords_sm = smooth_trajectories(coords, vis, args.smooth_window, keep=keep).astype(np.float32)

    # shake metric: rotational travel before/after (lower = smoother)
    tr_raw = _rotational_travel(coords, vis_keep)
    tr_sm = _rotational_travel(coords_sm.astype(np.float64), vis_keep)
    print(f"  rotational travel: raw {tr_raw:8.1f} deg -> smoothed {tr_sm:8.1f} deg "
          f"(shake cut {(1 - tr_sm / max(tr_raw, 1e-9)) * 100:.0f}%)")

    os.makedirs(args.output_path, exist_ok=True)
    out_pkl = os.path.join(args.output_path, "dense_3d_track.pkl")
    with open(out_pkl, "wb") as h:
        pickle.dump({"coords": coords_sm, "colors": colors, "vis": vis_keep}, h,
                    protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Saved {out_pkl}")

    W, H = (int(x) for x in args.image_size.split(","))
    bg = load_background(args.video, T, H, W)
    H, W = bg.shape[1:3]
    uv = reproject(coords_sm, vis_keep, K)
    viz_colors = rainbow_colors_by_position(uv, vis_keep)
    vid = render_2d_overlay(bg, uv, vis_keep, viz_colors, trace=8)
    out_mp4 = os.path.join(args.output_path, "tracks_2d.mp4")
    media.write_video(out_mp4, vid, fps=args.fps)
    print(f"Saved {out_mp4}")


if __name__ == "__main__":
    main()
