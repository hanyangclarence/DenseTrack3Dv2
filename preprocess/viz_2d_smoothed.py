#!/usr/bin/env python3
"""Standalone 2D trajectory overlay for a windowed dense_3d_track.pkl, with the
SAME temporal smoothing as visualizer/vis_densetrack3d_trails.py.

Why a separate script: track_windowed.py already renders a 2D overlay, but it
draws the raw per-frame pixel tracks (no smoothing) and throws the 2D coords
away. The saved pkl only holds 3D coords/colors/vis, so here we:

  1. load coords (T,N,3) / vis (T,N) from the pkl,
  2. smooth the 3D trajectories over continuous visible segments (Gaussian,
     pure-numpy -- identical algorithm to the trails viewer),
  3. re-project the smoothed 3D points to pixels with the camera intrinsics
     (u = fx*X/Z + cx, v = fy*Y/Z + cy -- the exact inverse of how the tracker
     lifted uv+depth to xyz), and
  4. draw fading trails over the RGB video, colored by seed position.

Because smoothing happens in 3D before projection, the 2D trails inherit it.

Frame alignment: the pkl was produced over absolute frames [start, start+num)
of the video (see track_windowed.py). Pass the SAME --start-frame / --num-frames
so the overlay lands on the right RGB frames.
"""
import argparse
import glob
import os
import pickle
import sys

import cv2
import matplotlib.pyplot as plt
import mediapy as media
import numpy as np
from PIL import Image, ImageDraw
from tqdm.auto import tqdm

# allow running from anywhere: repo root is the parent of preprocess/
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# --- Trajectory smoothing (ported from visualizer/vis_densetrack3d_trails.py) ---
# Pure-numpy so we don't pull in scipy. Temporal Gaussian per point, applied only
# over continuous visible segments (never across an occlusion gap).

def _gaussian_kernel1d(sigma):
    radius = max(int(3 * sigma + 0.5), 1)
    x = np.arange(-radius, radius + 1, dtype=np.float32)
    k = np.exp(-(x ** 2) / (2 * sigma ** 2))
    return k / k.sum()


def _smooth_segment(data, kernel):
    """Gaussian-smooth a single (L, 3) segment along time; 'nearest' edge handling."""
    r = len(kernel) // 2
    padded = np.pad(data, ((r, r), (0, 0)), mode="edge")  # clamp ends so they don't drift
    out = np.empty_like(data)
    for d in range(data.shape[1]):
        out[:, d] = np.convolve(padded[:, d], kernel, mode="valid")
    return out


def smooth_trajectories_temporal(trajs, mask, sigma=2.0):
    """Temporal Gaussian smoothing. trajs (T, N, 3), mask (T, N). Smooths each
    continuous visible run independently; runs shorter than ~3*sigma are left as-is."""
    T, N, _ = trajs.shape
    kernel = _gaussian_kernel1d(sigma)
    min_seg = max(int(sigma * 3), 5)
    out = trajs.copy()
    for i in tqdm(range(N), desc=f"Smoothing (sigma={sigma})"):
        valid = np.where(mask[:, i])[0]
        if len(valid) == 0:
            continue
        splits = np.where(np.diff(valid) > 1)[0] + 1  # break at occlusion gaps
        for seg in np.split(valid, splits):
            if len(seg) >= min_seg:
                out[seg, i, :] = _smooth_segment(trajs[seg, i, :], kernel)
    return out


def fill_trajectory_gaps(trajs, mask, max_gap=3):
    """Linearly interpolate occlusion gaps of <= max_gap frames, marking them visible.
    Run before smoothing so brief occlusions become a straight bridge the Gaussian rounds."""
    T, N, _ = trajs.shape
    ft, fm = trajs.copy(), mask.copy()
    for i in tqdm(range(N), desc=f"Filling gaps (<= {max_gap})"):
        valid = np.where(mask[:, i])[0]
        if len(valid) < 2:
            continue
        diffs = np.diff(valid)
        for g in np.where((diffs > 1) & (diffs <= max_gap + 1))[0]:
            s, e = valid[g], valid[g + 1]
            for t in range(s + 1, e):
                a = (t - s) / (e - s)
                ft[t, i] = (1 - a) * ft[s, i] + a * ft[e, i]
                fm[t, i] = True
    return ft, fm


def project_to_pixels(coords, fx, fy, cx, cy):
    """Re-project camera-centric 3D coords (T,N,3) back to pixels (T,N,2).

    Inverse of the tracker's lift (xyz = K^-1 [u,v,1] * Z): u = fx*X/Z + cx.
    Points with non-positive/NaN depth become NaN (drawn as invisible)."""
    X, Y, Z = coords[..., 0], coords[..., 1], coords[..., 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        u = fx * (X / Z) + cx
        v = fy * (Y / Z) + cy
    uv = np.stack([u, v], axis=-1)
    uv[(Z <= 0) | ~np.isfinite(Z)] = np.nan
    return uv


def rainbow_colors_by_position(uv, vis):
    """HSV color per track, keyed by its first-visible 2D position (like the trails viewer).
    Returns (N, 3) uint8. Tracks that are never visible get grey."""
    T, N, _ = uv.shape
    first_vis = np.argmax(vis, axis=0)          # (N,)
    never = ~np.any(vis, axis=0)
    first_vis[never] = 0
    idx = np.arange(N)
    first_xy = uv[first_vis, idx]               # (N, 2)
    first_xy[never] = np.nan
    xy_min = np.nanmin(first_xy, axis=0)
    xy_max = np.nanmax(first_xy, axis=0)
    xy_norm = (first_xy - xy_min) / (xy_max - xy_min + 1e-6)
    scalar = np.nansum(xy_norm, axis=1)
    scalar = (scalar - np.nanmin(scalar)) / (np.nanmax(scalar) - np.nanmin(scalar) + 1e-6)
    sort_idx = np.argsort(scalar)
    hsv = plt.cm.hsv(np.linspace(0, 1, N))[:, :3]
    out = (hsv[np.argsort(sort_idx)] * 255).astype(np.uint8)
    out[never] = 128
    return out


def render_2d_overlay(video_np, uv, vis, colors, trace=8):
    """Draw merged 2D tracks over the RGB frames.

    video_np (T,H,W,3) uint8 RGB; uv (T,N,2) pixel coords (NaN where inactive);
    vis (T,N) bool; colors (N,3) 0-255. A point/segment is drawn only where vis
    is True, so NaN-padded / occluded frames are skipped. Returns (T,H,W,3)."""
    T, N = vis.shape
    colors = colors.astype(np.uint8)
    frames = []
    for t in tqdm(range(T), desc="Rendering 2D overlay"):
        img = Image.fromarray(video_np[t].copy())
        draw = ImageDraw.Draw(img)
        # trailing lines: connect consecutive frames where both ends are visible
        if trace > 0:
            for t0 in range(max(0, t - trace), t):
                seg_ok = vis[t0] & vis[t0 + 1]
                for i in np.where(seg_ok)[0]:
                    p0, p1 = uv[t0, i], uv[t0 + 1, i]
                    if not (np.isfinite(p0).all() and np.isfinite(p1).all()):
                        continue
                    draw.line([p0[0], p0[1], p1[0], p1[1]], fill=tuple(int(c) for c in colors[i]), width=1)
        # points at the current frame
        for i in np.where(vis[t])[0]:
            x, y = float(uv[t, i, 0]), float(uv[t, i, 1])
            if not (np.isfinite(x) and np.isfinite(y)):
                continue
            c = tuple(int(v) for v in colors[i])
            draw.ellipse([x - 2, y - 2, x + 2, y + 2], fill=c, outline=c)
        frames.append(np.asarray(img))
    return np.stack(frames)


def load_color_frames(color_path):
    """List of RGB frames from an mp4 or an image folder (index-aligned to the pkl)."""
    if os.path.isdir(color_path):
        files = sorted(f for f in os.listdir(color_path) if f.lower().endswith((".png", ".jpg", ".jpeg")))
        out = []
        for f in files:
            bgr = cv2.imread(os.path.join(color_path, f))
            out.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        return out
    cap = cv2.VideoCapture(color_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open color video: {color_path}")
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    return frames


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--pkl", required=True, help="dense_3d_track.pkl from track_windowed.py")
    p.add_argument("--video", required=True, help="color.mp4 or a folder of RGB frames (the source)")
    p.add_argument("--output", default=None, help="output mp4 path (default: <pkl dir>/tracks_2d_smoothed.mp4)")
    p.add_argument("--intrinsics", default="771.59,771.365,645.555,349.653", help="fx,fy,cx,cy at native res")
    p.add_argument("--start-frame", type=int, default=0, help="absolute frame the pkl started at (must match the run)")
    p.add_argument("--num-frames", type=int, default=-1, help="frames the pkl covers; -1 = infer from pkl length")
    p.add_argument("--smooth-sigma", type=float, default=3.0, help="temporal Gaussian sigma (0 = off)")
    p.add_argument("--fill-gap", type=int, default=0, help="linearly bridge occlusion gaps up to N frames before smoothing (0 = off)")
    p.add_argument("--trace", type=int, default=16, help="trail length in frames")
    p.add_argument("--fps", type=int, default=10, help="output video fps")
    return p.parse_args()


def main():
    args = parse_args()

    with open(args.pkl, "rb") as h:
        d = pickle.load(h)
    coords = d["coords"].astype(np.float32)   # (T, N, 3)
    vis = d["vis"].astype(bool)               # (T, N)
    T, N = vis.shape
    print(f"Loaded {args.pkl}: {T} frames, {N} tracks")

    # smooth in 3D over visible segments, then project -> smoothed 2D trails.
    if args.fill_gap > 0:
        coords, vis = fill_trajectory_gaps(coords, vis, max_gap=args.fill_gap)
    if args.smooth_sigma > 0:
        coords = smooth_trajectories_temporal(coords, vis, sigma=args.smooth_sigma)

    fx, fy, cx, cy = (float(x) for x in args.intrinsics.split(","))
    uv = project_to_pixels(coords, fx, fy, cx, cy)   # (T, N, 2)
    # a projected point is only drawable where the track is visible AND finite
    vis = vis & np.isfinite(uv).all(axis=-1)

    # source frames, sliced to the pkl's absolute range
    print(f"Loading color from {args.video}")
    all_frames = load_color_frames(args.video)
    num = T if args.num_frames < 0 else args.num_frames
    end = min(args.start_frame + num, len(all_frames))
    frames = all_frames[args.start_frame:end]
    if len(frames) != T:
        raise ValueError(
            f"pkl covers {T} frames but video slice [{args.start_frame}:{end}] has {len(frames)}. "
            "Pass the same --start-frame / --num-frames used for the tracking run."
        )
    video_np = np.stack(frames)              # (T, H, W, 3) RGB

    colors = rainbow_colors_by_position(uv, vis)
    out_video = render_2d_overlay(video_np, uv, vis, colors, trace=args.trace)

    out_path = args.output or os.path.join(os.path.dirname(args.pkl), "tracks_2d_smoothed.mp4")
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    media.write_video(out_path, out_video, fps=args.fps)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
