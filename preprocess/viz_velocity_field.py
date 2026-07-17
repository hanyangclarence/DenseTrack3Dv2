#!/usr/bin/env python3
"""Visualize the instantaneous 3D VELOCITY FIELD of the object flow on the 2D video.

For each tracked point at frame t, estimate its 3D velocity v (m/s) by central
finite difference of its camera-frame position over +/-dt frames, then draw it as
the 2D projection of the (scaled) 3D velocity vector:

    tail = reproject(x_t)
    head = reproject(x_t + scale * v)

So the on-screen arrow is the true 3D velocity vector projected to pixels: its
length carries the FULL 3D magnitude (perspective-foreshortened by orientation --
a point moving straight toward/away from the camera correctly yields a short arrow),
and its color encodes the 3D speed ||v|| via a colormap so that depth motion, which
foreshortens the arrow, is still readable as "fast". A legend bar maps color->speed.

Reprojection reuses postprocess.smooth_object_motion.reproject (the exact inverse of
how the tracker lifted uv+depth to xyz), so arrows sit on the object the same way
object_flow_2d.mp4 does. Velocity is estimated in metric 3D BEFORE projection, so it
is a genuine 3D field, not 2D optical flow.

Usage:
  python preprocess/viz_velocity_field.py --folder results/test_manip_data_cube \
      [--dt 2] [--arrow-scale 0.5] [--point-stride 1] [--vmax auto] [--fps 30]
"""
import argparse
import os
import pickle
import sys

import cv2
import matplotlib
import matplotlib.colors as mcolors
import mediapy as media
import numpy as np
from tqdm.auto import tqdm

# repo root is the parent of preprocess/ -- allow running from anywhere
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# reuse the SAME reprojection + background loader as the other overlays
from postprocess.smooth_object_motion import reproject, load_background


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--folder", required=True,
                   help="result folder holding object_flow.pkl, color.mp4, intrinsics.txt")
    p.add_argument("--dt", type=int, default=2,
                   help="half-window (frames) for central-difference velocity: "
                        "v_t = (x_{t+dt} - x_{t-dt}) / (2*dt/fps) metres/second")
    p.add_argument("--arrow-scale", type=float, default=0.5,
                   help="seconds of velocity to draw: head = reproject(x + scale*v). "
                        "0.5 => arrow spans where the point would be in 0.5 s at current v")
    p.add_argument("--point-stride", type=int, default=1,
                   help="draw every k-th point (thin the ~115 visible pts/frame if cluttered)")
    p.add_argument("--vmax", default="auto",
                   help="speed (m/s) mapped to the top of the colormap; 'auto' = 95th "
                        "percentile of all valid speeds")
    p.add_argument("--cmap", default="turbo", help="matplotlib colormap for speed")
    p.add_argument("--fps", type=float, default=30.0,
                   help="capture fps -- sets the metric time step AND the output fps")
    p.add_argument("--pkl", default="object_flow.pkl", help="pkl name within --folder")
    p.add_argument("--video", default="color.mp4", help="RGB source name within --folder")
    p.add_argument("--intrinsics", default="intrinsics.txt",
                   help="intrinsics file (fx,fy,cx,cy) within --folder")
    p.add_argument("--out", default="velocity_field.mp4", help="output mp4 name within --folder")
    return p.parse_args()


def load_intrinsics(path):
    """Read 'fx,fy,cx,cy' from intrinsics.txt -> (fx,fy,cx,cy) floats."""
    with open(path) as f:
        vals = [float(x) for x in f.read().strip().split(",")]
    assert len(vals) == 4, f"expected 4 intrinsics (fx,fy,cx,cy), got {vals}"
    return tuple(vals)


def compute_velocity(coords, vis, dt, fps):
    """Central-difference 3D velocity (m/s) per point per frame.

    coords (T,N,3), vis (T,N) bool. Returns:
      vel   (T,N,3)  float32, NaN where velocity is undefined
      speed (T,N)    float32 = ||vel||, NaN where undefined
      vel_ok(T,N)    bool: both t-dt and t+dt exist AND are visible with finite xyz
    A velocity needs the point visible at both bracketing frames so the difference is
    a real displacement (not a jump across an occlusion gap). dt/fps is the time base.
    """
    T, N, _ = coords.shape
    vel = np.full((T, N, 3), np.nan, dtype=np.float32)
    vel_ok = np.zeros((T, N), dtype=bool)
    tsec = 2.0 * dt / fps
    lo, hi = dt, T - dt
    if hi > lo:
        a = coords[2 * dt:]          # x_{t+dt} for t in [lo,hi)
        b = coords[:T - 2 * dt]      # x_{t-dt}
        va = vis[2 * dt:] & vis[:T - 2 * dt]
        finite = np.isfinite(a).all(-1) & np.isfinite(b).all(-1)
        ok = va & finite
        v = (a - b) / tsec
        vel[lo:hi][ok] = v[ok]
        vel_ok[lo:hi] = ok
    speed = np.linalg.norm(vel, axis=-1)
    return vel, speed, vel_ok


def draw_legend(img, cmap, vmax, unit="m/s"):
    """Draw a vertical colorbar (0..vmax) with tick labels in the top-right corner."""
    H, W = img.shape[:2]
    bw, bh = 14, min(180, H - 40)
    x1, y0 = W - 24, 20
    for j in range(bh):
        frac = 1.0 - j / (bh - 1)  # top = vmax
        c = tuple(int(255 * x) for x in cmap(frac)[:3])
        cv2.line(img, (x1, y0 + j), (x1 + bw, y0 + j), c, 1)
    cv2.rectangle(img, (x1, y0), (x1 + bw, y0 + bh), (255, 255, 255), 1)
    for frac, val in [(0.0, vmax), (0.5, vmax / 2), (1.0, 0.0)]:
        yy = int(y0 + frac * (bh - 1))
        cv2.putText(img, f"{val:.2f}", (x1 - 42, yy + 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(img, unit, (x1 - 30, y0 - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1, cv2.LINE_AA)


def render_velocity_field(bg, coords, vis, vel, speed, vel_ok, K, scale, stride,
                          vmax, cmap):
    """Draw projected 3D velocity arrows colored by 3D speed over the RGB frames.

    Arrow = reproject(x_t) -> reproject(x_t + scale*v); both endpoints must reproject
    to finite pixels (a point moving out of frame or behind the camera is skipped for
    that frame). Color = cmap(speed / vmax). Returns (T,H,W,3).
    """
    T, N = vis.shape
    idx = np.arange(0, N, stride)
    norm = mcolors.Normalize(vmin=0.0, vmax=vmax, clip=True)
    fx, fy, cx, cy = K

    def proj(p):
        Z = p[2]
        if not np.isfinite(Z) or Z <= 1e-6:
            return None
        return (int(fx * p[0] / Z + cx), int(fy * p[1] / Z + cy))

    out = np.empty_like(bg)
    for t in tqdm(range(T), desc="Rendering velocity field"):
        img = np.ascontiguousarray(bg[t])
        for i in idx:
            if not vel_ok[t, i]:
                continue
            tail = proj(coords[t, i])
            head = proj(coords[t, i] + scale * vel[t, i])
            if tail is None or head is None:
                continue
            c = tuple(int(255 * x) for x in cmap(norm(speed[t, i]))[:3])
            cv2.arrowedLine(img, tail, head, c, 1, cv2.LINE_AA, tipLength=0.35)
            cv2.circle(img, tail, 2, c, -1, cv2.LINE_AA)
        draw_legend(img, cmap, vmax)
        out[t] = img
    return out


def main():
    args = parse_args()
    pkl_path = os.path.join(args.folder, args.pkl)
    vid_path = os.path.join(args.folder, args.video)
    intr_path = os.path.join(args.folder, args.intrinsics)

    with open(pkl_path, "rb") as f:
        data = pickle.load(f)
    coords, vis = data["coords"], data["vis"]
    T, N, _ = coords.shape
    K = load_intrinsics(intr_path)

    vel, speed, vel_ok = compute_velocity(coords, vis, args.dt, args.fps)
    valid = speed[vel_ok]
    if args.vmax == "auto":
        vmax = float(np.percentile(valid, 95)) if valid.size else 1.0
        vmax = max(vmax, 1e-3)
    else:
        vmax = float(args.vmax)
    cmap = matplotlib.colormaps[args.cmap]

    print(f"clip: T={T} frames, N={N} points, dt={args.dt} (+/-{args.dt/args.fps:.3f}s), "
          f"arrow-scale={args.arrow_scale}s")
    if valid.size:
        print(f"3D speed (m/s) over valid points: mean {valid.mean():.3f}, "
              f"median {np.median(valid):.3f}, p95 {np.percentile(valid,95):.3f}, "
              f"max {valid.max():.3f}  ->  vmax={vmax:.3f}")

    bg = load_background(vid_path, T, H=720, W=1280)  # real frame size wins
    overlay = render_velocity_field(bg, coords, vis, vel, speed, vel_ok, K,
                                    args.arrow_scale, args.point_stride, vmax, cmap)
    out_path = os.path.join(args.folder, args.out)
    media.write_video(out_path, overlay, fps=args.fps)
    print(f"wrote {out_path}  ({T} frames @ {args.fps} fps)")


if __name__ == "__main__":
    main()
