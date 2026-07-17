#!/usr/bin/env python3
"""Visualize the GOAL-POSE flow target: where each tracked point ends up after a
fixed horizon (default 1 s), as a dot-at-now + arrow-to-future overlay.

This is a sanity check for the simplified intent target used by the object-flow
world model. Instead of predicting a full per-step trajectory, that model predicts
ONE displacement per point over a ~1 s horizon (the "goal pose"). This script draws
exactly that target on the source video so it can be eyeballed before training:

  - each object-flow point visible at frame t is a DOT at its reprojected pixel;
  - if the SAME point is also visible at t+H (a valid 1 s goal exists), an ARROW is
    drawn from its position now to its position then;
  - a point visible now but occluded / out-of-view at t+H gets a HOLLOW, faded dot
    and NO arrow -- so the fraction of points that actually have a valid goal is
    visible at a glance (that survival question is the whole reason to check).

Reprojection reuses postprocess.smooth_object_motion.reproject (the exact inverse
of how the tracker lifted uv+depth to xyz), so the dots land on the object the same
way the object_flow_2d.mp4 overlay does.

Usage:
  python preprocess/viz_goal_flow.py --folder results/test_manip_data_cube \
      [--horizon-frames 30] [--point-stride 1] [--fps 30] [--out goal_flow.mp4]
"""
import argparse
import os
import pickle
import sys

import cv2
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
    p.add_argument("--horizon-frames", type=int, default=30,
                   help="goal horizon H in NATIVE frames (30 = 1 s at 30 fps); "
                        "the arrow goes from t to t+H")
    p.add_argument("--point-stride", type=int, default=1,
                   help="draw every k-th point (thin the ~115 visible pts/frame if cluttered)")
    p.add_argument("--fps", type=float, default=30.0, help="output video fps")
    p.add_argument("--pkl", default="object_flow.pkl", help="pkl name within --folder")
    p.add_argument("--video", default="color.mp4", help="RGB source name within --folder")
    p.add_argument("--intrinsics", default="intrinsics.txt",
                   help="intrinsics file (fx,fy,cx,cy) within --folder")
    p.add_argument("--out", default="goal_flow.mp4", help="output mp4 name within --folder")
    return p.parse_args()


def load_intrinsics(path):
    """Read 'fx,fy,cx,cy' from intrinsics.txt -> (fx,fy,cx,cy) floats."""
    with open(path) as f:
        vals = [float(x) for x in f.read().strip().split(",")]
    assert len(vals) == 4, f"expected 4 intrinsics (fx,fy,cx,cy), got {vals}"
    return tuple(vals)


def render_goal_flow(bg, uv, vis, colors, horizon, stride):
    """Draw dot-now + arrow-to-(t+horizon) per point over the RGB frames.

    bg (T,H,W,3) uint8 RGB; uv (T,N,2) reprojected pixels (NaN where invalid);
    vis (T,N) bool; colors (N,3) 0-255. For each frame t, a point is drawn iff it is
    visible with finite pixels at t. If it is ALSO visible with finite pixels at
    t+horizon, a solid dot + arrow (now -> future) is drawn; otherwise a hollow,
    dimmed dot and no arrow (no valid goal within the horizon). Returns (T,H,W,3).
    """
    T, N = vis.shape
    colors = colors.astype(np.uint8)
    idx = np.arange(0, N, stride)
    out = np.empty_like(bg)
    for t in tqdm(range(T), desc="Rendering goal-flow overlay"):
        img = np.ascontiguousarray(bg[t])
        tf = t + horizon
        for i in idx:
            p0 = uv[t, i]
            if not (vis[t, i] and np.isfinite(p0).all()):
                continue
            c = tuple(int(x) for x in colors[i])
            x0, y0 = int(p0[0]), int(p0[1])
            has_goal = tf < T and vis[tf, i] and np.isfinite(uv[tf, i]).all()
            if has_goal:
                p1 = uv[tf, i]
                cv2.arrowedLine(img, (x0, y0), (int(p1[0]), int(p1[1])), c, 1,
                                cv2.LINE_AA, tipLength=0.3)
                cv2.circle(img, (x0, y0), 3, c, -1, cv2.LINE_AA)   # solid = has goal
            else:
                dim = tuple(int(0.4 * v) for v in c)
                cv2.circle(img, (x0, y0), 3, dim, 1, cv2.LINE_AA)  # hollow = no goal
        out[t] = img
    return out


def main():
    args = parse_args()
    pkl_path = os.path.join(args.folder, args.pkl)
    vid_path = os.path.join(args.folder, args.video)
    intr_path = os.path.join(args.folder, args.intrinsics)

    with open(pkl_path, "rb") as f:
        data = pickle.load(f)
    coords, vis, colors = data["coords"], data["vis"], data["colors"]
    T, N, _ = coords.shape
    K = load_intrinsics(intr_path)

    uv = reproject(coords, vis, K)  # (T,N,2), NaN where invalid or Z<=0
    # real frame size wins; the H/W args only matter if the video is missing/short.
    bg = load_background(vid_path, T, H=720, W=1280)

    n_with_goal = 0
    for t in range(max(0, T - args.horizon_frames)):
        f = t + args.horizon_frames
        n_with_goal += int((vis[t] & vis[f]).sum())
    n_vis = int(vis[: max(0, T - args.horizon_frames)].sum())
    frac = (n_with_goal / n_vis) if n_vis else 0.0
    print(f"clip: T={T} frames, N={N} points, horizon={args.horizon_frames} frames "
          f"({args.horizon_frames / args.fps:.2f} s)")
    print(f"points visible-now with a valid {args.horizon_frames}-frame goal: "
          f"{n_with_goal}/{n_vis} = {frac:.1%}")

    overlay = render_goal_flow(bg, uv, vis, colors, args.horizon_frames, args.point_stride)
    out_path = os.path.join(args.folder, args.out)
    media.write_video(out_path, overlay, fps=args.fps)
    print(f"wrote {out_path}  ({T} frames @ {args.fps} fps)")


if __name__ == "__main__":
    main()
