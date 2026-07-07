#!/usr/bin/env python3
"""Convert a ZED RGB-D capture into the folder layout DenseTrack3Dv2 expects.

Input (from the ZED SDK rgbd_capture):
    color.mp4                 RGB video (or a folder of RGB frames)
    depth/depth_000000.png    16-bit PNG depth, one per RGB frame, in millimetres

Output (consumed by demo.py -> densetrack3d.datasets.custom_data.read_data):
    <out>/color/000000.png    selected RGB frames (native resolution)
    <out>/depth_pred.npy      (T, H, W) float32 depth in METRES, aligned to color/
    <out>/intrinsics.npy      3x3 camera matrix (only if --intrinsics given)

We write depth as depth_pred.npy (not a depth/ folder) on purpose: read_data's
`depth/` branch has a bug (it loads depth filenames from the color/ dir), while
the depth_pred.npy branch is clean.

Zero-valued depth pixels (ZED "no return") are kept as 0 == invalid.
"""
import argparse
import glob
import os

import cv2
import numpy as np


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--color", required=True, help="Path to color.mp4 or a folder of RGB frames")
    p.add_argument("--depth", required=True, help="Path to the depth/ folder of 16-bit PNGs (millimetres)")
    p.add_argument("--out", required=True, help="Output folder to create")
    p.add_argument("--start", type=int, default=0, help="First frame index to take (default 0)")
    p.add_argument("--max-frames", type=int, default=100, help="Max number of frames to write (default 100)")
    p.add_argument("--step", type=int, default=1, help="Take every Nth frame (default 1)")
    p.add_argument(
        "--depth-scale",
        type=float,
        default=1000.0,
        help="Divide raw depth by this to get metres (ZED millimetres -> m, default 1000)",
    )
    p.add_argument(
        "--intrinsics",
        type=str,
        default=None,
        help='Comma-separated fx,fy,cx,cy at the native resolution, e.g. "771.59,771.365,645.555,349.653"',
    )
    return p.parse_args()


def load_color_frames(color_path):
    """Return a list of BGR frames (as read by OpenCV) from an mp4 or an image folder."""
    if os.path.isdir(color_path):
        files = sorted(
            f for f in os.listdir(color_path) if f.lower().endswith((".png", ".jpg", ".jpeg"))
        )
        return [cv2.imread(os.path.join(color_path, f)) for f in files]

    cap = cv2.VideoCapture(color_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open color video: {color_path}")
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)
    cap.release()
    return frames


def main():
    args = parse_args()

    # --- gather sources -----------------------------------------------------
    color_frames = load_color_frames(args.color)
    depth_files = sorted(glob.glob(os.path.join(args.depth, "*.png")))
    if not depth_files:
        raise FileNotFoundError(f"No .png depth frames found in {args.depth}")

    n_color, n_depth = len(color_frames), len(depth_files)
    print(f"Found {n_color} color frames and {n_depth} depth frames")
    n = min(n_color, n_depth)
    if n_color != n_depth:
        print(f"WARNING: color/depth counts differ; using the first {n} frames of each (assumed index-aligned)")

    # --- pick frames: start, step, capped at max_frames ---------------------
    selected = list(range(args.start, n, args.step))[: args.max_frames]
    if not selected:
        raise ValueError(f"No frames selected (start={args.start}, step={args.step}, n={n})")
    print(f"Selecting {len(selected)} frames: indices {selected[0]}..{selected[-1]} step {args.step}")

    color_out = os.path.join(args.out, "color")
    os.makedirs(color_out, exist_ok=True)

    # --- write RGB + build depth stack --------------------------------------
    depth_stack = []
    ref_hw = None
    for out_idx, src_idx in enumerate(selected):
        bgr = color_frames[src_idx]
        # Saved BGR so read_data's cv2.imread + BGR2RGB yields correct RGB.
        cv2.imwrite(os.path.join(color_out, f"{out_idx:06d}.png"), bgr)

        depth_mm = cv2.imread(depth_files[src_idx], cv2.IMREAD_ANYDEPTH)
        if depth_mm is None:
            raise IOError(f"Failed to read depth frame: {depth_files[src_idx]}")
        depth_m = depth_mm.astype(np.float32) / args.depth_scale  # mm -> m; zeros stay 0 (invalid)

        # Ensure depth matches the RGB resolution (nearest, to not invent depth).
        h, w = bgr.shape[:2]
        if depth_m.shape != (h, w):
            depth_m = cv2.resize(depth_m, (w, h), interpolation=cv2.INTER_NEAREST)
        depth_stack.append(depth_m)
        ref_hw = (h, w)

    depth_stack = np.stack(depth_stack, axis=0).astype(np.float32)  # (T, H, W)
    np.save(os.path.join(args.out, "depth_pred.npy"), depth_stack)

    valid = depth_stack > 0
    print(
        f"Wrote {len(selected)} RGB frames to {color_out}\n"
        f"Wrote depth_pred.npy {depth_stack.shape} (metres); "
        f"valid depth {100 * valid.mean():.1f}%, "
        f"range {depth_stack[valid].min():.3f}-{depth_stack[valid].max():.3f} m"
    )

    # --- intrinsics ---------------------------------------------------------
    if args.intrinsics:
        fx, fy, cx, cy = (float(x) for x in args.intrinsics.split(","))
        K = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float32)
        np.save(os.path.join(args.out, "intrinsics.npy"), K)
        print(f"Wrote intrinsics.npy (3x3) for resolution {ref_hw[1]}x{ref_hw[0]}:\n{K}")
        print(
            "NOTE: demo.py does NOT read intrinsics.npy — by default it uses a rough fx=fy=W guess.\n"
            "      To use these real intrinsics, the predictor must be called with predefined_intrs."
        )


if __name__ == "__main__":
    main()
