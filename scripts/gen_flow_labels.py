#!/usr/bin/env python3
"""Offline precompute for the object-flow intent model (spec §5.1).

The expensive, epoch-invariant part of the label pipeline is turning each frame's
masked depth into a metric object point cloud. Decoding `depth.mkv`, applying the
mask, and back-projecting is far too slow to redo every epoch, yet the result is
tiny (a few hundred points per frame). So we precompute it ONCE per episode into a
compact `clouds.npz`; the online `Dataset` (spec §5.2) then only does cheap array
slicing on `coords`/`vis`/`hand.pkl` plus a single cloud load per item.

For EVERY frame t (masks are dense -- one per frame, not just tracker keyframes):

  1. read the 16-bit depth (mm) for frame t from depth.mkv and scale to metres;
  2. read the object mask seg/mask/{t:05d}.png (nonzero pixel = object) and keep
     only masked pixels with valid (>0) depth;
  3. back-project to the CAMERA-OPTICAL frame -- the SAME frame object_flow.pkl
     lives in -- via the crop-adjusted intrinsics in intrinsics.txt:
         x = (u - cx) * z / fx,  y = (v - cy) * z / fy,  z = depth;
  4. subsample to a fixed P points (random or farthest-point) -> row t.

A frame whose mask is empty (object fully occluded / out of view) or that has
fewer than --min-pts valid masked pixels gets an all-NaN row; the Dataset skips
those as window present-frames.

Output per episode: `clouds.npz` with
    clouds  (Tclip, P, 3) float32  camera-frame metres, NaN rows for empty frames
    n_valid (Tclip,)      int32    masked+valid pixel count before subsampling
    intrinsics (4,)       float32  fx,fy,cx,cy actually used (provenance)
At P=512 this is ~Tclip * 512 * 3 * 4 B ~= 7 MB for a ~1200-frame episode.

Usage:
  # one episode
  python scripts/gen_flow_labels.py --episode /path/to/inhand_manipulation/0713_cube/episode_1
  # a whole clip (all episode_* under it)
  python scripts/gen_flow_labels.py --clip /path/to/inhand_manipulation/0713_cube
  # the entire dataset root (all clips, all episodes)
  python scripts/gen_flow_labels.py --data-root /home/labeng/yanghan/data/inhand_manipulation
"""
import argparse
import glob
import os
import sys

import cv2
import numpy as np

# repo root is the parent of scripts/ -- allow running from anywhere, and reuse the
# exact depth-video decoder the extractor/tracker use (FFV1 gray16le, bit-exact).
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from preprocess.extract_mcap_rgbd import read_depth_video


def load_intrinsics(path):
    """Read 'fx,fy,cx,cy' from intrinsics.txt -> (fx, fy, cx, cy) floats."""
    with open(path) as f:
        vals = [float(x) for x in f.read().strip().split(",")]
    assert len(vals) == 4, f"expected 4 intrinsics (fx,fy,cx,cy), got {vals}"
    return tuple(vals)


def load_mask(path, hw):
    """Boolean object mask (H, W) from a seg PNG (nonzero pixel in any channel = object).

    Matches track_windowed.load_segm_mask: collapse RGB to a single channel, treat
    >0 as object, resize (nearest) to the depth resolution if they disagree.
    """
    H, W = hw
    mask = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if mask is None:
        raise IOError(f"failed to read mask: {path}")
    mask = np.asarray(mask)
    if mask.ndim == 3:  # collapse RGB/RGBA to one channel
        mask = mask[..., :3].max(axis=-1)
    if mask.shape != (H, W):
        mask = cv2.resize(mask.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST)
    return mask > 0


def backproject(depth_m, mask, K):
    """Masked, valid-depth pixels -> (M, 3) camera-optical xyz in metres.

    depth_m (H, W) float32 metres (0 = invalid); mask (H, W) bool; K = (fx,fy,cx,cy).
    Camera-optical frame (+X right, +Y down, +Z into scene) -- the same frame the
    tracker lifts uv+depth into, so these clouds are registered with object_flow.pkl.
    """
    fx, fy, cx, cy = K
    sel = mask & np.isfinite(depth_m) & (depth_m > 0)
    ys, xs = np.nonzero(sel)
    z = depth_m[ys, xs]
    x = (xs.astype(np.float32) - cx) * z / fx
    y = (ys.astype(np.float32) - cy) * z / fy
    return np.stack([x, y, z], axis=1).astype(np.float32)  # (M, 3)


def fps_subsample(pts, n, rng):
    """Farthest-point subsample (M,3) -> (n,3) for even surface coverage.

    Greedy: seed with a random point, then repeatedly take the point farthest from
    the chosen set (tracked via a running min-distance array, O(n*M)).
    """
    M = pts.shape[0]
    idx = np.empty(n, dtype=np.int64)
    idx[0] = rng.integers(M)
    d = np.sum((pts - pts[idx[0]]) ** 2, axis=1)
    for i in range(1, n):
        idx[i] = int(np.argmax(d))
        d = np.minimum(d, np.sum((pts - pts[idx[i]]) ** 2, axis=1))
    return pts[idx]


def subsample(pts, P, method, rng):
    """Fix a cloud to exactly P points. Fewer than P -> pad by sampling WITH
    replacement (duplicates are harmless for PointNet++ and keep a fixed shape)."""
    M = pts.shape[0]
    if M >= P:
        if method == "fps":
            return fps_subsample(pts, P, rng)
        return pts[rng.choice(M, P, replace=False)]
    pad = pts[rng.choice(M, P - M, replace=True)]
    return np.concatenate([pts, pad], axis=0)


def process_episode(ep, P, method, depth_scale, min_pts, seed, force):
    """Build clouds.npz for one episode dir. Returns a short status string."""
    out_path = os.path.join(ep, "clouds.npz")
    if os.path.exists(out_path) and not force:
        return f"skip (exists): {out_path}"

    intr_path = os.path.join(ep, "intrinsics.txt")
    depth_path = os.path.join(ep, "depth.mkv")
    mask_dir = os.path.join(ep, "seg", "mask")
    for pth in (intr_path, depth_path, mask_dir):
        if not os.path.exists(pth):
            return f"skip (missing {os.path.basename(pth)}): {ep}"

    K = load_intrinsics(intr_path)
    depth_all = read_depth_video(depth_path)          # (Td, H, W) uint16 mm
    Td, H, W = depth_all.shape
    mask_files = sorted(glob.glob(os.path.join(mask_dir, "*.png")))
    Tclip = min(Td, len(mask_files))                  # dense masks; both index-aligned
    if Tclip == 0:
        return f"skip (no frames): {ep}"

    rng = np.random.default_rng(seed)
    clouds = np.full((Tclip, P, 3), np.nan, dtype=np.float32)
    n_valid = np.zeros(Tclip, dtype=np.int32)
    n_empty = 0
    for t in range(Tclip):
        depth_m = depth_all[t].astype(np.float32) / depth_scale
        mask = load_mask(mask_files[t], (H, W))
        pts = backproject(depth_m, mask, K)
        n_valid[t] = pts.shape[0]
        if pts.shape[0] < min_pts:
            n_empty += 1
            continue                                  # leave the all-NaN row
        clouds[t] = subsample(pts, P, method, rng)

    np.savez_compressed(out_path, clouds=clouds, n_valid=n_valid,
                        intrinsics=np.asarray(K, dtype=np.float32))
    mb = clouds.nbytes / 1e6
    return (f"wrote {out_path}: {Tclip} frames, {n_empty} empty, "
            f"P={P} ({method}), ~{mb:.1f} MB")


def discover_episodes(args):
    """Resolve --episode / --clip / --data-root into a list of episode dirs."""
    if args.episode:
        return [os.path.abspath(args.episode)]
    if args.clip:
        return sorted(glob.glob(os.path.join(args.clip, "episode_*")))
    eps = []
    for clip in sorted(glob.glob(os.path.join(args.data_root, "*"))):
        if os.path.isdir(clip):
            eps.extend(sorted(glob.glob(os.path.join(clip, "episode_*"))))
    return eps


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = p.add_mutually_exclusive_group()
    src.add_argument("--episode", help="one episode dir (contains depth.mkv, seg/mask, intrinsics.txt)")
    src.add_argument("--clip", help="a clip dir; process all episode_* under it")
    src.add_argument("--data-root", default="/home/labeng/yanghan/data/inhand_manipulation",
                     help="dataset root; process every clip's episode_* (default)")
    p.add_argument("-P", "--num-points", type=int, default=512, help="fixed points per cloud")
    p.add_argument("--subsample", choices=["random", "fps"], default="random",
                   help="random (fast, default) or farthest-point (even coverage, slower)")
    p.add_argument("--depth-scale", type=float, default=1000.0, help="raw depth units per metre (mm=1000)")
    p.add_argument("--min-pts", type=int, default=16,
                   help="frames with fewer valid masked pixels get an all-NaN (skipped) row")
    p.add_argument("--seed", type=int, default=0, help="RNG seed for subsampling (reproducible clouds)")
    p.add_argument("--force", action="store_true", help="recompute even if clouds.npz exists")
    return p.parse_args()


def main():
    args = parse_args()
    episodes = discover_episodes(args)
    if not episodes:
        print("No episodes found.")
        return
    print(f"Precomputing clouds for {len(episodes)} episode(s), P={args.num_points} ({args.subsample})")
    for i, ep in enumerate(episodes):
        print(f"[{i + 1}/{len(episodes)}] {ep}")
        try:
            print("   " + process_episode(ep, args.num_points, args.subsample,
                                           args.depth_scale, args.min_pts, args.seed, args.force))
        except Exception as e:  # keep going; one bad episode shouldn't abort a batch
            print(f"   ERROR: {type(e).__name__}: {e}")


if __name__ == "__main__":
    main()
