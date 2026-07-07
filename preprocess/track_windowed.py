#!/usr/bin/env python3
"""Windowed object 3D tracking over a long ZED capture, merged into one pkl.

Idea (see conversation): a single tracking run seeds all its query points at the
first frame and never adds new ones, so anything entering later is missed and
long runs drift. Instead we sweep OVERLAPPING windows (e.g. 0-30, 20-50, ...),
and at each window's FIRST frame we re-seed using that frame's object mask. New
surface is therefore picked up at every window start ("samples refresh"), and
because we track only in-mask points the merged file stays small.

This uses the SPARSE tracker (demo_sparse.py's Predictor3D): a regular grid of
`grid_size x grid_size` query points is intersected with the seed-frame mask, so
only object points are ever tracked. Much faster than the dense per-pixel path,
at the cost of fewer points (tune --grid-size for density).

Each window is an independent predictor run. The 3D coords it returns are
camera-centric per frame (xyz = K^-1 [u,v,1] * depth(t)), so a point at absolute
frame t is in frame t's camera coordinates no matter which window produced it --
merging is just placing each window's tracks at their absolute frame indices.
There is NO cross-window identity stitching: a physical point seen in two windows
becomes two tracks (this is intended -- it re-anchors against drift). The merge
is a union, padded with NaN / vis=False outside each track's own window.

Output <out>/<name>/dense_3d_track.pkl is drop-in for both visualizers:
    coords (T, N, 3) float32  metric XYZ in metres; NaN where a track is inactive
    colors (N, 3)    float32  0-255 RGB (from each track's seed frame)
    vis    (T, N)    bool     True only inside the producing window AND model-visible

Inputs:
    --video    color.mp4 (or a folder of RGB frames)
    --depth    folder of 16-bit PNG depth (millimetres), index-aligned to video
    --mask-dir folder of per-frame masks named {abs_frame:05d}.png (nonzero=object)
    --intrinsics  fx,fy,cx,cy at native resolution
"""
import argparse
import glob
import os
import pickle
import sys
import time

import cv2
import matplotlib.pyplot as plt
import mediapy as media
import numpy as np
import torch
from PIL import Image, ImageDraw
from tqdm.auto import tqdm

# allow running from anywhere: repo root is the parent of preprocess/
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from densetrack3d.models.densetrack3d.densetrack3dv2 import DenseTrack3DV2
from densetrack3d.models.predictor.predictor import Predictor3D

device = torch.device("cuda")


class _TimerScope:
    """One measurement scope. Holds its own CUDA events so timers can nest
    (e.g. model.extract_features inside predictor_total) without clobbering
    each other's state."""

    def __init__(self, parent, label):
        self.parent = parent
        self.label = label

    def __enter__(self):
        torch.cuda.synchronize()
        self.start = torch.cuda.Event(enable_timing=True)
        self.end = torch.cuda.Event(enable_timing=True)
        self.start.record()
        return self

    def __exit__(self, *exc):
        self.end.record()
        torch.cuda.synchronize()
        ms = self.start.elapsed_time(self.end)
        self.parent.times.setdefault(self.label, []).append(ms)


class GPUTimer:
    """Accumulating, reentrant CUDA timer. `gtimer(label)` returns a fresh scope
    with its own events, so nested blocks are measured independently (with sync
    so numbers reflect real completion, not just kernel-launch return)."""

    def __init__(self):
        self.times = {}  # label -> list[ms]

    def __call__(self, label):
        return _TimerScope(self, label)

    def report(self, header="Timing breakdown"):
        print(f"\n=== {header} (per-window, warmup window 1 excluded) ===")
        for label, vals in self.times.items():
            v = vals[1:] if len(vals) > 1 else vals  # drop first (cuDNN autotune / alloc warmup)
            mean = sum(v) / len(v)
            fps = 1000.0 / mean if mean > 0 else float("inf")
            print(f"  {label:<24} mean {mean:8.2f} ms  min {min(v):8.2f}  max {max(v):8.2f}  "
                  f"({fps:6.2f} runs/s over {len(v)} runs)")


# Feature-extraction timer: monkeypatched around model.extract_features so we can
# isolate the encoder cost from the rest of the tracker without editing the model.
def wrap_feature_extraction(model, gtimer):
    orig = model.extract_features

    def timed(video):
        with gtimer("model.extract_features"):
            return orig(video)

    model.extract_features = timed


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt", default="checkpoints/densetrack3dv2.pth")
    p.add_argument("--video", required=True, help="color.mp4 or a folder of RGB frames")
    p.add_argument("--depth", required=True, help="folder of 16-bit PNG depth (millimetres)")
    p.add_argument("--mask-dir", required=True, help="folder of per-frame object masks")
    p.add_argument("--output-path", default="results/zed_windowed")
    p.add_argument("--intrinsics", default="771.59,771.365,645.555,349.653", help="fx,fy,cx,cy at native res")
    p.add_argument("--depth-scale", type=float, default=1000.0, help="divide raw depth by this to get metres")
    p.add_argument("--start-frame", type=int, default=0, help="first absolute frame to process")
    p.add_argument("--num-frames", type=int, default=400, help="number of frames to process from start-frame")
    p.add_argument("--win", type=int, default=30, help="window length in frames")
    p.add_argument("--stride", type=int, default=20, help="window start step (overlap = win - stride)")
    p.add_argument("--grid-size", type=int, default=80, help="sparse query grid side; more = denser object sampling")
    p.add_argument(
        "--keep-reappearing",
        action="store_true",
        help="by default, once a track is occluded it stays invalid for the rest of its window "
        "(no flicker-back with an unreliable position); pass this to allow it to reappear",
    )
    return p.parse_args()


def load_color_frames(color_path):
    """List of BGR frames from an mp4 or an image folder (index-aligned to depth)."""
    if os.path.isdir(color_path):
        files = sorted(f for f in os.listdir(color_path) if f.lower().endswith((".png", ".jpg", ".jpeg")))
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


def load_segm_mask(mask_path, hw):
    """Load a mask as a (1, 1, H, W) float tensor (1=object) at native resolution.

    The predictor resizes it to model resolution internally and keeps only grid
    query points that land inside it, so we just supply it at native (H, W).
    """
    H, W = hw
    if mask_path.endswith(".npy"):
        mask = np.load(mask_path)
    else:
        mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
    if mask is None:
        raise IOError(f"Failed to read mask: {mask_path}")
    mask = np.asarray(mask)
    if mask.ndim == 3:  # collapse RGB/RGBA to a single channel
        mask = mask[..., :3].max(axis=-1)
    if mask.shape != (H, W):
        mask = cv2.resize(mask.astype(np.float32), (W, H), interpolation=cv2.INTER_NEAREST)
    mask_bin = (mask > 0).astype(np.float32)
    return torch.from_numpy(mask_bin)[None, None].cuda(), int(mask_bin.sum())


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
    is True, so NaN-padded out-of-window frames are skipped. Returns (T,H,W,3).
    """
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
                    draw.line([p0[0], p0[1], p1[0], p1[1]], fill=tuple(int(c) for c in colors[i]), width=1)
        # points at the current frame
        for i in np.where(vis[t])[0]:
            x, y = float(uv[t, i, 0]), float(uv[t, i, 1])
            c = tuple(int(v) for v in colors[i])
            draw.ellipse([x - 2, y - 2, x + 2, y + 2], fill=c, outline=c)
        frames.append(np.asarray(img))
    return np.stack(frames)


def main():
    args = parse_args()

    # --- load all frames + build metric depth stack once --------------------
    print(f"Loading color from {args.video}")
    color_frames = load_color_frames(args.video)
    depth_files = sorted(glob.glob(os.path.join(args.depth, "*.png")))
    if not depth_files:
        raise FileNotFoundError(f"No .png depth frames in {args.depth}")
    n_avail = min(len(color_frames), len(depth_files))
    end = min(args.start_frame + args.num_frames, n_avail)
    abs_frames = list(range(args.start_frame, end))
    T = len(abs_frames)
    if T == 0:
        raise ValueError("No frames selected; check --start-frame / --num-frames.")
    print(f"Processing {T} frames (absolute {abs_frames[0]}..{abs_frames[-1]}) of {n_avail} available")

    # RGB (T,H,W,3) uint8 and depth (T,H,W) float32 metres, aligned to abs_frames order.
    # Time the CPU-side preprocessing (BGR->RGB, 16-bit depth decode + scale + resize).
    # This is per-frame work that a live stream would do frame-by-frame, so the
    # per-frame mean is the number that matters for a realtime budget.
    rgb_list, depth_list = [], []
    t_prep0 = time.perf_counter()
    for f in abs_frames:
        bgr = color_frames[f]
        rgb_list.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        d_mm = cv2.imread(depth_files[f], cv2.IMREAD_ANYDEPTH)
        d_m = d_mm.astype(np.float32) / args.depth_scale  # zeros stay 0 (invalid)
        h, w = bgr.shape[:2]
        if d_m.shape != (h, w):
            d_m = cv2.resize(d_m, (w, h), interpolation=cv2.INTER_NEAREST)
        depth_list.append(d_m)
    t_prep = (time.perf_counter() - t_prep0) * 1000.0
    print(f"CPU preprocessing (RGB+depth decode/resize): {t_prep:.1f} ms total, "
          f"{t_prep / max(T, 1):.2f} ms/frame ({1000.0 * T / max(t_prep, 1e-6):.1f} frames/s)")
    video_np = np.stack(rgb_list)          # (T,H,W,3)
    depth_np = np.stack(depth_list)        # (T,H,W)
    H, W = video_np.shape[1:3]

    fx, fy, cx, cy = (float(x) for x in args.intrinsics.split(","))
    K = torch.tensor([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=torch.float32, device=device)
    print(f"Intrinsics fx,fy,cx,cy = {fx},{fy},{cx},{cy} at {W}x{H}")

    # --- build model once ---------------------------------------------------
    print("Create DenseTrack3DV2 model (sparse predictor)")
    model = DenseTrack3DV2(
        stride=4,
        window_len=16,
        add_space_attn=True,
        num_virtual_tracks=64,
        model_resolution=(384, 512),
        coarse_to_fine_dense=True,
    )
    with open(args.ckpt, "rb") as f:
        state_dict = torch.load(f, map_location="cpu")
        state_dict = state_dict.get("model", state_dict)
    model.load_state_dict(state_dict, strict=False)
    predictor = Predictor3D(model=model).eval().cuda()

    # Timing harness: wrap the encoder to isolate feature-extraction cost, and
    # accumulate per-window stats for the full predictor call.
    gtimer = GPUTimer()
    wrap_feature_extraction(model, gtimer)

    # --- window plan --------------------------------------------------------
    # Local starts (into abs_frames), stepping by stride, last window clamped to cover the tail.
    starts = [s for s in range(0, T, args.stride) if s < T]
    if starts and starts[-1] + args.win < T:
        starts.append(T - args.win)  # ensure the final frames are covered
    starts = sorted(set(max(0, s) for s in starts))
    windows = [(s, min(s + args.win, T)) for s in starts]
    print(f"Windows (win={args.win}, stride={args.stride}, grid_size={args.grid_size}): "
          + ", ".join(f"{abs_frames[s]}-{abs_frames[e-1]}" for s, e in windows))

    # --- run each window ----------------------------------------------------
    # Accumulators: each track gets a full-length (T,...) buffer, active only in its window.
    all_coords, all_colors, all_vis, all_uv = [], [], [], []
    for wi, (s, e) in enumerate(windows):
        abs_start = abs_frames[s]
        Lw = e - s
        print(f"\n[window {wi + 1}/{len(windows)}] local {s}:{e} (abs {abs_start}..{abs_frames[e - 1]}), {Lw} frames")

        # host->device transfer of this window's RGB + depth
        with gtimer("h2d_transfer"):
            vid_w = torch.from_numpy(video_np[s:e]).permute(0, 3, 1, 2).cuda()[None].float()   # (1,Lw,3,H,W)
            dep_w = torch.from_numpy(depth_np[s:e]).unsqueeze(1).cuda()[None].float()          # (1,Lw,1,H,W)

        # seed mask: this window's start-frame mask, restricting the query grid to the object
        mask_path = os.path.join(args.mask_dir, f"{abs_start:05d}.png")
        segm_mask, n_obj_px = load_segm_mask(mask_path, (H, W))
        print(f"  seed mask {os.path.basename(mask_path)} ({n_obj_px} object px)")

        # full predictor call (includes model.extract_features, timed separately via the wrapper)
        with gtimer("predictor_total"):
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=False):
                out = predictor(
                    vid_w,
                    dep_w,
                    queries=None,
                    segm_mask=segm_mask,
                    grid_size=args.grid_size,
                    grid_query_frame=0,          # seed at this window's first frame
                    backward_tracking=False,     # windows only look forward from their start
                    predefined_intrs=K,
                )
        # device->host readback of the results
        with gtimer("d2h_readback"):
            d3d = {k: v[0].cpu().numpy() for k, v in out["trajs_3d_dict"].items()}
            w_uv = out["trajs_uv"][0].cpu().numpy()  # (Lw, n, 2) native-resolution pixel coords
        w_coords = d3d["coords"]   # (Lw, n, 3)
        w_colors = d3d["colors"]   # (n, 3)
        w_vis = d3d["vis"].astype(bool)  # (Lw, n)
        n_keep = w_coords.shape[1]
        if n_keep == 0:
            print(f"  WARNING: no query points fell inside mask; skipping window")
            continue
        print(f"  tracked {n_keep} object points")

        # First-occlusion cutoff (within this window, before padding): once a track
        # goes invisible it stays invisible for the rest of its window.
        if not args.keep_reappearing:
            w_vis = np.logical_and.accumulate(w_vis, axis=0)

        # place into full-length buffers: NaN coords / vis=False outside [s,e)
        full_coords = np.full((T, n_keep, 3), np.nan, dtype=np.float32)
        full_uv = np.full((T, n_keep, 2), np.nan, dtype=np.float32)
        full_vis = np.zeros((T, n_keep), dtype=bool)
        full_coords[s:e] = w_coords
        full_uv[s:e] = w_uv
        full_vis[s:e] = w_vis
        all_coords.append(full_coords)
        all_colors.append(w_colors.astype(np.float32))
        all_vis.append(full_vis)
        all_uv.append(full_uv)

    if not all_coords:
        raise RuntimeError("No windows produced tracks (all masks empty?).")

    # --- timing summary -----------------------------------------------------
    # predictor_total is the per-window "detection" cost; model.extract_features
    # is the encoder slice of it; h2d/d2h are the transfer overheads. Divide the
    # predictor mean by --win to get an approximate per-frame tracking cost.
    gtimer.report("Speed breakdown")
    pt = gtimer.times.get("predictor_total", [])
    if len(pt) > 1:
        mean_win = sum(pt[1:]) / len(pt[1:])
        print(f"  -> per-frame tracking (predictor / win={args.win}): "
              f"{mean_win / args.win:.2f} ms/frame ({1000.0 * args.win / mean_win:.1f} frames/s within a window)")

    # --- merge (union along the track axis) ---------------------------------
    coords = np.concatenate(all_coords, axis=1)  # (T, N_total, 3)
    colors = np.concatenate(all_colors, axis=0)  # (N_total, 3)
    vis = np.concatenate(all_vis, axis=1)         # (T, N_total)
    uv = np.concatenate(all_uv, axis=1)           # (T, N_total, 2)
    print(f"\nMerged: {coords.shape[1]} tracks over {T} frames "
          f"(~{coords.nbytes / 1e6:.0f} MB coords)")

    save_dir = args.output_path
    os.makedirs(save_dir, exist_ok=True)
    out_path = os.path.join(save_dir, "dense_3d_track.pkl")
    with open(out_path, "wb") as h:
        pickle.dump({"coords": coords, "colors": colors, "vis": vis}, h, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Saved {out_path}")

    # --- 2D overlay video ---------------------------------------------------
    # rainbow colors (keyed by seed-frame position) read best against the object;
    # 8-frame trailing line, 10 fps.
    viz_colors = rainbow_colors_by_position(uv, vis)
    vid_out = render_2d_overlay(video_np, uv, vis, viz_colors, trace=8)
    mp4_path = os.path.join(save_dir, "tracks_2d.mp4")
    media.write_video(mp4_path, vid_out, fps=10)
    print(f"Saved {mp4_path}")


if __name__ == "__main__":
    main()
