"""Dense 3D tracking visualizer with trajectory trails (viser).

Adapted from Track4World/visualization/vis_3d_efep.py for DenseTrack3Dv2's
dense_3d_track.pkl output. Renders, per frame:
  - the point cloud (toggleable), and
  - trajectory trails: a fading tail of each tracked point's last N frames,
    drawn as line segments and HSV-colored by the point's start position.

Data (from demo.py):
  coords (T, N, 3) float32 metric XYZ in metres
  colors (N, 3)    float32 0-255 RGB
  vis    (T, N)    bool/float visibility

Only numpy + matplotlib + viser are required (no open3d/faiss/scipy).
"""

import logging
import os
import pickle
import sys
import time
from typing import List, Optional

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import matplotlib.pyplot as plt
import numpy as np
import tyro
import viser
from tqdm.auto import tqdm

# viser's websocket server logs a full traceback for every stray/incomplete
# connection (browser preconnects, share proxy, port probes). Harmless; silence it.
logging.getLogger("websockets").setLevel(logging.CRITICAL)


# --- Trajectory smoothing (ported from Track4World/vis_3d_efep.py) -----------
# Pure-numpy so we don't pull in scipy. Same idea: temporal Gaussian per point,
# applied only over continuous visible segments (never across an occlusion gap).

def _gaussian_kernel1d(sigma: float) -> np.ndarray:
    radius = max(int(3 * sigma + 0.5), 1)
    x = np.arange(-radius, radius + 1, dtype=np.float32)
    k = np.exp(-(x ** 2) / (2 * sigma ** 2))
    return k / k.sum()


def _smooth_segment(data: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """Gaussian-smooth a single (L, 3) segment along time; 'nearest' edge handling."""
    r = len(kernel) // 2
    padded = np.pad(data, ((r, r), (0, 0)), mode="edge")  # clamp ends so they don't drift
    out = np.empty_like(data)
    for d in range(data.shape[1]):
        out[:, d] = np.convolve(padded[:, d], kernel, mode="valid")
    return out


def smooth_trajectories_temporal(trajs: np.ndarray, mask: np.ndarray, sigma: float = 2.0) -> np.ndarray:
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


def fill_trajectory_gaps(trajs: np.ndarray, mask: np.ndarray, max_gap: int = 3):
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


def main(
    filepath: str = "results/zed/zed_capture/dense_3d_track.pkl",
    mask_path: Optional[str] = None,
    mask_reso: tuple = (384, 512),
    share: bool = True,
    port: int = 8080,
    min_depth: float = 0.0,
    max_depth: float = 0.5,
    point_size: float = 0.002,
    line_width: float = 1.5,
    max_traj_length: int = 12,
    max_displacement: float = 0.1,
    smooth_sigma: float = 0.0,
    fill_gap: int = 0,
) -> None:
    """
    Args:
        filepath: path to dense_3d_track.pkl
        mask_path: optional first-frame object mask (image or .npy, nonzero=object).
            If given, only tracks whose query pixel falls inside the mask are shown.
            Only needed for a FULL (unmasked) pkl; skip it if demo.py already applied --mask_path.
        mask_reso: (H, W) of the model grid the full pkl was produced at (model_resolution).
        min_depth/max_depth: keep only points whose Z (metres) is in this range
        point_size: point cloud point radius (scene units = metres)
        line_width: trajectory line width (pixels)
        max_traj_length: number of past frames each trail retains
        max_displacement: hide a trail segment if a point jumps more than this (metres) between frames
        smooth_sigma: temporal Gaussian smoothing strength on the trails (0 = off; 2-3 = smooth ribbons)
        fill_gap: linearly interpolate occlusion gaps up to this many frames before smoothing (0 = off)
    """
    server = viser.ViserServer(port=port)
    if share:
        server.request_share_url()

    print("Loading dense 3D tracks!")
    with open(filepath, "rb") as handle:
        d = pickle.load(handle)

    coords = d["coords"].astype(np.float32)          # (T, N, 3)
    colors_rgb = d["colors"].astype(np.float32) / 255.0  # (N, 3)
    vis = d["vis"].astype(bool)                      # (T, N)
    T, N_full = coords.shape[:2]
    print(f"Num frames {T}, Num points {N_full}")

    # Optionally restrict to a first-frame object mask. A full pkl has one track
    # per pixel of the model grid (mask_reso, row-major), so a mask resized
    # (nearest) to that grid indexes straight onto the N axis.
    if mask_path is not None:
        H_m, W_m = mask_reso
        if N_full != H_m * W_m:
            raise ValueError(
                f"pkl has {N_full} tracks but mask_reso {mask_reso} implies {H_m * W_m}. "
                "This pkl was likely already masked, or produced at a different resolution."
            )
        mask = np.load(mask_path) if mask_path.endswith(".npy") else plt.imread(mask_path)
        mask = np.asarray(mask)
        if mask.ndim == 3:
            mask = mask[..., :3].max(axis=-1)
        mask_bin = (mask > 0).astype(np.uint8)
        # avoid a hard cv2 dependency: nearest-resize via index gather
        ys = (np.arange(H_m) * mask_bin.shape[0] / H_m).astype(int)
        xs = (np.arange(W_m) * mask_bin.shape[1] / W_m).astype(int)
        keep = mask_bin[np.ix_(ys, xs)].astype(bool).reshape(-1)
        coords, colors_rgb, vis = coords[:, keep], colors_rgb[keep], vis[:, keep]
        N_full = coords.shape[1]
        print(f"Applied object mask {mask_path}: {N_full} tracks kept")

    # Per-(frame, point) validity: visible AND depth within range.
    z = coords[..., 2]                               # (T, N)
    valid = vis & (z >= min_depth) & (z <= max_depth)  # (T, N)

    # --- Point cloud: one node per frame (visibility toggled during playback) ---
    point_nodes: List[viser.PointCloudHandle] = []
    for i in tqdm(range(T), desc="Point clouds"):
        m = valid[i]
        point_nodes.append(
            server.scene.add_point_cloud(
                name=f"/frames/t{i}/points",
                points=coords[i][m],
                colors=(colors_rgb[m] * 255).astype(np.uint8),
                point_size=point_size,
                point_shape="rounded",
                visible=(i == 0),
            )
        )

    # --- Trajectories: subsample points, HSV-color by start position ----------
    # Auto-thin only when there are many tracks (~6000 target): the dense 196k pkl
    # gets subsampled, a small object-only pkl keeps every track.
    downsample = max(1, N_full // 6000)
    sub = np.arange(0, N_full, downsample)           # subsampled point indices
    trajs = coords[:, sub].copy()                    # (T, Ns, 3)
    tvis = valid[:, sub].copy()                      # (T, Ns)
    Ns = sub.shape[0]
    print(f"Trajectories: {Ns} trails (downsample 1/{downsample})")

    # Optional offline smoothing of the trails (see helpers above). Gap-fill first
    # so brief occlusions become a straight bridge, then Gaussian-round the result.
    if fill_gap > 0:
        trajs, tvis = fill_trajectory_gaps(trajs, tvis, max_gap=fill_gap)
    if smooth_sigma > 0:
        trajs = smooth_trajectories_temporal(trajs, tvis, sigma=smooth_sigma)

    # Color each trail by its first-visible XYZ, mapped through HSV (matches reference look).
    first_vis = np.argmax(tvis, axis=0)              # (Ns,)
    never = ~np.any(tvis, axis=0)
    first_vis[never] = 0
    idx = np.arange(Ns)
    first_xyz = trajs[first_vis, idx]                # (Ns, 3)
    first_xyz[never] = np.nan
    xyz_min = np.nanmin(first_xyz, axis=0)
    xyz_max = np.nanmax(first_xyz, axis=0)
    xyz_norm = (first_xyz - xyz_min) / (xyz_max - xyz_min + 1e-6)
    scalar = np.nansum(xyz_norm, axis=1)
    scalar = (scalar - np.nanmin(scalar)) / (np.nanmax(scalar) - np.nanmin(scalar) + 1e-6)
    sort_idx = np.argsort(scalar)
    colors_hsv = plt.cm.hsv(np.linspace(0, 1, Ns))[:, :3]
    traj_colors = (colors_hsv[np.argsort(sort_idx)] * 255).astype(np.uint8)  # (Ns, 3)

    line_node = server.scene.add_line_segments(
        name="/trajectories",
        points=np.zeros((1, 2, 3), dtype=np.float32),
        colors=np.zeros((1, 2, 3), dtype=np.uint8),
        line_width=line_width,
        visible=True,
    )

    # --- GUI ------------------------------------------------------------------
    with server.gui.add_folder("Playback"):
        gui_point_size = server.gui.add_slider("Point size", min=0.0005, max=0.05, step=0.0005, initial_value=point_size)
        gui_line_width = server.gui.add_slider("Line width", min=0.1, max=10, step=0.1, initial_value=line_width)
        gui_timestep = server.gui.add_slider("Timestep", min=0, max=T - 1, step=1, initial_value=0, disabled=True)
        gui_playing = server.gui.add_checkbox("Playing", True)
        gui_framerate = server.gui.add_slider("FPS", min=1, max=60, step=1, initial_value=12)
        gui_max_traj_length = server.gui.add_slider("Trail length", min=1, max=T, step=1, initial_value=max_traj_length)
        gui_max_disp = server.gui.add_slider("Max displacement", min=0.001, max=2.0, step=0.001, initial_value=max_displacement)
        gui_vis_mode = server.gui.add_button_group("Mode", ("PointCloud", "Tracking", "Both"))
        gui_vis_mode.value = "Both"

    def apply_vis_mode(mode: str, t: int):
        for i, node in enumerate(point_nodes):
            node.visible = (i == t) and (mode in ("PointCloud", "Both"))
        line_node.visible = mode in ("Tracking", "Both")

    # --- Trail history --------------------------------------------------------
    hist_pos: List[np.ndarray] = []   # each (M, 2, 3)
    hist_ind: List[np.ndarray] = []   # each (M,) trail ids
    hist_col: List[np.ndarray] = []   # each (M, 2, 3)

    def rebuild_trails(t_curr: int, t_prev: int):
        if not gui_vis_mode.value in ("Tracking", "Both"):
            return
        # Reset when looping back to the start.
        if t_curr == 0 and t_prev != 0:
            hist_pos.clear(); hist_ind.clear(); hist_col.clear()
            line_node.points = np.zeros((0, 2, 3), dtype=np.float32)
            return
        if t_curr <= 0:
            return

        p1 = trajs[t_curr - 1]                       # (Ns, 3)
        p2 = trajs[t_curr]
        seg_valid = tvis[t_curr - 1] & tvis[t_curr]
        if np.any(seg_valid):
            ids = idx[seg_valid]
            a, b = p1[seg_valid], p2[seg_valid]
            # Drop teleport segments (invalid-depth jumps).
            dist = np.linalg.norm(b - a, axis=1)
            keep = dist < gui_max_disp.value
            if np.any(keep):
                segs = np.stack([a[keep], b[keep]], axis=1)      # (M, 2, 3)
                cols = traj_colors[ids[keep]]
                seg_cols = np.stack([cols, cols], axis=1)         # (M, 2, 3)
                hist_pos.append(segs); hist_ind.append(ids[keep]); hist_col.append(seg_cols)
                active = ids[keep]
            else:
                active = np.array([], dtype=int)
        else:
            active = np.array([], dtype=int)

        # Trim to trail length.
        while len(hist_pos) > int(gui_max_traj_length.value):
            hist_pos.pop(0); hist_ind.pop(0); hist_col.pop(0)

        # Render only trails whose point is currently active (fading tail).
        if hist_pos and active.size:
            active_lookup = np.zeros(Ns, dtype=bool)
            active_lookup[active] = True
            pos_list, col_list = [], []
            for h_pos, h_ind, h_col in zip(hist_pos, hist_ind, hist_col):
                km = active_lookup[h_ind]
                if np.any(km):
                    pos_list.append(h_pos[km]); col_list.append(h_col[km])
            if pos_list:
                line_node.points = np.concatenate(pos_list, axis=0)
                line_node.colors = np.concatenate(col_list, axis=0)
            else:
                line_node.points = np.zeros((0, 2, 3), dtype=np.float32)
        else:
            line_node.points = np.zeros((0, 2, 3), dtype=np.float32)

    apply_vis_mode(gui_vis_mode.value, 0)

    @gui_vis_mode.on_click
    def _(_):
        apply_vis_mode(gui_vis_mode.value, gui_timestep.value)

    # --- Main loop ------------------------------------------------------------
    prev = gui_timestep.value
    while True:
        if gui_playing.value:
            gui_timestep.value = (gui_timestep.value + 1) % T
        t = gui_timestep.value

        line_node.line_width = gui_line_width.value
        if gui_point_size.value != point_nodes[t].point_size:
            for node in point_nodes:
                node.point_size = gui_point_size.value

        if t != prev:
            if gui_vis_mode.value in ("PointCloud", "Both"):
                point_nodes[prev].visible = False
                point_nodes[t].visible = True
            else:
                point_nodes[prev].visible = False
            rebuild_trails(t, prev)
            prev = t

        time.sleep(1.0 / gui_framerate.value)


if __name__ == "__main__":
    tyro.cli(main)
