#!/usr/bin/env python3
"""Unit test for carry-forward stitching identity bookkeeping in track_windowed.py.

Runs on CPU with a STUB predictor (no model / GPU / masks needed): we feed crafted
per-window outputs and assert the stitcher (a) continues carried identities across
windows, (b) creates new identities for un-deduped fresh points, (c) takes each
frame's value from the freshest window covering it, and (d) that --merge-radius
dedups new grid points near carried ones.

Run:  python preprocess/_verify_stitch.py
"""
import os
import sys
import tempfile

import cv2
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import preprocess.track_windowed as tw


class _NullTimer:
    """Stand-in for GPUTimer: `with gtimer('x'):` is a no-op (no CUDA)."""
    def __call__(self, label):
        return self
    def __enter__(self):
        return self
    def __exit__(self, *exc):
        return False


class StubPredictor:
    """Returns, for whatever queries it's given, tracks whose coords/uv equal the
    query (x,y) held constant over the window, depth channel = a per-window tag so
    we can tell which window produced a given frame's value. vis all True.

    This lets the test assert both identity continuity (a carried query keeps its
    x,y) and freshest-wins (the tag at each frame is the latest window's tag)."""
    def __init__(self):
        self.tag = 0.0

    def __call__(self, vid_w, dep_w, queries=None, segm_mask=None, grid_size=0,
                 grid_query_frame=0, backward_tracking=False, predefined_intrs=None):
        self.tag += 1.0
        Lw = vid_w.shape[1]
        q = queries[0].detach().cpu().numpy()  # (N,3) frame,x,y
        N = q.shape[0]
        coords = np.zeros((Lw, N, 3), dtype=np.float32)
        coords[:, :, 0] = q[None, :, 1]   # x
        coords[:, :, 1] = q[None, :, 2]   # y
        coords[:, :, 2] = self.tag        # window tag in the z channel
        uv = coords[:, :, :2].copy()
        vis = np.ones((Lw, N), dtype=bool)
        colors = np.full((N, 3), self.tag, dtype=np.float32)
        return {
            "trajs_3d_dict": {
                "coords": torch.from_numpy(coords[None]),
                "colors": torch.from_numpy(colors[None]),
                "vis": torch.from_numpy(vis[None]),
            },
            "trajs_uv": torch.from_numpy(uv[None]),
        }


def _write_full_mask(mask_dir, abs_frames, H, W):
    """One full-frame (all-object) mask per window-start abs frame."""
    os.makedirs(mask_dir, exist_ok=True)
    for f in abs_frames:
        cv2.imwrite(os.path.join(mask_dir, f"{f:05d}.png"),
                    np.full((H, W), 255, dtype=np.uint8))


def test_stitch_bookkeeping():
    H, W, T, grid_size = 48, 64, 20, 4
    video_np = np.zeros((T, H, W, 3), dtype=np.uint8)
    depth_np = np.ones((T, H, W), dtype=np.float32)  # all-valid depth
    K = torch.eye(3, dtype=torch.float32)
    # windows: 0-9, 5-14, 10-19  (win=10, stride=5) -> overlaps 5 frames
    windows = [(0, 10), (5, 15), (10, 20)]
    abs_frames = list(range(T))

    with tempfile.TemporaryDirectory() as td:
        mask_dir = os.path.join(td, "mask")
        _write_full_mask(mask_dir, [abs_frames[s] for s, _ in windows], H, W)
        pred = StubPredictor()
        coords, colors, vis, uv = tw._run_windows_stitched(
            pred, _NullTimer(), windows, abs_frames, video_np, depth_np, mask_dir,
            K, H, W, T, grid_size, merge_radius=W / grid_size, keep_reappearing=False)

    N = coords.shape[1]
    G = grid_size * grid_size  # 16 grid points, all in the full mask

    # (a) window 1 seeds G fresh identities; every one is visible at seam frame 5
    #     with valid depth, so all are carried -> NO new identities added by later
    #     windows (fresh points all dedup against carried, being the same grid).
    assert N == G, f"expected {G} identities (all carried forward), got {N}"

    # (b) each identity is visible across ALL frames 0..T-1 (carried unbroken)
    assert vis.all(), "carried identities should stay visible across the whole clip"

    # (c) freshest-wins: z-tag at each frame is the LATEST window covering it.
    #     frames 0-4 -> win1 (tag 1); 5-9 -> win2 (tag 2); 10-19 -> win3 (tag 3).
    z = coords[:, 0, 2]  # any identity; all share the tag pattern
    assert np.all(z[0:5] == 1.0), f"frames 0-4 should be tag 1, got {z[0:5]}"
    assert np.all(z[5:10] == 2.0), f"frames 5-9 should be tag 2, got {z[5:10]}"
    assert np.all(z[10:20] == 3.0), f"frames 10-19 should be tag 3, got {z[10:20]}"

    # (d) identity continuity: an identity's x,y is constant across the seam
    #     (carried query injected its own position) -> no jump at frames 4->5, 9->10.
    for t in (4, 9):
        assert np.allclose(uv[t], uv[t + 1]), f"seam jump at frame {t}->{t+1}"

    # (e) colors kept from FIRST seed (tag 1), not overwritten by later windows.
    assert np.all(colors == 1.0), f"colors should be from first seed (tag 1), got {colors[:3]}"

    print("test_stitch_bookkeeping PASSED "
          f"(N={N} identities, all carried, freshest-wins + seam-continuity verified)")


def test_merge_radius_dedup():
    """--merge-radius controls whether fresh grid points near carried points are
    dropped. Uses a stub that DRIFTS tracked points by half a grid cell, so at the
    seam the carried positions are offset from the fresh grid: a small radius lets
    every fresh point survive (identities added), a large radius suppresses them."""
    H, W, T, grid_size = 48, 64, 6, 4
    cell = W / grid_size            # 16 px grid spacing
    offset = cell / 2               # 8 px: carried points land between grid nodes
    video_np = np.zeros((T, H, W, 3), dtype=np.uint8)
    depth_np = np.ones((T, H, W), dtype=np.float32)
    K = torch.eye(3, dtype=torch.float32)
    windows = [(0, 4), (2, 6)]      # overlapping (win=4, stride=2); seam frame 2 in window 1
    abs_frames = list(range(T))
    G = grid_size * grid_size

    class DriftStub(StubPredictor):
        def __call__(self, vid_w, dep_w, queries=None, **kw):
            out = super().__call__(vid_w, dep_w, queries=queries, **kw)
            out["trajs_3d_dict"]["coords"][..., 0] += offset  # shift x
            out["trajs_uv"][..., 0] += offset
            return out

    with tempfile.TemporaryDirectory() as td:
        mask_dir = os.path.join(td, "mask")
        _write_full_mask(mask_dir, [abs_frames[s] for s, _ in windows], H, W)
        # small radius (2 < 8px offset): every window-2 fresh grid point is far
        # enough from the drifted carried points to survive -> G carried + G fresh.
        c_small, *_ = tw._run_windows_stitched(
            DriftStub(), _NullTimer(), windows, abs_frames, video_np, depth_np,
            mask_dir, K, H, W, T, grid_size, merge_radius=2.0, keep_reappearing=False)
        # large radius (12 > 8px offset): every fresh point is within radius of a
        # carried point -> all deduped -> only the G carried identities remain.
        c_large, *_ = tw._run_windows_stitched(
            DriftStub(), _NullTimer(), windows, abs_frames, video_np, depth_np,
            mask_dir, K, H, W, T, grid_size, merge_radius=12.0, keep_reappearing=False)

    # Both runs share the identical carried set (same drift), so the DIFFERENCE in
    # identity count isolates the dedup effect on fresh points: a small radius keeps
    # fresh points that a large radius suppresses. (Exact counts vary with edge
    # geometry — a carried point drifting off-frame is correctly dropped — so assert
    # the monotonic relationship, which is what --merge-radius is supposed to do.)
    assert c_small.shape[1] > c_large.shape[1], (
        f"smaller merge-radius must keep more identities: {c_small.shape[1]} !> {c_large.shape[1]}")
    assert c_small.shape[1] > G, f"small radius should add fresh beyond carried: {c_small.shape[1]} <= {G}"
    print(f"test_merge_radius_dedup PASSED (small->{c_small.shape[1]} > large->{c_large.shape[1]} identities)")


def test_no_stitch_matches_union_shape():
    """--no-stitch path returns a valid union of independent windows."""
    H, W, T, grid_size = 48, 64, 6, 4
    video_np = np.zeros((T, H, W, 3), dtype=np.uint8)
    depth_np = np.ones((T, H, W), dtype=np.float32)
    K = torch.eye(3, dtype=torch.float32)
    windows = [(0, 3), (3, 6)]
    abs_frames = list(range(T))

    with tempfile.TemporaryDirectory() as td:
        mask_dir = os.path.join(td, "mask")
        _write_full_mask(mask_dir, [abs_frames[s] for s, _ in windows], H, W)
        # union path seeds grid∩mask via segm_mask; our StubPredictor echoes queries,
        # but the union path passes segm_mask (queries=None). The real predictor would
        # build the grid; the stub can't, so we only assert the branch runs and shapes
        # are internally consistent by echoing a single query it does receive: none.
        # Instead, validate union bookkeeping directly with a stub that ignores queries.
        class GridStub(StubPredictor):
            def __call__(self, vid_w, dep_w, queries=None, segm_mask=None, grid_size=0,
                         grid_query_frame=0, backward_tracking=False, predefined_intrs=None):
                self.tag += 1.0
                Lw = vid_w.shape[1]
                N = grid_size * grid_size
                coords = np.zeros((Lw, N, 3), dtype=np.float32)
                coords[:, :, 2] = self.tag
                uv = coords[:, :, :2].copy()
                vis = np.ones((Lw, N), dtype=bool)
                colors = np.full((N, 3), self.tag, dtype=np.float32)
                return {
                    "trajs_3d_dict": {
                        "coords": torch.from_numpy(coords[None]),
                        "colors": torch.from_numpy(colors[None]),
                        "vis": torch.from_numpy(vis[None]),
                    },
                    "trajs_uv": torch.from_numpy(uv[None]),
                }
        coords, colors, vis, uv = tw._run_windows_union(
            GridStub(), _NullTimer(), windows, abs_frames, video_np, depth_np,
            mask_dir, K, H, W, T, grid_size, keep_reappearing=False)

    G = grid_size * grid_size
    # union: 2 windows x G tracks, concatenated -> 2G columns; each active only in
    # its own window (NaN elsewhere).
    assert coords.shape == (T, 2 * G, 3), f"union coords shape {coords.shape}"
    assert np.isnan(coords[3:, :G, 0]).all(), "window-1 tracks must be NaN after frame 3"
    assert np.isnan(coords[:3, G:, 0]).all(), "window-2 tracks must be NaN before frame 3"
    print(f"test_no_stitch_matches_union_shape PASSED (2x{G}={2*G} independent tracks)")


if __name__ == "__main__":
    test_stitch_bookkeeping()
    test_merge_radius_dedup()
    test_no_stitch_matches_union_shape()
    print("\nAll stitching unit tests PASSED")
