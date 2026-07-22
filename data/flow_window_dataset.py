#!/usr/bin/env python3
"""Online window sampling for the object-flow intent model (spec §5.2).

The offline pass (`scripts/gen_flow_labels.py`) precomputed, per episode, the object
point cloud at every frame -> `clouds.npz`. This Dataset does the cheap, sweepable
part on the fly: for a present frame `t`, slice one prediction window out of the
already-on-disk `object_flow.pkl` (targets), `clouds.npz` (state cloud), and
`hand.pkl` (action), with rate downsampling, query sampling, and `M_rel` derivation
all happening here so they cost nothing to change.

One item = one window anchored at a present frame `t` (spec §4):

    x_{n, t:t+L_pred} = f( P_t, q_{t-T_hist:t}, q_{t:t+L_pred} )

  - state  P_t         : the precomputed object cloud at t  (P, 3), camera metres;
  - action q_future    : hand pose over the predicted horizon (L_pred steps);
  - cue    q_hist      : hand pose over the preceding T_hist steps (current velocity);
  - target             : the N query points' camera-frame positions + visibility.

Rate. The native capture is ~30 fps; `stride_hz` decimates it. stride_hz=4 -> ~8 Hz
(T_pred=8 ~ 1 s), stride_hz=2 -> ~15 Hz (T_pred=16). A window therefore spans native
frames  t - T_hist*s : t + L_pred*s  (step s = stride_hz), all present in the clip.

Query points (spec §5.2 step 3). Sampled from tracks visible at t; their positions
come from `object_flow.pkl` coords. They are NOT required to be members of the
precomputed cloud (tracker uv+depth lift vs. back-projected masked depth are two
samplings of the same surface) -- instead the N query seeds are CONCATENATED into the
cloud, so `cloud` has shape (P+N, 3) and every query point is guaranteed a scene
token. `x0` (the seeds) is also the last N rows of `cloud`.

Hand action (spec §6.3). Per frame: articulation (`ergonomics` or flattened
`raw_node_pose` keypoints) optionally concatenated with the anchor-relative wrist
rotation M_rel(tau) = R(q_t)^T R(q_tau) (anchor = present frame t), as 6D or 3x3.
The default (ergonomics + M_rel-6D) gives d_q = 26.

Item dict (numpy; a collate wrapper can torch-ify):
    cloud       (P+N, 3)        float32  present-frame object cloud, query seeds appended;
                                         cloud-standardized when normalize=True (network input)
    x0          (N, 3)          float32  query seeds x_{n,t}, always METRIC
    target      (L_pred, N, 3)  float32  camera-frame positions x_{n, t+1 : t+L_pred+1}, always METRIC
    target_vis  (L_pred, N)     bool     per-step visibility of each query point
    q_hist      (T_hist, d_q)   float32  hand pose cue (history)
    q_future    (L_pred, d_q)   float32  hand pose action (future)
    K           (4,)            float32  fx, fy, cx, cy (this episode's intrinsics)
    frame_meta  dict                     episode dir, present frame t, stride_hz, ...
"""
import glob
import os
import pickle
import sys
from typing import Optional, Union

import numpy as np
from torch.utils.data import Dataset

# repo root is the parent of data/ -- allow running as a script (python data/...py)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from preprocess.hand_frame_transforms import wrist_M_rel, placed_hand_camera
from densetrack3d.models.worldmodel.types import FlowItem


def matrix_to_rotation_6d(R: np.ndarray) -> np.ndarray:
    """(..., 3, 3) rotation -> (..., 6): the first two columns, flattened.

    The 6D continuous rotation representation (Zhou et al. 2019): drop the third
    column (recoverable by cross product), which avoids the discontinuities of
    quaternions/Euler and is friendlier to regress against / condition on."""
    return R[..., :, :2].reshape(*R.shape[:-2], 6)


def train_eval_split(
        episodes: Union[str, list[str]], val_frac: float = 0.15,
        seed: int = 0
        ) -> tuple[list[str], list[str]]:
    """Split episodes into (train_eps, eval_eps) BY EPISODE, not by window."""
    eps = FlowWindowDataset._resolve_episodes(episodes)
    if len(eps) <= 1:
        return eps, []
    perm = np.random.default_rng(seed).permutation(len(eps))
    n_val = max(1, int(round(len(eps) * val_frac)))
    val = {int(k) for k in perm[:n_val]}
    train_eps = [eps[i] for i in range(len(eps)) if i not in val]
    eval_eps = [eps[i] for i in range(len(eps)) if i in val]
    return train_eps, eval_eps


class FlowWindowDataset(Dataset):
    """Sliding-window samples over precomputed episodes (spec §5.2).

    Each episode must have `object_flow.pkl`, `hand.pkl`, `intrinsics.txt`, and a
    precomputed `clouds.npz` (run `scripts/gen_flow_labels.py` first). Every frame
    with a valid (non-NaN) cloud and a full window in bounds is a present-frame; the
    index is built once at construction.
    """

    def __init__(
            self, episodes: Union[str, list[str]], *, split: str = "all",
            val_frac: float = 0.15, split_seed: int = 0,
            stride_hz: int = 4, t_pred: int = 8, t_hist: int = 4, pred_pad: int = 0,
            n_query: int = 16, min_visible: Optional[int] = None, 
            stride_win: int = 1,
            articulation: str = "ergonomics", use_wrist: bool = True, 
            wrist_repr: str = "6d", 
            normalize: bool = True,
            stats: Optional[str] = None,
            seed: int = 0
            ):
        """
        episodes     : list of episode dirs, OR a clip dir, OR the dataset root
                       (auto-expanded to all episode_* that have a clouds.npz).
        split        : which episode partition to use -- "train", "eval", or "all"
        val_frac     : eval fraction of episodes when split != "all" (default 0.15).
        split_seed   : RNG seed for the episode partition (must match across train/eval).
        stride_hz    : native-frame decimation (4 -> ~8 Hz, 2 -> ~15 Hz).
        t_pred       : reported prediction horizon in decimated steps (~1 s).
        t_hist       : history horizon in decimated steps (~0.5 s), < t_pred.
        pred_pad     : extra predicted steps beyond t_pred (L_pred = t_pred + pred_pad).
        n_query      : query points sampled per window (N).
        min_visible  : a present-frame is kept only if at least this many tracks are visible at t
        stride_win   : present-frame stride over the episode (1 = every frame).
        articulation : "ergonomics" (20), "raw_node_pose" (24 hand-LOCAL keypoints x 3 = 72),
                       or "camera_node_pose" (24 keypoints x 3 = 72 in the CAMERA frame)
        use_wrist    : append the anchor-relative wrist rotation M_rel to each frame.
        wrist_repr   : "6d" (default) or "matrix" (9) -- ignored when use_wrist=False.
        normalize    : normalize=False should only be in scripts/compute_flow_stats.py
        stats        : path to the .npz written by scripts/compute_flow_stats.py
        seed         : base RNG seed (per-item seeds are derived so sampling is stable).
        """
        assert articulation in ("ergonomics", "raw_node_pose", "camera_node_pose")
        assert wrist_repr in ("6d", "matrix")
        # camera_node_pose bakes the wrist rotation into absolute keypoint positions, so an
        # explicit M_rel block would be redundant
        if articulation == "camera_node_pose":
            use_wrist = False
        assert split in ("all", "train", "eval")
        self.split = split
        self.stride_hz = int(stride_hz)
        self.t_pred = int(t_pred)
        self.t_hist = int(t_hist)
        self.pred_pad = int(pred_pad)
        self.l_pred = self.t_pred + self.pred_pad
        self.n_query = int(n_query)
        self.min_visible = int(min_visible) if min_visible is not None else max(1, self.n_query // 2)
        self.stride_win = int(stride_win)
        self.articulation = articulation
        self.use_wrist = bool(use_wrist)
        self.wrist_repr = wrist_repr
        self.normalize = bool(normalize)
        self.seed = int(seed)
        self._stats = dict(np.load(stats)) if stats is not None else None
        if self.normalize and self._stats is None:
            raise ValueError("normalize=True requires stats=<path to compute_flow_stats.py .npz>; "
                             "pass normalize=False to emit raw features.")

        if split == "all":
            self.episodes = self._resolve_episodes(episodes)
        else:
            train_eps, eval_eps = train_eval_split(episodes, val_frac=val_frac, seed=split_seed)
            self.episodes = train_eps if split == "train" else eval_eps
            if not self.episodes:
                raise RuntimeError(f"split='{split}' is empty (need >=2 episodes for a split; "
                                   f"got {len(self._resolve_episodes(episodes))}).")
        self._ep_cache = {}   # episode dir -> loaded arrays
        self.index, self._dropped, counts = [], 0, []
        for ei, ep in enumerate(self.episodes):
            for t, nvis in self._candidate_present_frames(ep):
                if nvis < self.min_visible:
                    self._dropped += 1
                    continue
                self.index.append((ei, t))
                counts.append(nvis)
        self._vis_counts = np.asarray(counts, dtype=np.int32)
        if not self.index:
            raise RuntimeError("No valid windows found (check clouds.npz exists, "
                               "horizons fit the clip lengths, and min_visible is not too high).")

    # ------------------------------------------------------------------ setup
    @staticmethod
    def _resolve_episodes(episodes: Union[str, list[str]]) -> list[str]:
        """Accept a list of episode dirs, a single clip dir, or the dataset root."""
        if isinstance(episodes, str):
            episodes = [episodes]
        out = []
        for p in episodes:
            if os.path.exists(os.path.join(p, "clouds.npz")):
                out.append(os.path.abspath(p))                 # an episode dir
            else:                                              # a clip or root dir
                eps = sorted(glob.glob(os.path.join(p, "episode_*"))) \
                    or sorted(glob.glob(os.path.join(p, "*", "episode_*")))
                out.extend(e for e in eps if os.path.exists(os.path.join(e, "clouds.npz")))
        return out

    def _span(self) -> tuple[int, int]:
        """Native-frame reach on each side of t: (back, forward) inclusive counts."""
        s = self.stride_hz
        return self.t_hist * s, self.l_pred * s

    def _candidate_present_frames(self, ep: str) -> list[tuple[int, int]]:
        """(t, n_visible) for frames whose window is in bounds and cloud at t is valid.

        The min_visible threshold is applied by the caller so it can also count drops."""
        arrs = self._load(ep)
        Tclip = arrs["Tclip"]
        back, fwd = self._span()
        cloud_ok = np.isfinite(arrs["clouds"][:, 0, 0])        # non-NaN cloud rows
        n_vis = arrs["vis"].sum(axis=1)                        # (T,) visible tracks per frame
        lo, hi = back, Tclip - fwd - 1                         # inclusive present-frame range
        return [(t, int(n_vis[t])) for t in range(lo, hi + 1, self.stride_win) if cloud_ok[t]]

    def coverage_summary(self) -> str:
        """Sanity report: how many candidate present-frames were dropped for having
        fewer than min_visible tracks, and the distribution of visible-track counts
        among the kept windows (so you can tell whether n_query is comfortably met)."""
        c = self._vis_counts
        pct = np.percentile(c, [0, 5, 25, 50]).astype(int) if c.size else []
        kept = len(self.index)
        return (f"windows: {kept} kept, {self._dropped} dropped "
                f"(< min_visible={self.min_visible} tracks at t)\n"
                f"visible tracks at t among kept -- min {pct[0] if len(pct) else '-'}, "
                f"p5 {pct[1] if len(pct) else '-'}, p25 {pct[2] if len(pct) else '-'}, "
                f"median {pct[3] if len(pct) else '-'}  (n_query={self.n_query}, "
                f"{int((c < self.n_query).sum())} kept windows need replacement sampling)")

    def _load(self, ep: str) -> dict:
        """Lazy-load + cache one episode's arrays (flow, hand, clouds, intrinsics).

        Keys: coords/vis/clouds/K/ergo/node/wrist_quat (np.ndarray) + Tclip (int)."""
        if ep in self._ep_cache:
            return self._ep_cache[ep]
        with open(os.path.join(ep, "object_flow.pkl"), "rb") as f:
            flow = pickle.load(f)
        with open(os.path.join(ep, "hand.pkl"), "rb") as f:
            hand = pickle.load(f)
        npz = np.load(os.path.join(ep, "clouds.npz"))
        coords = np.asarray(flow["coords"], dtype=np.float32)  # (T, Nf, 3) camera metres
        vis = np.asarray(flow["vis"], dtype=bool)              # (T, Nf)
        clouds = np.asarray(npz["clouds"], dtype=np.float32)   # (T, P, 3), NaN rows = empty
        K = np.asarray(npz["intrinsics"], dtype=np.float32)    # (4,) fx,fy,cx,cy

        ergo = np.asarray(hand["ergonomics"], dtype=np.float32)        # (T, 20)
        node = np.asarray(hand["raw_node_pose"], dtype=np.float32)     # (T, 25, 7)
        wrist_quat = np.asarray(hand["wrist_quat"], dtype=np.float32)  # (T, 4) xyzw
        # One shared frame count across every modality (all index-aligned, spec §2.3).
        Tclip = min(coords.shape[0], clouds.shape[0], ergo.shape[0])
        arrs = dict(coords=coords, vis=vis, clouds=clouds, K=K, ergo=ergo,
                    node=node, wrist_quat=wrist_quat, Tclip=Tclip)
        self._ep_cache[ep] = arrs
        return arrs

    # ------------------------------------------------------------------ access
    def __len__(self) -> int:
        return len(self.index)

    def _frame_grid(self, t: int) -> tuple[np.ndarray, np.ndarray]:
        """Decimated native-frame indices for history and prediction, anchored at t.

        Returns (hist_frames [T_hist], pred_frames [L_pred]). History is the T_hist
        steps strictly before t; prediction is the L_pred steps strictly after t. The
        present frame t itself is the state anchor / query seed and the M_rel anchor."""
        s = self.stride_hz
        hist = [t - (self.t_hist - k) * s for k in range(self.t_hist)]   # ..., t-2s, t-s
        pred = [t + (k + 1) * s for k in range(self.l_pred)]             # t+s, t+2s, ...
        return np.asarray(hist, dtype=np.int64), np.asarray(pred, dtype=np.int64)

    def _hand_features(self, arrs: dict, t: int, frames: np.ndarray) -> np.ndarray:
        """Assemble (len(frames), d_q) hand features for the given native frames.

        articulation (ergonomics / raw_node_pose / camera_node_pose) optionally concatenated
        with the anchor-relative wrist rotation M_rel(tau), anchor = present frame t (so history
        and future are both relative to now).

        - ergonomics       : (20) joint angles.
        - raw_node_pose    : (72) hand-LOCAL keypoints
        - camera_node_pose : (72) keypoints 1..24 placed in the CAMERA frame (M_rel anchored at
                             t, then P->camera)
        """
        if self.articulation == "camera_node_pose":
            idx = np.concatenate([[t], frames])                         # anchor first
            M = wrist_M_rel(arrs["wrist_quat"][idx], anchor=0)[1:]      # (F, 3, 3), rel to t
            node = arrs["node"][frames, :, :3].astype(np.float64)      # (F, 25, 3) local, node0=origin
            kp_cam = placed_hand_camera(node, M)[:, 1:]                # (F, 24, 3) camera; drop const wrist
            art = kp_cam.reshape(len(frames), -1).astype(np.float32)   # (F, 72)
            if self.normalize:                                         # SAME transform as the cloud
                art = (art.reshape(len(frames), 24, 3) - self._stats["cloud_mean"]) \
                    / self._stats["cloud_scale"]
                art = art.reshape(len(frames), -1).astype(np.float32)
            return art                                                 # no wrist block (implicit)

        if self.articulation == "ergonomics":
            art = arrs["ergo"][frames].astype(np.float32)                # (F, 20) deg
            key = "ergo"
        else:
            art = arrs["node"][frames, 1:, :3].reshape(len(frames), -1).astype(np.float32)  # (F, 72) m
            key = "node"
        if self.normalize:                                              # articulation only
            art = (art - self._stats[f"{key}_mean"]) / self._stats[f"{key}_std"]
        if not self.use_wrist:
            return art
        # M_rel over just this window's frames + the anchor, anchor-relative to t.
        idx = np.concatenate([[t], frames])                             # anchor first
        M = wrist_M_rel(arrs["wrist_quat"][idx], anchor=0)[1:]          # (F, 3, 3)
        wr = matrix_to_rotation_6d(M) if self.wrist_repr == "6d" \
            else M.reshape(len(frames), 9)                             # left raw (bounded)
        return np.concatenate([art, wr.astype(np.float32)], axis=1)     # (F, d_q)

    def __getitem__(self, i: int) -> FlowItem:
        ei, t = self.index[i]
        ep = self.episodes[ei]
        arrs = self._load(ep)
        rng = np.random.default_rng(self.seed + i)                      # stable per item
        coords, vis = arrs["coords"], arrs["vis"]
        hist_f, pred_f = self._frame_grid(t)

        # --- query points: sample N uniformly at random from tracks visible at t -----
        visible = np.nonzero(vis[t])[0]
        qidx = rng.choice(visible, self.n_query, replace=visible.size < self.n_query)

        x0 = coords[t, qidx]                                            # (N, 3) query seeds (METRIC)
        target = coords[pred_f][:, qidx]                               # (L_pred, N, 3) METRIC
        target_vis = vis[pred_f][:, qidx]                              # (L_pred, N)

        # --- state cloud: precomputed cloud at t, with query seeds concatenated ----
        cloud = arrs["clouds"][t]                                      # (P, 3) metric
        cloud = np.concatenate([cloud, x0], axis=0)                    # (P+N, 3) metric
        if self.normalize:
            cloud = (cloud - self._stats["cloud_mean"]) / self._stats["cloud_scale"]

        # --- hand: history (cue) + future (action) ---------------------------------
        q_hist = self._hand_features(arrs, t, hist_f)                  # (T_hist, d_q)
        q_future = self._hand_features(arrs, t, pred_f)                # (L_pred, d_q)

        item = dict(
            cloud=cloud.astype(np.float32),
            x0=x0.astype(np.float32),
            target=target.astype(np.float32),
            target_vis=target_vis,
            q_hist=q_hist.astype(np.float32),
            q_future=q_future.astype(np.float32),
            K=arrs["K"].astype(np.float32),
            frame_meta=dict(episode=ep, t=int(t), stride_hz=self.stride_hz,
                            t_pred=self.t_pred, l_pred=self.l_pred, query_idx=qidx),
        )
        if self.normalize:
            item["dxyz_mean"] = self._stats["dxyz_mean"].astype(np.float32)
            item["dxyz_std"] = self._stats["dxyz_std"].astype(np.float32)
        return item


# --------------------------------------------------------------------------- #
# Benchmark: construction (index build) + item-emit throughput.
#
#   python data/flow_window_dataset.py [--data-root DIR] [--num-items N] [--workers W]
#
# Reports three numbers that matter for training throughput:
#   1. construct  -- one-time index build; this also loads every episode's arrays
#                    into the process cache (all pkls + clouds.npz read once).
#   2. cold emit  -- first touch of each episode (already cached by construct, so this
#                    isolates the per-item slicing/M_rel cost, not disk I/O).
#   3. warm emit  -- random-access items with everything cached: the steady-state
#                    per-item cost a single-process loop would see.
#   4. DataLoader -- items/s through torch's DataLoader with W workers (default_collate
#                    stacks the array fields; frame_meta stays a list of dicts).
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    import argparse
    import time

    p = argparse.ArgumentParser(description="Benchmark FlowWindowDataset load + emit speed")
    p.add_argument("--data-root", default="/home/labeng/yanghan/data/inhand_manipulation",
                   help="dataset root (all clips' episode_* with a clouds.npz)")
    p.add_argument("--num-items", type=int, default=2000, help="items to time for warm throughput")
    p.add_argument("--workers", type=int, default=4, help="DataLoader workers (0 = main process)")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--stride-hz", type=int, default=4)
    p.add_argument("--n-query", type=int, default=16)
    args = p.parse_args()

    # 1. construction (index build + first read of every episode's arrays)
    t0 = time.perf_counter()
    ds = FlowWindowDataset(args.data_root, stride_hz=args.stride_hz, n_query=args.n_query)
    t_build = time.perf_counter() - t0
    n = len(ds)
    print(f"construct: {t_build:.2f} s  ({len(ds.episodes)} episodes, {n} windows, "
          f"{1000 * t_build / max(len(ds.episodes), 1):.1f} ms/episode)")
    print(ds.coverage_summary())

    # 2. cold emit: one item per episode boundary (arrays already cached from build,
    #    so this is pure per-item compute, not disk).
    probe = [ds.index.index(next(x for x in ds.index if x[0] == ei))
             for ei in range(len(ds.episodes))]
    t0 = time.perf_counter()
    for i in probe:
        _ = ds[i]
    t_cold = time.perf_counter() - t0
    print(f"cold emit: {1000 * t_cold / len(probe):.2f} ms/item over {len(probe)} episodes")

    # 3. warm emit: random-access items, single process, everything cached.
    rng = np.random.default_rng(0)
    idxs = rng.integers(0, n, size=min(args.num_items, n))
    t0 = time.perf_counter()
    for i in idxs:
        _ = ds[int(i)]
    t_warm = time.perf_counter() - t0
    print(f"warm emit: {1000 * t_warm / len(idxs):.3f} ms/item  "
          f"({len(idxs) / t_warm:.0f} items/s single-process) over {len(idxs)} items")

    # 4. DataLoader throughput (multi-worker; each worker re-reads arrays into its own
    #    cache on first touch, so the first batches pay that once-per-worker cost).
    from torch.utils.data import DataLoader
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True,
                    num_workers=args.workers, persistent_workers=args.workers > 0)
    n_batches = max(1, args.num_items // args.batch_size)
    it = iter(dl)
    batch = next(it)                                        # warm up workers (excluded)
    t0 = time.perf_counter()
    seen = 0
    for _ in range(n_batches):
        try:
            batch = next(it)
        except StopIteration:
            break
        seen += args.batch_size
    t_dl = time.perf_counter() - t0
    print(f"DataLoader: {seen / max(t_dl, 1e-9):.0f} items/s "
          f"(batch {args.batch_size}, {args.workers} workers) -- "
          f"cloud batch shape {tuple(batch['cloud'].shape)}")
