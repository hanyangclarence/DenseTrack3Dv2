#!/usr/bin/env python3
"""Compute per-channel normalization statistics for the object-flow intent model.

LUCID standardizes its regression targets per channel using TRAIN-set statistics
(§A.2.2). We follow that, and also standardize the hand articulation inputs (raw
joint angles / keypoints have wide, uneven ranges). Rather than fit lazily inside
the Dataset, we compute the stats ONCE here into a small .npz that both the train
and eval Datasets load via `stats=<path>` -- reproducible, and decoupled from
training.

Stats are grouped by what they normalize, because the pieces are semantically
different and must be treated differently:
  - dxyz_mean / dxyz_std   (3,)  per-step target displacement Delta_tau = x_tau - x_{tau-1}
                                 (x_0 = x0), over VISIBLE steps only -> the LOSS
                                 standardizes the displacement it regresses.
  - ergo_mean  / ergo_std  (20,) ergonomics joint angles (deg) -> whiten the
                                 articulation input when articulation="ergonomics".
  - node_mean  / node_std  (72,) flattened raw_node_pose keypoints 1..24 (m) -> whiten
                                 the articulation input when articulation="raw_node_pose".
  - cloud_mean (3,) + cloud_scale (scalar)  object-cloud xyz normalization: the network
                                 input is standardized as (cloud - cloud_mean) / cloud_scale.
The wrist-rotation block (M_rel as 6D / matrix) is deliberately NOT normalized -- it
is already a bounded rotation representation with unit-norm structure -- so no stats
are produced for it.

Because ergo/node stats depend only on the hand data (not on which articulation the
Dataset is configured for), BOTH are always computed, so one stats file serves either
articulation choice. dxyz depends on the rate, so pass the same --stride-hz you train
with. Streaming sums keep memory flat over the whole dataset (~125k windows).

Usage:
  python scripts/compute_flow_stats.py --data-root /home/.../inhand_manipulation \
      --out data/flow_stats.npz [--stride-hz 4] [--t-pred 8] [--t-hist 4] [--n-query 16]
  # strict train-only stats: pass the split's episode dirs instead of the root
  python scripts/compute_flow_stats.py --episodes-file train_eps.txt --out data/flow_stats_train.npz
"""
import argparse
import os
import sys

import numpy as np
from tqdm.auto import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.flow_window_dataset import FlowWindowDataset


class Accum:
    """Streaming per-channel mean/std via running count + sum + sum-of-squares."""

    def __init__(self, dim):
        self.n = 0
        self.s = np.zeros(dim, np.float64)
        self.ss = np.zeros(dim, np.float64)

    def add(self, x):
        """x: (M, dim) rows to fold in (NaN rows should be filtered by the caller)."""
        x = x.reshape(-1, self.s.shape[0]).astype(np.float64)
        self.n += x.shape[0]
        self.s += x.sum(0)
        self.ss += (x * x).sum(0)

    def mean_std(self, floor=1e-6):
        if self.n == 0:
            return np.zeros_like(self.s), np.ones_like(self.s)
        mean = self.s / self.n
        var = np.maximum(self.ss / self.n - mean * mean, 0.0)
        return mean.astype(np.float32), np.maximum(np.sqrt(var), floor).astype(np.float32)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = p.add_mutually_exclusive_group()
    src.add_argument("--data-root", default="/home/labeng/yanghan/data/inhand_manipulation",
                     help="dataset root (all clips' episode_* with a clouds.npz)")
    src.add_argument("--clip", help="a single clip dir")
    src.add_argument("--episodes-file", help="text file of episode dirs, one per line "
                     "(e.g. a train split) -- for strict train-only stats")
    p.add_argument("--out", default="data/flow_stats.npz", help="output .npz path")
    p.add_argument("--stride-hz", type=int, default=4, help="rate (must match training)")
    p.add_argument("--t-pred", type=int, default=8)
    p.add_argument("--t-hist", type=int, default=4)
    p.add_argument("--pred-pad", type=int, default=0)
    p.add_argument("--n-query", type=int, default=16)
    p.add_argument("--floor", type=float, default=1e-6, help="std floor (avoid /0 on flat channels)")
    return p.parse_args()


def main():
    args = parse_args()
    if args.episodes_file:
        with open(args.episodes_file) as f:
            source = [ln.strip() for ln in f if ln.strip()]
    else:
        source = args.clip or args.data_root

    # normalize=False -> the Dataset emits RAW features; we fit the stats over them.
    # Force raw_node_pose so BOTH ergo and node articulations are reachable... but the
    # Dataset only emits one articulation per instance. So compute cloud/dxyz from any
    # config, and read ergo/node straight from each episode's hand.pkl (below), which is
    # articulation-agnostic and avoids two Dataset passes.
    ds = FlowWindowDataset(source, stride_hz=args.stride_hz, t_pred=args.t_pred,
                           t_hist=args.t_hist, pred_pad=args.pred_pad, n_query=args.n_query,
                           use_wrist=False, normalize=False)
    print(f"Fitting stats over {len(ds)} windows in {len(ds.episodes)} episodes")

    dxyz = Accum(3)
    cloud = Accum(3)
    for i in tqdm(range(len(ds)), desc="windows (dxyz, cloud)"):
        it = ds[i]
        traj = np.concatenate([it["x0"][None], it["target"]], axis=0)   # (L+1, N, 3)
        d = np.diff(traj, axis=0)                                       # (L_pred, N, 3)
        vis = it["target_vis"]
        good = vis & np.isfinite(d).all(-1)
        dxyz.add(d[good])                                              # visible steps only
        obj = it["cloud"][:-args.n_query]                              # drop concatenated queries
        cloud.add(obj[np.isfinite(obj).all(-1)])

    # ergo / node articulation stats straight from hand.pkl (articulation-agnostic; both
    # produced so one file serves either Dataset articulation choice).
    import pickle
    ergo = Accum(20)
    node = Accum(72)
    for ep in tqdm(ds.episodes, desc="episodes (hand)"):
        with open(os.path.join(ep, "hand.pkl"), "rb") as f:
            hd = pickle.load(f)
        e = np.asarray(hd["ergonomics"], np.float64)                   # (T, 20)
        ergo.add(e[np.isfinite(e).all(-1)])
        n = np.asarray(hd["raw_node_pose"], np.float64)[:, 1:, :3].reshape(len(e), -1)  # (T, 72)
        node.add(n[np.isfinite(n).all(-1)])

    dxyz_m, dxyz_s = dxyz.mean_std(args.floor)
    cloud_m, cloud_s = cloud.mean_std(args.floor)
    ergo_m, ergo_s = ergo.mean_std(args.floor)
    node_m, node_s = node.mean_std(args.floor)
    cloud_scale = np.float32(np.sqrt((cloud_s.astype(np.float64) ** 2).mean()))

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    np.savez(args.out,
             dxyz_mean=dxyz_m, dxyz_std=dxyz_s,
             ergo_mean=ergo_m, ergo_std=ergo_s,
             node_mean=node_m, node_std=node_s,
             cloud_mean=cloud_m, cloud_std=cloud_s, cloud_scale=cloud_scale,
             stride_hz=np.int32(args.stride_hz))
    print(f"\nwrote {args.out}")
    print(f"  dxyz  mean {dxyz_m.round(5)}  std {dxyz_s.round(5)}  (m/step, visible)")
    print(f"  cloud mean {cloud_m.round(4)}  std {cloud_s.round(4)}  (m, per-ch provenance)")
    print(f"  cloud_scale {float(cloud_scale):.4f} m  (isotropic; the divisor used to normalize)")
    print(f"  ergo  std range [{ergo_s.min():.2f}, {ergo_s.max():.2f}] deg  (20 ch)")
    print(f"  node  std range [{node_s.min():.4f}, {node_s.max():.4f}] m    (72 ch)")


if __name__ == "__main__":
    main()
