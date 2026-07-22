#!/usr/bin/env python3
"""Qualitative 2D-video viz of the object-flow intent model's predictions.

The sweep (docs/intent_experiments.md) measures the model quantitatively (ADE/FDE).
This script is the QUALITATIVE companion: for one inference sample (= one dataset
window, i.e. present frame t in one episode) it renders an mp4 that plays the window's
history+future timeline as a 2D video with, side by side,

    [ GROUND-TRUTH flow on RGB | PREDICTED flow on RGB ]

so the model's flow can be eyeballed against the real object motion -- catching failure
modes (drift, mode-averaged "blurry" flow) that a scalar ADE hides.

Timeline (mirrors the viser viewers data/viz_flow_window_item.py):
  - HISTORY steps: the N query points sit FROZEN at x0 (dimmed), hand winds up off-screen;
  - PRESENT step : the seam (flow == 0), where the state cloud P_t lives;
  - FUTURE steps : query points follow the flow -- GT panel along `target`, Pred panel along
                   the model's `x_pred` -- each drawing a growing trail.
Both panels share ONE per-point colour, so query point i is the same hue on the left and
right and divergence is read at a glance.

Everything is glue over verified repo pieces (reused, not reimplemented):
  - inference : scripts/train_intent.load_ema_model / collate + IntentModel.predict_trajectory
  - reproject : postprocess/smooth_object_motion.reproject  (metric xyz -> pixels, NaN-safe)
  - draw      : preprocess/track_windowed.render_2d_overlay / rainbow_colors_by_position
  - video     : mediapy.write_video  (repo standard)

Frame alignment: color.mp4 frame index == object_flow coords index t (1:1, verified), image
cropped to 640x720, and each item's K already carries the crop-adjusted intrinsics -- so
reproject(metric, vis, item["K"]) lands pixels on the right color.mp4 frame.

Run in densetrack3d:
  /home/labeng/miniconda3/envs/densetrack3d/bin/python scripts/viz_intent_predictions.py \
      --ckpt logdirs/intent/ckpts/epoch009-ade0.0043.ckpt --num-samples 4
  # a specific window:
  ... --episode /home/labeng/yanghan/data/inhand_manipulation/0718_sphere_small/episode_21 --frame 400
"""
import argparse
import os
import sys

import cv2
import mediapy as media
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.flow_window_dataset import FlowWindowDataset
from densetrack3d.models.worldmodel import IntentModel
from densetrack3d.models.worldmodel.types import FlowItem
from postprocess.smooth_object_motion import reproject
from preprocess.track_windowed import render_2d_overlay, rainbow_colors_by_position
from scripts.train_intent import FlowWindowDataModule, collate, load_ema_model


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt", required=True, help="intent-model checkpoint (EMA weights loaded)")
    p.add_argument("--out-dir", default="logdirs/intent/viz", help="output dir for mp4s")
    p.add_argument("--num-samples", type=int, default=6, help="random eval-split windows to render")
    p.add_argument("--seed", type=int, default=0, help="which random windows are picked")
    p.add_argument("--split", default="eval", choices=("eval", "train", "all"),
                   help="which episode partition to sample from (default eval = held-out)")
    p.add_argument("--episode", default=None,
                   help="render a specific episode dir (overrides random sampling)")
    p.add_argument("--frame", type=int, default=None,
                   help="present native frame t within --episode (default: a random valid one)")
    p.add_argument("--fps", type=int, default=6, help="output mp4 fps (short window; slow to read)")
    p.add_argument("--trace", type=int, default=None,
                   help="trail length in frames (default: full timeline = growing trail)")
    p.add_argument("--pred-vis-gate", action="store_true",
                   help="hide predicted future points whose vis_prob < 0.5 (default: show all)")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


# --------------------------------------------------------------------------- #
# Dataset reconstruction (rebuild the EXACT split the checkpoint reported on)
# --------------------------------------------------------------------------- #
def _dh_from_ckpt(ck: dict) -> dict:
    """The datamodule hyperparameters saved in the checkpoint (drop private keys)."""
    return {k: v for k, v in ck["datamodule_hyper_parameters"].items() if not k.startswith("_")}


def build_dataset(ck: dict, args: argparse.Namespace) -> FlowWindowDataset:
    """A FlowWindowDataset matching the ckpt's data config.

    Default: the held-out eval split (what val/ade_m was measured on) via FlowWindowDataModule,
    which fills training defaults for any hand-repr keys an older ckpt omitted. With --episode,
    build a single-episode dataset instead, deriving the same window kwargs from those hparams
    so a future ckpt with a non-default hand representation still shapes correctly."""
    dm = FlowWindowDataModule(**_dh_from_ckpt(ck))
    dm.setup("validate")                                    # builds train_ds + val_ds
    if args.episode is None:
        ds = {"eval": dm.val_ds, "train": dm.train_ds}.get(args.split)
        if ds is None:                                      # split == "all": union via a fresh build
            h = dm.hparams
            ds = FlowWindowDataset(h.clip or h.data_root, split="all",
                                   **_window_kwargs(h))
        return ds
    return FlowWindowDataset(args.episode, split="all", **_window_kwargs(dm.hparams))


def _window_kwargs(h) -> dict:
    """The FlowWindowDataset kwargs implied by a DataModule's resolved hparams."""
    return dict(stats=h.stats if h.normalize else None, stride_hz=h.stride_hz,
                t_pred=h.t_pred, t_hist=h.t_hist, pred_pad=h.pred_pad, n_query=h.n_query,
                articulation=h.articulation, use_wrist=h.use_wrist,
                wrist_repr=h.wrist_repr, normalize=h.normalize)


def select_indices(ds: FlowWindowDataset, args: argparse.Namespace) -> list[int]:
    """Dataset indices to render: random-N (seeded) or the one matching --episode/--frame."""
    if args.episode is not None and args.frame is not None:
        matches = [i for i, (_, t) in enumerate(ds.index) if t == args.frame]
        if not matches:
            valid = sorted({t for _, t in ds.index})
            raise SystemExit(f"frame t={args.frame} is not a valid present-frame in {args.episode}. "
                             f"Valid t range [{valid[0]}, {valid[-1]}], e.g. {valid[:8]} ...")
        return matches[:1]
    rng = np.random.default_rng(args.seed)
    n = min(args.num_samples, len(ds))
    return rng.choice(len(ds), size=n, replace=False).tolist()


# --------------------------------------------------------------------------- #
# Inference on one item
# --------------------------------------------------------------------------- #
@torch.no_grad()
def run_inference(model: IntentModel, item: FlowItem, device: str) -> dict:
    """One item -> numpy dict {x0, gt, gt_vis, pred, vis_prob, K, meta}. All metric except vis."""
    batch = collate([item])
    batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
    out = model(batch)
    x_pred, vis_prob = model.predict_trajectory(batch, out)     # (1,H,N,3) metric, (1,H,N)
    H = x_pred.shape[1]
    return dict(
        x0=item["x0"].astype(np.float32),                        # (N,3) metric anchor
        gt=batch["target"][0, :H].cpu().numpy(),                 # (H,N,3) metric, NaN where occluded
        gt_vis=batch["target_vis"][0, :H].cpu().numpy().astype(bool),   # (H,N)
        pred=x_pred[0].cpu().numpy(),                            # (H,N,3) metric
        vis_prob=vis_prob[0].cpu().numpy(),                     # (H,N)
        K=tuple(float(x) for x in item["K"]),                   # (fx,fy,cx,cy)
        meta=item["frame_meta"],
    )


def sample_ade_mm(res: dict) -> float:
    """Per-sample mean displacement error (mm) over visible steps -- the cheap correctness oracle."""
    err = np.linalg.norm(res["pred"] - np.nan_to_num(res["gt"]), axis=-1)   # (H,N)
    m = res["gt_vis"].astype(np.float32)
    return 1000.0 * float((err * m).sum() / max(m.sum(), 1.0))


# --------------------------------------------------------------------------- #
# Timeline assembly + RGB frames
# --------------------------------------------------------------------------- #
def frame_grid(t: int, t_hist: int, s: int, H: int) -> list[int]:
    """Native frame indices for the render timeline: history -> present -> future (len S)."""
    hist = [t - (t_hist - k) * s for k in range(t_hist)]
    pred = [t + (j + 1) * s for j in range(H)]
    return hist + [t] + pred                                     # len = t_hist + 1 + H


def load_window_frames(video_path: str, native_indices: list[int]) -> np.ndarray:
    """(S,H,W,3) RGB uint8 read at the given native frame indices via seek (not a full decode)."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"could not open {video_path}")
    frames = []
    for idx in native_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, bgr = cap.read()
        if not ok:
            raise RuntimeError(f"failed to read frame {idx} of {video_path}")
        frames.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    cap.release()
    return np.stack(frames)


def build_timeline(res: dict, t_hist: int, s: int, pred_vis_gate: bool) -> tuple[
        np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Per-panel metric position stacks (S,N,3) + visibility (S,N) for GT and Pred.

    History+present steps freeze both panels at x0 (visible); future steps use gt/pred.
    Predicted xyz is always finite so it is drawable; --pred-vis-gate hides low-confidence steps.
    """
    x0, gt, gt_vis, pred, vis_prob = (res[k] for k in ("x0", "gt", "gt_vis", "pred", "vis_prob"))
    N, H = x0.shape[0], gt.shape[0]
    n_seam = t_hist + 1                                         # history + present frozen at x0

    seam_pos = np.broadcast_to(x0, (n_seam, N, 3))
    seam_vis = np.ones((n_seam, N), bool)

    coords_gt = np.concatenate([seam_pos, gt], axis=0)          # (S,N,3)
    vis_gt = np.concatenate([seam_vis, gt_vis], axis=0)         # (S,N)

    pred_vis = np.isfinite(pred[..., 2]) & (pred[..., 2] > 1e-6)
    if pred_vis_gate:
        pred_vis = pred_vis & (vis_prob >= 0.5)
    coords_pred = np.concatenate([seam_pos, pred], axis=0)
    vis_pred = np.concatenate([seam_vis, pred_vis], axis=0)
    return coords_gt, vis_gt, coords_pred, vis_pred


# --------------------------------------------------------------------------- #
# Rendering
# --------------------------------------------------------------------------- #
def _label(img: np.ndarray, text: str, org: tuple[int, int],
           color: tuple[int, int, int] = (255, 255, 255)) -> None:
    """Draw text with a dark outline so it reads on any background (in place; expects contiguous)."""
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1, cv2.LINE_AA)


def render_sample(res: dict, bg: np.ndarray, t_hist: int, args: argparse.Namespace) -> np.ndarray:
    """One sample -> (S, H+header, 2W, 3) RGB video: [GT | Pred], labeled, with a per-step header."""
    s = int(res["meta"]["stride_hz"])
    H_pred = res["pred"].shape[0]
    S = t_hist + 1 + H_pred

    coords_gt, vis_gt, coords_pred, vis_pred = build_timeline(res, t_hist, s, args.pred_vis_gate)
    uv_gt = reproject(coords_gt, vis_gt, res["K"])              # (S,N,2)
    uv_pred = reproject(coords_pred, vis_pred, res["K"])
    vis_gt &= np.isfinite(uv_gt).all(-1)                       # don't draw off-image / NaN
    vis_pred &= np.isfinite(uv_pred).all(-1)

    # ONE shared per-point colour (keyed on x0's pixel position), used by both panels.
    colors = rainbow_colors_by_position(uv_gt, vis_gt)          # (N,3) uint8
    trace = args.trace if args.trace is not None else S

    panel_gt = render_2d_overlay(bg.copy(), uv_gt, vis_gt, colors, trace=trace)
    panel_pred = render_2d_overlay(bg.copy(), uv_pred, vis_pred, colors, trace=trace)

    m = res["meta"]
    clip = os.path.basename(os.path.dirname(m["episode"]))
    ep = os.path.basename(m["episode"])
    Himg, W = bg.shape[1], bg.shape[2]
    hdr_h = 28
    frames = []
    for k in range(S):
        gt_f = np.ascontiguousarray(panel_gt[k])
        pr_f = np.ascontiguousarray(panel_pred[k])
        _label(gt_f, "GT", (8, 22))
        _label(pr_f, "Pred", (8, 22))
        row = np.concatenate([gt_f, pr_f], axis=1)             # (H, 2W, 3)
        phase = ("HISTORY" if k < t_hist else "PRESENT" if k == t_hist
                 else f"FUTURE {k - t_hist}/{H_pred}")
        header = np.zeros((hdr_h, 2 * W, 3), np.uint8)
        _label(header, f"{clip}/{ep}  t={m['t']}  step {k + 1}/{S}  [{phase}]", (8, 20))
        frames.append(np.concatenate([header, row], axis=0))
    return np.stack(frames)


def viz_one_sample(model: IntentModel, ds: FlowWindowDataset, i: int,
                   args: argparse.Namespace) -> str:
    item = ds[i]
    res = run_inference(model, item, args.device)
    m = res["meta"]
    t_hist = ds.t_hist                                          # authoritative (frame_meta omits it)
    native = frame_grid(int(m["t"]), t_hist, int(m["stride_hz"]), res["pred"].shape[0])
    bg = load_window_frames(os.path.join(m["episode"], "color.mp4"), native)
    video = render_sample(res, bg, t_hist, args)

    clip = os.path.basename(os.path.dirname(m["episode"]))
    ep = os.path.basename(m["episode"])
    out_path = os.path.join(args.out_dir, f"{clip}_{ep}_t{int(m['t'])}.mp4")
    media.write_video(out_path, video, fps=args.fps)
    print(f"  [{clip}/{ep} t={m['t']}]  ADE={sample_ade_mm(res):.2f}mm  ->  {out_path}")
    return out_path


def main() -> None:
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    model = load_ema_model(args.ckpt, args.device)
    ds = build_dataset(ck, args)
    idxs = select_indices(ds, args)
    print(f"{len(ds)} windows in {'episode ' + args.episode if args.episode else args.split + ' split'}; "
          f"rendering {len(idxs)} sample(s) at {args.fps} fps -> {args.out_dir}")
    for i in idxs:
        viz_one_sample(model, ds, i, args)


if __name__ == "__main__":
    main()
