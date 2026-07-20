#!/usr/bin/env python3
"""Live viser QC viewer for FlowWindowDataset items, in Genesis frame P.

Final sanity check on the data pipeline: sample ONE window from the Dataset and
render exactly what that item contains -- nothing re-loaded from disk -- so what you
see is what the model would train on. Everything is placed into frame P with the same
transforms as preprocess/viz_hand_cloud_live.py, so a correct item shows the hand,
the object cloud, and the flow all agreeing in one coordinate frame.

What is drawn (all reconstructed from the item dict alone):
  - STATIC object cloud   : item['cloud'][:-N] (the present-frame P_t), camera->P.
  - QUERY points + FLOW   : item['x0'] (seeds at t) and item['target'] (their future
                            positions), camera->P; a moving dot + growing trail per
                            point, one rainbow colour each.
  - HAND skeleton         : rebuilt from item['q_hist'] / item['q_future'] -- the 24
                            keypoints (node0 = wrist at origin) and M_rel(tau) -- via
                            placed_hand_P (G @ M_rel @ p_raw) + T_HAND_TO_P.

The dataset anchors M_rel at the present frame t (identity there) for BOTH history and
future, so the hand's wrist orientation is expressed relative to now -- consistent with
the camera-frame cloud/flow. This viewer therefore needs the item configured with
articulation='raw_node_pose' and wrist_repr='matrix' (forced below) so the skeleton and
M_rel can be recovered exactly.

Timeline. The item has hand over the history steps (strictly before t) and the future
steps (strictly after t); the flow is defined from t onward. Playback runs
T_hist + L_pred frames: during the HISTORY phase the hand winds up while the query
points sit frozen at x0 (dim); during the FUTURE phase the hand continues and the query
points move along the flow (bright, with trail). The present t is the seam between the
two phases -- where the static cloud lives.

Runs in densetrack3d (needs viser + torch). Usage:
  python data/viz_flow_window_item.py [--data-root DIR] [--clip DIR] [--port 8080]
Then open the printed URL; use "Sample new item" to draw a fresh window.
"""
import argparse
import os
import sys
import time

import numpy as np
import viser

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.flow_window_dataset import FlowWindowDataset
from preprocess.hand_frame_transforms import (
    T_HAND_TO_P, transform_cloud_to_P, placed_hand_P,
)

# finger chains + per-chain colours, matching viz_hand_cloud_live.py
CHAINS = [[0, 1, 2, 3, 4], [0, 5, 6, 7, 8, 9], [0, 10, 11, 12, 13, 14],
          [0, 15, 16, 17, 18, 19], [0, 20, 21, 22, 23, 24]]
CHAIN_RGB = np.array([[230, 25, 75], [60, 180, 75], [67, 99, 216],
                      [245, 130, 49], [145, 30, 180]], dtype=np.uint8)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = p.add_mutually_exclusive_group()
    src.add_argument("--data-root", default="/home/labeng/yanghan/data/inhand_manipulation",
                     help="dataset root (all clips' episode_* with a clouds.npz)")
    src.add_argument("--clip", help="a single clip dir to sample from")
    src.add_argument("--episode", help="a single episode dir to sample from")
    p.add_argument("--port", type=int, default=8080)
    p.add_argument("--stride-hz", type=int, default=4)
    p.add_argument("--t-pred", type=int, default=8)
    p.add_argument("--t-hist", type=int, default=4)
    p.add_argument("--pred-pad", type=int, default=0)
    p.add_argument("--n-query", type=int, default=16)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def rainbow(n):
    """n distinct RGB uint8 colours around the HSV wheel (for the query points)."""
    import matplotlib.pyplot as plt
    return (plt.cm.hsv(np.linspace(0, 1, n, endpoint=False))[:, :3] * 255).astype(np.uint8)


def hand_from_features(q, n_kp=24):
    """(F, d_q) raw_node_pose+matrix features -> (F, 25, 3) hand skeleton in P.

    Layout (see FlowWindowDataset._hand_features): first n_kp*3 = keypoints 1..24
    (node0 wrist dropped, pinned at origin); last 9 = M_rel(tau) row-major 3x3.
    Rebuild the 25-node skeleton and place it: G @ M_rel @ p_raw + T_HAND_TO_P."""
    F = q.shape[0]
    node = np.zeros((F, 25, 3), dtype=np.float64)
    node[:, 1:] = q[:, :n_kp * 3].reshape(F, n_kp, 3)
    M_rel = q[:, n_kp * 3:].reshape(F, 3, 3)
    return placed_hand_P(node, M_rel) + T_HAND_TO_P          # (F, 25, 3) in P


def finger_segments(hand_P, chain):
    """(S,2,3) start/end points for one finger chain's bones."""
    return np.array([[hand_P[a], hand_P[b]] for a, b in zip(chain[:-1], chain[1:])],
                    dtype=np.float32)


def build_item_frames(item):
    """Reconstruct everything the viewer draws, in frame P, from one item dict."""
    N = item["x0"].shape[0]
    cloud_all_P = transform_cloud_to_P(item["cloud"].astype(np.float64))   # (P+N, 3)
    obj_cloud_P = cloud_all_P[:-N]                                         # static object cloud
    x0_P = cloud_all_P[-N:]                                                # query seeds (== last N)
    flow_P = transform_cloud_to_P(item["target"].astype(np.float64))      # (L_pred, N, 3)
    vis = item["target_vis"]                                              # (L_pred, N) bool
    hand_hist = hand_from_features(item["q_hist"].astype(np.float64))     # (T_hist, 25, 3)
    hand_fut = hand_from_features(item["q_future"].astype(np.float64))    # (L_pred, 25, 3)
    return dict(obj_cloud_P=obj_cloud_P, x0_P=x0_P, flow_P=flow_P, vis=vis,
                hand_hist=hand_hist, hand_fut=hand_fut, N=N,
                T_hist=hand_hist.shape[0], L_pred=flow_P.shape[0],
                meta=item["frame_meta"])


def main():
    args = parse_args()
    source = args.episode or args.clip or args.data_root
    ds = FlowWindowDataset(source, stride_hz=args.stride_hz, t_pred=args.t_pred,
                           t_hist=args.t_hist, pred_pad=args.pred_pad,
                           n_query=args.n_query, articulation="raw_node_pose",
                           use_wrist=True, wrist_repr="matrix", seed=args.seed)
    print(f"Dataset: {len(ds.episodes)} episodes, {len(ds)} windows")
    print(ds.coverage_summary())
    qcolors = rainbow(args.n_query)

    server = viser.ViserServer(port=args.port)
    server.scene.set_up_direction("+z")
    server.scene.add_frame("/P", axes_length=0.1, axes_radius=0.003)
    server.scene.add_grid("/grid", width=1.0, height=1.0, cell_size=0.05)

    with server.gui.add_folder("Playback"):
        g_play = server.gui.add_checkbox("Playing", True)
        g_fps = server.gui.add_slider("FPS", 1, 30, 1, 8)
        g_psize = server.gui.add_slider("Point size", 0.001, 0.02, 0.001, 0.001)
        g_qsize = server.gui.add_slider("Query size", 0.001, 0.03, 0.001, 0.001)
    g_sample = server.gui.add_button("Sample new item")
    g_info = server.gui.add_text("Item", initial_value="", disabled=True)
    g_phase = server.gui.add_text("Phase", initial_value="", disabled=True)

    # mutable playback state (list-boxed so the button callback can rebind it)
    state = {"F": build_item_frames(ds[np.random.default_rng().integers(len(ds))]), "k": 0}

    def describe(F):
        m = F["meta"]
        ep = os.path.basename(os.path.dirname(m["episode"])) + "/" + os.path.basename(m["episode"])
        return f"{ep}  t={m['t']}  N={F['N']}  T_hist={F['T_hist']}  L_pred={F['L_pred']}"

    g_info.value = describe(state["F"])

    @g_sample.on_click
    def _(_):
        i = int(np.random.default_rng().integers(len(ds)))
        state["F"] = build_item_frames(ds[i])
        state["k"] = 0
        g_info.value = describe(state["F"])
        print(f"[sample #{i}] {g_info.value}")

    def render(F, k):
        T_hist, L_pred, N = F["T_hist"], F["L_pred"], F["N"]

        # static object cloud (present observation P_t), grey
        server.scene.add_point_cloud("/obj", points=F["obj_cloud_P"].astype(np.float32),
                                     colors=np.full((len(F["obj_cloud_P"]), 3), 170, np.uint8),
                                     point_size=g_psize.value)

        if k < T_hist:                                   # HISTORY: hand winds up, flow frozen at x0
            hand = F["hand_hist"][k]
            q_pos = F["x0_P"]
            q_vis = np.ones(N, bool)
            trail_len = 0
            g_phase.value = f"HISTORY  step {k + 1}/{T_hist}"
        else:                                            # FUTURE: query points follow the flow
            j = k - T_hist
            hand = F["hand_fut"][j]
            q_pos = F["flow_P"][j]
            q_vis = F["vis"][j]
            trail_len = j
            g_phase.value = f"FUTURE  step {j + 1}/{L_pred}"

        # query points: bright rainbow dots, dimmed during history / when occluded
        dim = 0.35 if k < T_hist else 1.0
        vis_pts = q_pos[q_vis]
        vis_col = (qcolors[q_vis].astype(np.float32) * dim).astype(np.uint8)
        server.scene.add_point_cloud("/query", points=vis_pts.astype(np.float32),
                                     colors=vis_col, point_size=g_qsize.value)

        # growing flow trail per query point: the polyline x0 -> flow[0] -> ... -> flow[j-1].
        # Only segments between CONSECUTIVE visible steps are drawn (occluded steps carry
        # NaN coords, so a broken trail correctly shows where the point disappeared). A
        # stale trail from the previous frame/item is cleared with a degenerate segment.
        for n in range(N):
            # sequence of (point, visible) from t (x0, always visible) through flow[0..j-1]
            seq_pts = [F["x0_P"][n]] + [F["flow_P"][s, n] for s in range(trail_len)]
            seq_vis = [True] + [bool(F["vis"][s, n]) for s in range(trail_len)]
            segs = [[seq_pts[s], seq_pts[s + 1]] for s in range(len(seq_pts) - 1)
                    if seq_vis[s] and seq_vis[s + 1]]
            seg = np.asarray(segs, np.float32) if segs else np.zeros((1, 2, 3), np.float32)
            server.scene.add_line_segments(f"/trail/{n}", points=seg,
                                           colors=tuple(int(v) for v in qcolors[n]),
                                           line_width=2.0)

        # hand skeleton
        for ci, (chain, rgb) in enumerate(zip(CHAINS, CHAIN_RGB)):
            server.scene.add_line_segments(f"/hand/f{ci}", points=finger_segments(hand, chain),
                                           colors=tuple(int(v) for v in rgb), line_width=3.0)
        server.scene.add_point_cloud("/wrist", points=hand[:1].astype(np.float32),
                                     colors=np.array([[0, 0, 0]], np.uint8),
                                     point_size=g_qsize.value)

    print(f"viser up on port {args.port} -- open the printed URL.")
    last = time.time()
    while True:
        F = state["F"]
        total = F["T_hist"] + F["L_pred"]
        if g_play.value and time.time() - last >= 1.0 / g_fps.value:
            state["k"] = (state["k"] + 1) % total
            last = time.time()
        render(F, state["k"])
        time.sleep(0.01)


if __name__ == "__main__":
    main()
