#!/usr/bin/env python3
"""Live viser viewer: hand skeleton + object point cloud in Genesis frame P.

Free orbit / zoom / pan in the browser, timestep playback, plus live sliders for an
extra hand rotation/translation on top of the fixed placement -- so a residual mount
offset can be dialed in by eye and read straight off the GUI.

Frame P (Genesis): +x front, +y left, +z up.

  CLOUD: p_P = T_CAM_P @ [p_cam;1]                              (from object_flow.pkl)

  HAND : p_P = R_extra @ ( G @ M_rel(t) @ p_raw(t) ) + T_HAND_TO_P + t_extra
         p_raw(t)  = raw hand-LOCAL skeleton at frame t (wrist pinned, fingers only)
         M_rel(t)  = R(q_a)^T R(q_t)   wrist rotation RELATIVE to anchor frame a
                     (cancels the arbitrary per-session Manus world frame; identity at a)
         G         = fixed reflection placing the raw hand-local frame into P
         R_extra / t_extra = live GUI fine-tune sliders (start at identity / 0)

All fixed transforms (G, T_HAND_TO_P, the cam->P chain) and the placement / wrist-rotation
helpers live in preprocess/hand_frame_transforms.py -- the single source of truth shared
with any downstream world-model loader. See that module for the full derivation.

Self-contained on a run_pipeline.sh output folder: reads object_flow.pkl (cloud) and
hand.pkl (raw_node_pose + wrist_quat) -- no mcap needed. wrist_quat is already row-aligned
to the camera frames by preprocess/extract_hand.py.

Runs in track4world (needs viser). Usage:
  python preprocess/viz_hand_cloud_live.py --folder results/test_cube \
      [--port 8080] [--max-points 1500] [--anchor-frame 0]
Then open the printed URL in a browser.
"""
import argparse
import os
import pickle
import time

import numpy as np
import viser

from hand_frame_transforms import (
    T_HAND_TO_P, transform_cloud_to_P,
    wrist_M_rel, stabilized_cloud_P, stabilized_hand_P, placed_hand_P,
)

CHAINS = [[0, 1, 2, 3, 4], [0, 5, 6, 7, 8, 9], [0, 10, 11, 12, 13, 14],
          [0, 15, 16, 17, 18, 19], [0, 20, 21, 22, 23, 24]]
CHAIN_RGB = np.array([[230, 25, 75], [60, 180, 75], [67, 99, 216],
                      [245, 130, 49], [145, 30, 180]], dtype=np.uint8)

# skeleton node indices for the palm-basis sanity print
WRIST, THUMB_MCP, INDEX_MCP, MIDDLE_MCP, PINKY_MCP = 0, 1, 5, 10, 20


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--folder", required=True, help="run_pipeline.sh output (object_flow.pkl + hand.pkl)")
    p.add_argument("--port", type=int, default=8080)
    p.add_argument("--max-points", type=int, default=1500)
    p.add_argument("--anchor-frame", type=int, default=0,
                   help="frame taken as the wrist-rotation reference; M_rel is identity here "
                        "so the anchor sits at its unrotated hand-local pose, which G maps to "
                        "the canonical palm->-x / fingers->+z orientation")
    return p.parse_args()


def euler_R(rx, ry, rz):
    """Intrinsic XYZ Euler (degrees) -> 3x3. Applied as extra hand rotation."""
    rx, ry, rz = np.radians([rx, ry, rz])
    cx, sx = np.cos(rx), np.sin(rx)
    cy, sy = np.cos(ry), np.sin(ry)
    cz, sz = np.cos(rz), np.sin(rz)
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    return Rx @ Ry @ Rz


def palm_axes(hand_pts):
    """Report-only: dominant P-axis of distal (fingers) and toPinky (knuckle spread).

    Both are plain transformed vectors, so they are reflection-SAFE under G. We do NOT
    report the palm normal via cross(toPinky, distal): a cross product is chirality-
    sensitive and G is a reflection (det -1), so cross(Ga,Gb) = -G cross(a,b) -- the
    reported normal would flip sign for the mirrored hand even though the true palm
    normal maps to -x for BOTH hands by construction (G @ native_palm_normal(-y) = -x)."""
    distal = hand_pts[MIDDLE_MCP] - hand_pts[WRIST]
    to_pinky = hand_pts[PINKY_MCP] - hand_pts[INDEX_MCP]
    def _ax(v):
        i = int(np.argmax(np.abs(v))); return ("+" if v[i] > 0 else "-") + "xyz"[i]
    return _ax(distal), _ax(to_pinky)


def finger_segments(hand_P, chain):
    """(S,2,3) start/end points for one finger chain's bones."""
    return np.array([[hand_P[a], hand_P[b]] for a, b in zip(chain[:-1], chain[1:])],
                    dtype=np.float32)


def main():
    args = parse_args()
    of = pickle.load(open(os.path.join(args.folder, "object_flow.pkl"), "rb"))
    hd = pickle.load(open(os.path.join(args.folder, "hand.pkl"), "rb"))
    coords, vis, colors = of["coords"], of["vis"], of["colors"]
    node_pos = hd["raw_node_pose"][..., :3]
    T = coords.shape[0]
    assert node_pos.shape[0] == T, f"cloud T={T} != hand T={node_pos.shape[0]}"
    if "wrist_quat" not in hd:
        raise KeyError(f"{args.folder}/hand.pkl has no 'wrist_quat' -- this folder predates the "
                       f"wrist-quat extractor. Re-run run_pipeline.sh (or extract_hand.py) to add it.")
    quat_al = hd["wrist_quat"]                            # (T,4), already frame-aligned
    assert quat_al.shape[0] == T, f"cloud T={T} != wrist_quat T={quat_al.shape[0]}"
    coords_P = transform_cloud_to_P(coords)              # (T,N,3)
    colors_u8 = np.clip(colors, 0, 255).astype(np.uint8)

    # Anchor-relative wrist rotation M_rel(t) = R(q_a)^T R(q_t) (identity at the anchor;
    # cancels the arbitrary per-session Manus world frame). See hand_frame_transforms.
    a = int(np.clip(args.anchor_frame, 0, T - 1))
    M_rel = wrist_M_rel(quat_al, anchor=a)                          # (T,3,3)

    # Placed hand: G @ M_rel @ p_raw (order matters -- M_rel is hand-local). Fine-tuned by sliders.
    hand_pre = placed_hand_P(node_pos, M_rel)                      # (T,25,3) in P
    # Wrist-stabilized variant: hand freezes at its anchor pose (G @ p_raw, no M_rel); the cloud
    # cancels the wrist rotation expressed in P via S_P = G @ M_rel^T @ G, keeping P orientation.
    hand_stab = stabilized_hand_P(node_pos)                        # (T,25,3): G @ p_raw
    coords_stab = stabilized_cloud_P(coords_P, M_rel)              # (T,N,3) stabilized cloud

    d0, l0 = palm_axes(hand_pre[a])
    print(f"anchor frame {a}: in P  fingers(distal) {d0}  toPinky {l0}  "
          f"(target fingers +z; palm-normal -> -x is guaranteed by G for both hands; "
          f"thumb side: +y for left / -y for right)")

    server = viser.ViserServer(port=args.port)
    server.scene.set_up_direction("+z")
    server.scene.add_frame("/P", axes_length=0.1, axes_radius=0.003)  # origin triad
    server.scene.add_grid("/grid", width=1.0, height=1.0, cell_size=0.05)

    with server.gui.add_folder("Playback"):
        g_play = server.gui.add_checkbox("Playing", True)
        g_t = server.gui.add_slider("Frame", 0, T - 1, 1, 0)
        g_fps = server.gui.add_slider("FPS", 1, 60, 1, 15)
        g_psize = server.gui.add_slider("Point size", 0.001, 0.02, 0.001, 0.004)
        g_stab = server.gui.add_checkbox("Stabilize (cancel wrist rot)", False)
    with server.gui.add_folder("Hand extra rotation (deg)"):
        g_rx = server.gui.add_slider("rx", -180, 180, 1, 0)
        g_ry = server.gui.add_slider("ry", -180, 180, 1, 0)
        g_rz = server.gui.add_slider("rz", -180, 180, 1, 0)
    with server.gui.add_folder("Hand extra translation (m)"):
        g_tx = server.gui.add_slider("tx", -0.3, 0.3, 0.005, 0.0)
        g_ty = server.gui.add_slider("ty", -0.3, 0.3, 0.005, 0.0)
        g_tz = server.gui.add_slider("tz", -0.3, 0.3, 0.005, 0.0)
    g_print = server.gui.add_button("Print current R_extra + translation")

    @g_print.on_click
    def _(_):
        Re = euler_R(g_rx.value, g_ry.value, g_rz.value)
        print("\n--- fine-tune on top of the fixed G-placed hand ---")
        print(f"R_extra (euler XYZ deg = {g_rx.value},{g_ry.value},{g_rz.value}):\n"
              f"{np.array2string(Re, precision=4)}")
        t_total = T_HAND_TO_P + np.array([g_tx.value, g_ty.value, g_tz.value])
        print(f"total translation (T_HAND_TO_P + sliders) = "
              f"[{t_total[0]:.3f}, {t_total[1]:.3f}, {t_total[2]:.3f}]\n")

    def hand_P(t):
        # stabilized: G @ p_raw (wrist frozen at anchor, fingers only); else G @ M_rel @ p_raw.
        pts = hand_stab[t] if g_stab.value else hand_pre[t]
        Re = euler_R(g_rx.value, g_ry.value, g_rz.value)
        pts = (Re @ pts.T).T + T_HAND_TO_P + np.array([g_tx.value, g_ty.value, g_tz.value])
        return pts

    def render(t):
        cloud_t = coords_stab[t] if g_stab.value else coords_P[t]
        m = vis[t] & np.isfinite(cloud_t).all(-1)
        pts = cloud_t[m]; rgb = colors_u8[m]
        if len(pts) > args.max_points:
            sel = np.linspace(0, len(pts) - 1, args.max_points).astype(int)
            pts, rgb = pts[sel], rgb[sel]
        server.scene.add_point_cloud("/cloud", points=pts.astype(np.float32),
                                     colors=rgb, point_size=g_psize.value)
        hp = hand_P(t)
        for ci, (chain, rgb) in enumerate(zip(CHAINS, CHAIN_RGB)):
            server.scene.add_line_segments(f"/hand/f{ci}", points=finger_segments(hp, chain),
                                           colors=tuple(int(v) for v in rgb), line_width=3.0)
        server.scene.add_point_cloud("/wrist", points=hp[:1].astype(np.float32),
                                     colors=np.array([[0, 0, 0]], np.uint8),
                                     point_size=g_psize.value * 2.5)

    print(f"viser up on port {args.port} -- open the printed URL. T={T} frames.")
    last = time.time()
    while True:
        if g_play.value:
            if time.time() - last >= 1.0 / g_fps.value:
                g_t.value = (g_t.value + 1) % T
                last = time.time()
        render(g_t.value)
        time.sleep(0.01)


if __name__ == "__main__":
    main()
