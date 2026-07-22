#!/usr/bin/env python3
"""Shared coordinate transforms: Manus hand + object cloud -> Genesis frame P.

Single source of truth for the fixed transforms that place the object point cloud
(from object_flow.pkl, camera-optical frame) and the Manus hand skeleton
(raw_node_pose + wrist quaternion) into Genesis **frame P** (+x front, +y left, +z up),
and for the wrist-rotation cancellation that removes wrist drift.

Both the live viewer (preprocess/viz_hand_cloud_live.py) and any downstream world-model
label loader import from here, so the math cannot diverge between them.

Frames:
  camera = OpenCV optical (+X right, +Y down, +Z into scene) -- object_flow.pkl coords.
  P      = Genesis canonical (+x front, +y left, +z up).

Cloud placement (camera -> P), a fixed rigid mount chain:
  p_P = T_PP_P @ T_LENS_PP @ T_CAM_LENS @ [p_cam; 1]   (== T_CAM_P @ [p_cam; 1])

Hand placement (raw hand-local -> P):
  p_P(t) = G @ M_rel(t) @ p_raw(t)  (+ T_HAND_TO_P mount offset, applied by the caller)
  p_raw(t) = raw 25-node skeleton at frame t (wrist pinned at origin, fingers only).
  M_rel(t) = R(q_a)^T R(q_t)   wrist rotation RELATIVE to anchor frame a (default 0).
             raw_sensor_orientation is absolute in an arbitrary per-session Manus world
             frame; conjugating by the anchor cancels it. Identity at t=a.

Two hard-won facts (see memory hand-to-framep-transform.md):
  1. ORDER: it must be G @ M_rel @ p_raw, NOT "canonicalize the anchor into P, then apply
     raw M_rel in P". M_rel lives in the hand-LOCAL frame, so assemble the full motion in
     hand-local first (M_rel inside), then apply the single G to the whole result.
  2. G is a REFLECTION (det -1), the SAME matrix for both hands. Manus stores the skeleton
     mirror-wrong (left as mirrored-right and vice versa). Native raw axes (both hands):
     distal->+z, palm-normal->-y, thumb->+x(left)/-x(right). Constraints distal->+z and
     palm-normal->-x pin two axes; the free thumb axis is chosen so det<0. A proper
     rotation (det +1) would keep the WRONG chirality. Because the raw left/right hands are
     exact mirror images, one reflection un-mirrors BOTH -> no per-side branch.
       native +z(distal)->+z, native -y(palm-normal)->-x, native +x(thumb)->+y.

Wrist-rotation cancellation (the viewer "Stabilize" checkbox), expressed in P:
  cloud (P-oriented, drift removed) = S_P(t) @ p_P(t),  S_P(t) = G @ M_rel(t)^T @ G
  hand  (P-oriented, wrist frozen)  = G @ p_raw(t)      (drops M_rel entirely)
G is its own inverse (G @ G = I), so S_P re-expresses the hand-local cancelling rotation
M_rel^T in P coords. This cancels ROTATION only -- wrist translation is not in the data.
"""
import numpy as np

# ============================ SHARED FIXED TRANSFORMS ========================
# Raw hand-local -> P placement: a REFLECTION (det -1), same for both hands (see docstring).
#   native +z(distal)->+z, native -y(palm-normal)->-x, native +x(thumb)->+y.
G = np.array([[0, 1, 0],
              [1, 0, 0],
              [0, 0, 1]], dtype=np.float64)

# Wrist-origin mount offset in P (added to the placed hand by the caller).
T_HAND_TO_P = np.array([0.03, 0.0, 0.0], dtype=np.float64)

# Camera-optical -> P rigid mount chain (fixed by the physical ZED mount).
T_CAM_LENS = np.array([[1, 0, 0, 0],
                       [0, 1, 0, 0],
                       [0, 0, 1, -0.024],
                       [0, 0, 0, 1]], dtype=np.float64)
T_LENS_PP = np.array([
    [0.0,  2.13030386e-01,  9.77045574e-01, -2.27305568e-01],
    [0.0, -9.77045574e-01,  2.13030386e-01,  3.05639044e-02],
    [1.0,  0.0,             0.0,             2.46926827e-04],
    [0.0,  0.0,             0.0,             1.0]], dtype=np.float64)
T_PP_P = np.array([[1, 0, 0, 0],
                   [0, 0, -1, 0],
                   [0, 1, 0, 0],
                   [0, 0, 0, 1]], dtype=np.float64)

# Composed camera-optical -> P (4x4 homogeneous), and its inverse P -> camera-optical.
T_CAM_P = T_PP_P @ T_LENS_PP @ T_CAM_LENS
T_P_CAM = np.linalg.inv(T_CAM_P)
# =============================================================================


def quat_to_R(q):
    """Quaternion (x, y, z, w) -> 3x3 rotation. Returns identity for a ~zero quat."""
    x, y, z, w = q
    n = x * x + y * y + z * z + w * w
    if n < 1e-12:
        return np.eye(3)
    s = 2.0 / n
    return np.array([
        [1 - (y*y+z*z)*s, (x*y-w*z)*s,     (x*z+w*y)*s],
        [(x*y+w*z)*s,     1 - (x*x+z*z)*s, (y*z-w*x)*s],
        [(x*z-w*y)*s,     (y*z+w*x)*s,     1 - (x*x+y*y)*s]], dtype=np.float64)


def transform_cloud_to_P(coords_cam):
    """Object cloud camera-optical -> P: p_P = T_CAM_P @ [p_cam; 1].

    coords_cam: (..., 3) in camera-optical frame. Returns same shape, in P.
    NaN/inf entries pass through as-is (they map to NaN/inf)."""
    shp = coords_cam.shape
    flat = coords_cam.reshape(-1, 3)
    homo = np.concatenate([flat, np.ones((flat.shape[0], 1))], axis=1)
    return (T_CAM_P @ homo.T).T[:, :3].reshape(shp)


def wrist_M_rel(quat_al, anchor=0):
    """Anchor-relative wrist rotation M_rel(t) = R(q_a)^T R(q_t) for every frame.

    quat_al: (T, 4) per-frame wrist quaternions (x, y, z, w), already frame-aligned.
    anchor:  reference frame index a (M_rel is identity there). Clamped to [0, T-1].
    Returns (T, 3, 3). The arbitrary per-session Manus world frame cancels in the conjugation."""
    quat_al = np.asarray(quat_al, dtype=np.float64)
    T = quat_al.shape[0]
    a = int(np.clip(anchor, 0, T - 1))
    Ra_inv = quat_to_R(quat_al[a]).T
    return np.stack([Ra_inv @ quat_to_R(quat_al[t]) for t in range(T)])  # (T,3,3)


def stabilized_cloud_P(coords_P, M_rel):
    """Cancel wrist rotation on a P-frame cloud, keeping P orientation.

    cloud_stab(t) = S_P(t) @ p_P(t),  S_P(t) = G @ M_rel(t)^T @ G   (G @ G = I).
    coords_P: (T, N, 3) in P (e.g. from transform_cloud_to_P). M_rel: (T, 3, 3).
    Returns (T, N, 3), drift-free, still in P."""
    S_P = np.einsum("ij,tkj,kl->til", G, M_rel, G)          # (T,3,3): G @ M_rel^T @ G
    return np.einsum("tij,tnj->tni", S_P, coords_P)          # (T,N,3)


def stabilized_hand_P(node_pos):
    """Wrist-frozen hand in P: G @ p_raw (drops M_rel; the wrist is fixed in its own frame).

    node_pos: (T, 25, 3) raw hand-local skeleton positions. Returns (T, 25, 3) in P."""
    return np.einsum("ij,tnj->tni", G, node_pos)


def placed_hand_P(node_pos, M_rel):
    """Full moving hand in P: G @ M_rel(t) @ p_raw(t)  (order matters -- M_rel is hand-local).

    node_pos: (T, 25, 3) raw hand-local skeleton. M_rel: (T, 3, 3). Returns (T, 25, 3) in P.
    Add T_HAND_TO_P (and any fine-tune) afterwards at the call site."""
    return np.einsum("ij,tjk,tnk->tni", G, M_rel, node_pos)


def placed_hand_camera(node_pos, M_rel):
    """Full moving hand in CAMERA-OPTICAL frame: (P placement) then P -> camera.

    Places the raw hand-local skeleton into P exactly as placed_hand_P, then maps back to the camera-optical
    frame so the keypoints share ONE frame with the object cloud / query points

    node_pos: (T, 25, 3) raw hand-local skeleton. M_rel: (T, 3, 3) anchored wrist rotation.
    Returns (T, 25, 3) in camera-optical metres (same frame as object_flow.pkl coords)."""
    hand_P = placed_hand_P(node_pos, M_rel) + T_HAND_TO_P    # (T,25,3) in P
    shp = hand_P.shape
    flat = hand_P.reshape(-1, 3)
    homo = np.concatenate([flat, np.ones((flat.shape[0], 1))], axis=1)
    return (T_P_CAM @ homo.T).T[:, :3].reshape(shp)          # (T,25,3) camera


def wrist_frame_flow(coords_cam, wrist_quat, anchor=0):
    """One-shot camera-frame object flow -> P-frame, wrist-rotation cancelled.

    coords_cam: (T, N, 3) object flow in camera-optical frame (object_flow.pkl coords).
    wrist_quat: (T, 4) frame-aligned wrist quaternions (x, y, z, w), e.g. hand.pkl["wrist_quat"].
    anchor:     reference frame (default 0). Returns (T, N, 3), drift-free, in P."""
    coords_P = transform_cloud_to_P(coords_cam)
    M_rel = wrist_M_rel(wrist_quat, anchor=anchor)
    return stabilized_cloud_P(coords_P, M_rel)
