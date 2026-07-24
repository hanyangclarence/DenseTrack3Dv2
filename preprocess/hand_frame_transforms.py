# VENDORED COPY. The CANONICAL version lives in the intent-model repo at
# data/hand_frame_transforms.py. This copy exists only for preprocess/viz_hand_cloud_live.py.
# If you change the transforms, update the canonical copy in intent-model too (must match:
# the intent model trains on placed_hand_camera from the canonical copy).
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


def quat_to_R(q: np.ndarray) -> np.ndarray:
    """Quaternion (x, y, z, w) -> 3x3 rotation. Returns identity for a ~zero quat.

    q: (4,) array-like. Returns (3, 3) float64."""
    x, y, z, w = q
    n = x * x + y * y + z * z + w * w
    if n < 1e-12:
        return np.eye(3)
    s = 2.0 / n
    return np.array([
        [1 - (y*y+z*z)*s, (x*y-w*z)*s,     (x*z+w*y)*s],
        [(x*y+w*z)*s,     1 - (x*x+z*z)*s, (y*z-w*x)*s],
        [(x*z-w*y)*s,     (y*z+w*x)*s,     1 - (x*x+y*y)*s]], dtype=np.float64)


def transform_cloud_to_P(coords_cam: np.ndarray) -> np.ndarray:
    """Object cloud camera-optical -> P: p_P = T_CAM_P @ [p_cam; 1].

    coords_cam: (..., 3) in camera-optical frame. Returns same shape, in P.
    NaN/inf entries pass through as-is (they map to NaN/inf)."""
    shp = coords_cam.shape
    flat = coords_cam.reshape(-1, 3)
    homo = np.concatenate([flat, np.ones((flat.shape[0], 1))], axis=1)
    return (T_CAM_P @ homo.T).T[:, :3].reshape(shp)


def wrist_M_rel(quat_al: np.ndarray, anchor: int = 0) -> np.ndarray:
    """Anchor-relative wrist rotation M_rel(t) = R(q_a)^T R(q_t) for every frame.

    quat_al: (T, 4) per-frame wrist quaternions (x, y, z, w), already frame-aligned.
    anchor:  reference frame index a (M_rel is identity there). Clamped to [0, T-1].
    Returns (T, 3, 3). The arbitrary per-session Manus world frame cancels in the conjugation."""
    quat_al = np.asarray(quat_al, dtype=np.float64)
    T = quat_al.shape[0]
    a = int(np.clip(anchor, 0, T - 1))
    Ra_inv = quat_to_R(quat_al[a]).T
    return np.stack([Ra_inv @ quat_to_R(quat_al[t]) for t in range(T)])  # (T,3,3)


def stabilized_cloud_P(coords_P: np.ndarray, M_rel: np.ndarray) -> np.ndarray:
    """Cancel wrist rotation on a P-frame cloud, keeping P orientation.

    cloud_stab(t) = S_P(t) @ p_P(t),  S_P(t) = G @ M_rel(t)^T @ G   (G @ G = I).
    coords_P: (T, N, 3) in P (e.g. from transform_cloud_to_P). M_rel: (T, 3, 3).
    Returns (T, N, 3), drift-free, still in P."""
    S_P = np.einsum("ij,tkj,kl->til", G, M_rel, G)          # (T,3,3): G @ M_rel^T @ G
    return np.einsum("tij,tnj->tni", S_P, coords_P)          # (T,N,3)


def stabilized_hand_P(node_pos: np.ndarray) -> np.ndarray:
    """Wrist-frozen hand in P: G @ p_raw (drops M_rel; the wrist is fixed in its own frame).

    node_pos: (T, 25, 3) raw hand-local skeleton positions. Returns (T, 25, 3) in P."""
    return np.einsum("ij,tnj->tni", G, node_pos)


def placed_hand_P(node_pos: np.ndarray, M_rel: np.ndarray) -> np.ndarray:
    """Full moving hand in P: G @ M_rel(t) @ p_raw(t)  (order matters -- M_rel is hand-local).

    node_pos: (T, 25, 3) raw hand-local skeleton. M_rel: (T, 3, 3). Returns (T, 25, 3) in P.
    Add T_HAND_TO_P (and any fine-tune) afterwards at the call site."""
    return np.einsum("ij,tjk,tnk->tni", G, M_rel, node_pos)


def placed_hand_camera(node_pos: np.ndarray, M_rel: np.ndarray,
                       delta_R: np.ndarray = None, delta_t: np.ndarray = None) -> np.ndarray:
    """Full moving hand in CAMERA-OPTICAL frame: (P placement) then P -> camera.

    Places the raw hand-local skeleton into P exactly as placed_hand_P, then maps back to the camera-optical
    frame so the keypoints share ONE frame with the object cloud / query points

    node_pos: (T, 25, 3) raw hand-local skeleton. M_rel: (T, 3, 3) anchored wrist rotation.
    delta_R:  (3, 3) optional rigid rotation applied about the WRIST. None = identity.
    delta_t:  (3,) optional rigid translation (metres, in P) added to the mount offset. None = zero.
    Returns (T, 25, 3) in camera-optical metres (same frame as object_flow.pkl coords)."""
    hand_P = placed_hand_P(node_pos, M_rel)                  # (T,25,3) in P, node0 at origin
    if delta_R is not None:                                  # rigid rotation about the wrist origin
        hand_P = np.einsum("ij,tnj->tni", delta_R, hand_P)
    hand_P = hand_P + T_HAND_TO_P
    if delta_t is not None:                                  # rigid mount translation
        hand_P = hand_P + delta_t
    shp = hand_P.shape
    flat = hand_P.reshape(-1, 3)
    homo = np.concatenate([flat, np.ones((flat.shape[0], 1))], axis=1)
    return (T_P_CAM @ homo.T).T[:, :3].reshape(shp)          # (T,25,3) camera


def wrist_frame_flow(coords_cam: np.ndarray, wrist_quat: np.ndarray, anchor: int = 0) -> np.ndarray:
    """One-shot camera-frame object flow -> P-frame, wrist-rotation cancelled.

    coords_cam: (T, N, 3) object flow in camera-optical frame (object_flow.pkl coords).
    wrist_quat: (T, 4) frame-aligned wrist quaternions (x, y, z, w), e.g. hand.pkl["wrist_quat"].
    anchor:     reference frame (default 0). Returns (T, N, 3), drift-free, in P."""
    coords_P = transform_cloud_to_P(coords_cam)
    M_rel = wrist_M_rel(wrist_quat, anchor=anchor)
    return stabilized_cloud_P(coords_P, M_rel)


# ============================ JOINT-JITTER AUGMENTATION ======================
# Helpers for reshaping the Manus skeleton into a plausible alternate grasp by
# perturbing each 1-DOF revolute joint about its own (data-recovered) axis. The
# skeleton is a rigid-bone tree (bone lengths constant over time) and every moving
# joint spins about a fixed parent-local axis, so a delta angle is a proper,
# bone-preserving re-pose. See docs/superpowers/specs/2026-07-22-hand-joint-jitter-*.

# Policy: joints we deliberately do NOT jitter, even though they move. These are the
# root-closest (MCP knuckle) joints of index/middle/ring/pinky -- nodes 5,10,15,20, the
# first node of each of those finger chains. Perturbing them swings the whole finger from
# the knuckle (finger splay / palm arch), which reshapes the hand's gross posture rather
# than the grasp curl we want to augment; the thumb MCP (node 1) is kept. recover_joint_axes
# drops these from has_dof so BOTH the training dataset and viz_aug_skeletons inherit the
# exclusion from this one place (both gate their per-joint draw on has_dof only).
AUG_EXCLUDE_NODES = (5, 10, 15, 20)

def quats_to_R(q: np.ndarray) -> np.ndarray:
    """Batched quaternion (x, y, z, w) -> rotation. q: (..., 4) -> (..., 3, 3).

    Zero-norm quaternions map to identity (matches quat_to_R)."""
    q = np.asarray(q, dtype=np.float64)
    x, y, z, w = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    n = x * x + y * y + z * z + w * w
    s = np.where(n < 1e-12, 0.0, 2.0 / n)               # s=0 -> identity below
    R = np.empty(q.shape[:-1] + (3, 3), dtype=np.float64)
    R[..., 0, 0] = 1 - (y * y + z * z) * s; R[..., 0, 1] = (x * y - w * z) * s; R[..., 0, 2] = (x * z + w * y) * s
    R[..., 1, 0] = (x * y + w * z) * s;     R[..., 1, 1] = 1 - (x * x + z * z) * s; R[..., 1, 2] = (y * z - w * x) * s
    R[..., 2, 0] = (x * z - w * y) * s;     R[..., 2, 1] = (y * z + w * x) * s;     R[..., 2, 2] = 1 - (x * x + y * y) * s
    return R


def rot_about_axis(axis: np.ndarray, theta: np.ndarray) -> np.ndarray:
    """Rodrigues rotation about a (per-row) axis by theta radians.

    axis: (M, 3), need not be unit; a zero-length axis yields identity for any theta.
    theta: (M,). Returns (M, 3, 3)."""
    axis = np.asarray(axis, dtype=np.float64).reshape(-1, 3)
    theta = np.asarray(theta, dtype=np.float64).reshape(-1)
    norm = np.linalg.norm(axis, axis=1, keepdims=True)
    u = np.where(norm < 1e-12, 0.0, axis / np.where(norm < 1e-12, 1.0, norm))  # (M,3), 0 for zero axis
    M = axis.shape[0]
    K = np.zeros((M, 3, 3))
    K[:, 0, 1], K[:, 0, 2] = -u[:, 2], u[:, 1]
    K[:, 1, 0], K[:, 1, 2] = u[:, 2], -u[:, 0]
    K[:, 2, 0], K[:, 2, 1] = -u[:, 1], u[:, 0]
    s, c = np.sin(theta)[:, None, None], np.cos(theta)[:, None, None]
    return np.eye(3)[None] + s * K + (1 - c) * (K @ K)  # zero K (zero axis) -> identity


def random_wrist_delta(rng, deg: float, trans_m: float) -> tuple[np.ndarray, np.ndarray]:
    """Draw one small rigid (delta_R, delta_t) for a hand->camera placement augmentation.

    rng:     a numpy Generator (np.random.default_rng); all draws come from it so callers
             control determinism / RNG ordering.
    deg:     rotation magnitude bound -- angle ~ U(-deg, +deg) about a uniformly-random unit axis.
    trans_m: translation bound -- each of delta_t's 3 components ~ U(-trans_m, +trans_m) metres.
    Returns (delta_R (3,3), delta_t (3,)) suitable for placed_hand_camera. With deg=0 and
    trans_m=0 this returns (I, 0) exactly (rot_about_axis of 0 is identity)."""
    v = rng.standard_normal(3)                              # uniform-on-sphere axis (direction only)
    n = np.linalg.norm(v)
    axis = v / n if n > 1e-12 else np.array([1.0, 0.0, 0.0])
    angle = np.deg2rad(rng.uniform(-deg, deg))
    delta_R = rot_about_axis(axis[None], np.array([angle]))[0]   # (3,3)
    delta_t = rng.uniform(-trans_m, trans_m, 3)
    return delta_R, delta_t


def recover_joint_axes(node_quat: np.ndarray, parent: np.ndarray,
                       min_deg: float = 3.0, min_frames: int = 20
                       ) -> tuple[np.ndarray, np.ndarray]:
    """Recover each joint's fixed parent-local revolute axis from its own motion.

    node_quat: (T, 25, 4) per-frame node orientations (x, y, z, w).
    parent:    (25,) parent index per node (root's parent is itself / 0).
    A joint's parent-relative rotation R_rel(t) = R(q_parent)^T R(q_child) spins about a
    fixed axis; we average the per-frame axis over frames with angle > min_deg. Joints with
    fewer than min_frames active frames (tips / coupled joints) get has_dof=False. Joints in
    AUG_EXCLUDE_NODES (index/middle/ring/pinky MCP knuckles) are also forced has_dof=False by
    policy -- we do not want to jitter finger splay, only grasp curl.
    Returns axes (25, 3) (unit rows where has_dof, zero rows otherwise) and has_dof (25,) bool."""
    node_quat = np.asarray(node_quat, dtype=np.float64)
    Rn = quats_to_R(node_quat)                          # (T,25,3,3)
    axes = np.zeros((25, 3)); has_dof = np.zeros(25, dtype=bool)
    thr = np.deg2rad(min_deg)
    for j in range(1, 25):
        if j in AUG_EXCLUDE_NODES:                      # policy exclusion (see AUG_EXCLUDE_NODES)
            continue
        p = int(parent[j])
        Rp, Rj = Rn[:, p], Rn[:, j]                     # (T,3,3)
        Rrel = np.einsum("tij,tjk->tik", np.transpose(Rp, (0, 2, 1)), Rj)  # Rp^T Rj
        tr = np.clip((np.trace(Rrel, axis1=1, axis2=2) - 1.0) / 2.0, -1.0, 1.0)
        ang = np.arccos(tr)                             # (T,)
        ax = np.stack([Rrel[:, 2, 1] - Rrel[:, 1, 2],
                       Rrel[:, 0, 2] - Rrel[:, 2, 0],
                       Rrel[:, 1, 0] - Rrel[:, 0, 1]], axis=-1)             # (T,3), = 2 sin(ang) u
        m = ang > thr
        if int(m.sum()) < min_frames:
            continue
        u = ax[m] / (2.0 * np.sin(ang[m]))[:, None]     # unit axis per active frame
        ref = u[0]
        sgn = np.sign(u @ ref); sgn[sgn == 0] = 1.0     # hemisphere-align before averaging
        um = (u * sgn[:, None]).mean(0)
        nn = np.linalg.norm(um)
        if nn < 1e-6:
            continue
        axes[j] = um / nn; has_dof[j] = True
    return axes, has_dof


def repose_skeleton(node_pos: np.ndarray, node_quat: np.ndarray, parent: np.ndarray,
                    axes: np.ndarray, dtheta_rad: np.ndarray) -> np.ndarray:
    """Re-pose the rigid-bone skeleton by adding a per-joint delta about each joint's axis.

    Forward kinematics per frame, parent-before-child (node order is topologically sorted):
      R_rel'(j) = R_rel(j) @ rot_about_axis(axes[j], dtheta[j]);  A_j = A_parent @ R_rel'(j);
      p_j = p_parent + A_parent @ b_j,  where b_j = R(q_parent)^T (pos_j - pos_parent) (rigid bone).
    With dtheta_rad all-zero this telescopes back to the recorded positions exactly.

    node_pos: (F, 25, 3) absolute wrist-frame positions. node_quat: (F, 25, 4).
    parent: (25,). axes: (25, 3). dtheta_rad: (25,). Returns (F, 25, 3)."""
    node_pos = np.asarray(node_pos, dtype=np.float64)
    Rn = quats_to_R(np.asarray(node_quat, dtype=np.float64))   # (F,25,3,3)
    F = node_pos.shape[0]
    dR = rot_about_axis(axes, dtheta_rad)                       # (25,3,3), identity where dtheta=0
    A = np.zeros((F, 25, 3, 3)); p = np.zeros((F, 25, 3))
    A[:, 0] = np.eye(3); p[:, 0] = node_pos[:, 0]               # root anchor (wrist at origin)
    for j in range(1, 25):
        par = int(parent[j])
        Rp = Rn[:, par]                                         # (F,3,3)
        RpT = np.transpose(Rp, (0, 2, 1))
        Rrel = np.einsum("fij,fjk->fik", RpT, Rn[:, j])         # Rp^T Rj
        Rrel_p = np.einsum("fij,jk->fik", Rrel, dR[j])          # inject delta about joint axis
        A[:, j] = np.einsum("fij,fjk->fik", A[:, par], Rrel_p)
        b = np.einsum("fij,fj->fi", RpT, node_pos[:, j] - node_pos[:, par])  # rigid bone, parent-local
        p[:, j] = p[:, par] + np.einsum("fij,fj->fi", A[:, par], b)
    return p
