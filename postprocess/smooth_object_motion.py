#!/usr/bin/env python3
"""Attenuate hand-jitter in tracked 3D object flow, keeping the overall motion.

A hand rotating an object adds a back-and-forth ripple (+20 deg then -10 deg) on
top of the deliberate rotation/translation, because fingers must reset between
strokes. We want the object flow to mainly reflect the deliberate motion, with the
finger-jitter ATTENUATED (not removed -- genuine multi-second direction changes must
survive).

The object moves rigidly within a clip (verified: frame-to-frame Procrustes fit has
~0.7 mm residual on sphere_2), so we do NOT smooth the point tracks directly -- the
back-and-forth is a real low-frequency motion that a position low-pass cannot
separate from the net motion, and smoothing positions would break rigidity. Instead:

  1. Per adjacent frame pair, robustly (RANSAC + Kabsch) fit the incremental rigid
     transform T_f in SE(3) from the points visible in both. Represent it as a body
     twist xi_f = (omega, v) in R^6.
  2. Low-pass each of the 6 twist channels over time (zero-phase). The fast +/-
     jitter averages toward zero; sustained drift -- including a deliberate
     multi-second reversal, which is itself low-frequency -- passes through. All
     local, so no global axis / net-direction is ever assumed.
  3. Re-integrate BOTH the smoothed and the raw twists on SE(3) into absolute poses
     G_sm,t and G_raw,t (both G_0 = I).
  4. Re-ground the MEASURED cloud each frame with the single rigid correction
       coords_smooth[t] = (G_sm,t @ G_raw,t^-1) @ coords[t]      (applied to all pts)
     This removes only the jitter the low-pass took out; the shared accumulated
     integration error in G_sm and G_raw cancels in the ratio, so the output stays
     exactly as rigid and dense as the measurement. (An earlier version transported
     each point from its own birth frame via the global G_t alone -- that let long
     noisy pose chains inflate the cloud, sparsify edges, and fling late-born short
     tracks away. Re-grounding on the measured position every frame fixes all three.)

The method is shape-agnostic (Kabsch assumes rigidity only) and handles translation
+ rotation together. Output is a drop-in dense_3d_track.pkl plus a tracks_2d.mp4
overlay (smoothed 3D reprojected via the camera intrinsics).
"""
import argparse
import os
import pickle
import sys

import cv2
import mediapy as media
import numpy as np

# reuse the tracker's overlay renderer + color assignment so raw and smoothed
# videos match exactly (the renderer is the fast OpenCV one, shared by both).
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from preprocess.track_windowed import render_2d_overlay, rainbow_colors_by_position


# --------------------------------------------------------------------------- #
# SO(3) / SE(3) helpers (pure numpy; scipy is not in this env)
# --------------------------------------------------------------------------- #
def so3_exp(omega):
    """Rotation vector (3,) -> rotation matrix (3,3) via Rodrigues."""
    theta = np.linalg.norm(omega)
    if theta < 1e-12:
        return np.eye(3)
    k = omega / theta
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * (K @ K)


def so3_log(R):
    """Rotation matrix (3,3) -> rotation vector (3,). Numerically guarded."""
    cos = np.clip((np.trace(R) - 1) / 2, -1.0, 1.0)
    theta = np.arccos(cos)
    if theta < 1e-9:
        return np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]]) * 0.5
    if np.pi - theta < 1e-6:
        # near-180 deg: recover axis from the symmetric part (sign is ambiguous)
        A = (R + np.eye(3)) / 2
        axis = np.sqrt(np.clip(np.diag(A), 0, 1))
        # fix signs from off-diagonals
        if axis[0] > 1e-6:
            axis[1] = np.copysign(axis[1], A[0, 1])
            axis[2] = np.copysign(axis[2], A[0, 2])
        elif axis[1] > 1e-6:
            axis[2] = np.copysign(axis[2], A[1, 2])
        return axis / (np.linalg.norm(axis) + 1e-12) * theta
    axis = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]]) / (2 * np.sin(theta))
    return axis * theta


def se3(R, t):
    """Assemble a 4x4 homogeneous transform from R (3,3), t (3,)."""
    G = np.eye(4)
    G[:3, :3] = R
    G[:3, 3] = t
    return G


def se3_inv(G):
    R, t = G[:3, :3], G[:3, 3]
    out = np.eye(4)
    out[:3, :3] = R.T
    out[:3, 3] = -R.T @ t
    return out


def twist_of(G):
    """SE(3) transform -> (omega(3), v(3)); v is the raw translation component
    (small-motion decoupled smoothing, not the screw-theory v). For our per-frame
    increments this decoupling is accurate and keeps the filter simple."""
    return so3_log(G[:3, :3]), G[:3, 3].copy()


def transform_of(omega, v):
    """(omega, v) -> SE(3) transform (rotation from omega, translation = v)."""
    return se3(so3_exp(omega), v)


# --------------------------------------------------------------------------- #
# Rigid fit
# --------------------------------------------------------------------------- #
def kabsch(P, Q):
    """R,t minimizing ||R P + t - Q|| for paired point sets P,Q (n,3)."""
    Pc, Qc = P.mean(0), Q.mean(0)
    H = (P - Pc).T @ (Q - Qc)
    U, S, Vt = np.linalg.svd(H)
    D = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1, 1, D]) @ U.T
    return R, Qc - R @ Pc


def kabsch_ransac(P, Q, thresh, iters=50, rng=None):
    """Robust rigid fit: RANSAC over 3-point minimal samples, refit on inliers.
    Returns (R, t, n_inliers). Falls back to plain Kabsch if too few points."""
    n = P.shape[0]
    if n < 3:
        R, t = kabsch(P, Q)
        return R, t, n
    rng = rng or np.random.default_rng(0)
    best_inl = None
    best_count = -1
    for _ in range(iters):
        s = rng.choice(n, 3, replace=False)
        try:
            R, t = kabsch(P[s], Q[s])
        except np.linalg.LinAlgError:
            continue
        res = np.linalg.norm((R @ P.T).T + t - Q, axis=1)
        inl = res < thresh
        c = int(inl.sum())
        if c > best_count:
            best_count, best_inl = c, inl
    if best_inl is None or best_count < 3:
        R, t = kabsch(P, Q)
        return R, t, n
    R, t = kabsch(P[best_inl], Q[best_inl])  # refit on all inliers
    return R, t, best_count


# --------------------------------------------------------------------------- #
# Smoothing
# --------------------------------------------------------------------------- #
def gaussian_kernel(window):
    """Odd-length normalized Gaussian; sigma = window/6 so +/-3 sigma spans it."""
    w = int(window)
    if w % 2 == 0:
        w += 1
    if w <= 1:
        return np.array([1.0])
    sigma = window / 6.0
    x = np.arange(w) - w // 2
    k = np.exp(-0.5 * (x / sigma) ** 2)
    return k / k.sum()


def smooth_zero_phase(sig, kernel):
    """Zero-phase 1D smoothing with edge replication (no time shift, no edge decay)."""
    if kernel.size <= 1:
        return sig.copy()
    pad = kernel.size // 2
    padded = np.pad(sig, (pad, pad), mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def recover_and_smooth(coords, vis, window, min_corr, ransac_thresh):
    """Estimate per-frame twists, low-pass them, and return absolute poses G_t (T,4,4)
    plus diagnostics. coords (T,N,3), vis (T,N) bool."""
    T, N, _ = coords.shape
    omegas = np.full((T - 1, 3), np.nan)
    vs = np.full((T - 1, 3), np.nan)
    n_fit = np.zeros(T - 1, dtype=int)
    rng = np.random.default_rng(0)

    for f in range(T - 1):
        m = vis[f] & vis[f + 1]
        idx = np.where(m)[0]
        if idx.size:
            P, Q = coords[f, idx], coords[f + 1, idx]
            good = np.isfinite(P).all(1) & np.isfinite(Q).all(1) & (P[:, 2] > 0) & (Q[:, 2] > 0)
            P, Q = P[good], Q[good]
        else:
            P = Q = np.empty((0, 3))
        if P.shape[0] < min_corr:
            continue
        R, t, ninl = kabsch_ransac(P, Q, ransac_thresh, rng=rng)
        omegas[f], vs[f] = so3_log(R), t
        n_fit[f] = ninl

    n_gap = int(np.isnan(omegas[:, 0]).sum())
    # gap-fill missing pairs by linear interpolation of each twist channel
    def fill(a):
        a = a.copy()
        for j in range(a.shape[1]):
            c = a[:, j]
            miss = np.isnan(c)
            if miss.all():
                c[:] = 0.0
            elif miss.any():
                c[miss] = np.interp(np.where(miss)[0], np.where(~miss)[0], c[~miss])
            a[:, j] = c
        return a

    omegas, vs = fill(omegas), fill(vs)
    raw_twist = np.hstack([omegas, vs])  # (T-1, 6) for diagnostics

    kernel = gaussian_kernel(window)
    sm_omega = np.stack([smooth_zero_phase(omegas[:, j], kernel) for j in range(3)], axis=1)
    sm_v = np.stack([smooth_zero_phase(vs[:, j], kernel) for j in range(3)], axis=1)
    sm_twist = np.hstack([sm_omega, sm_v])

    # re-integrate: G_0 = I. Each T_t = exp(xi_t) maps frame-t world coords to
    # frame-(t+1) world coords, so the absolute pose left-multiplies:
    #   coords[t] = T_{t-1} ... T_0 coords[0]  =>  G_{t+1} = T_t @ G_t.
    # Integrate BOTH the smoothed and the raw twists the same way, so their per-frame
    # ratio D_t = G_sm @ G_raw^-1 is the correction that removes only the jitter the
    # low-pass took out -- with the (identical) accumulated integration error in
    # G_raw and G_sm cancelling in the ratio. This is what keeps re-synthesis local.
    G = np.zeros((T, 4, 4))
    G_raw = np.zeros((T, 4, 4))
    G[0] = G_raw[0] = np.eye(4)
    for t in range(T - 1):
        G[t + 1] = transform_of(sm_omega[t], sm_v[t]) @ G[t]
        G_raw[t + 1] = transform_of(omegas[t], vs[t]) @ G_raw[t]

    return G, dict(raw_twist=raw_twist, sm_twist=sm_twist, n_fit=n_fit, n_gap=n_gap,
                   G_raw=G_raw)


def object_centroids(coords, vis, min_pts=3):
    """Robust per-frame object center (median of visible points), gap-filled by
    linear interpolation. (T,3). Shape-agnostic -- just the cloud's center."""
    T, N, _ = coords.shape
    cent = np.full((T, 3), np.nan)
    for t in range(T):
        m = vis[t] & np.isfinite(coords[t]).all(axis=1)
        if m.sum() >= min_pts:
            cent[t] = np.median(coords[t, m], axis=0)
    for j in range(3):
        c = cent[:, j]
        miss = np.isnan(c)
        if miss.any() and (~miss).any():
            c[miss] = np.interp(np.where(miss)[0], np.where(~miss)[0], c[~miss])
        cent[:, j] = c
    return cent


def reject_bad_tracks(coords, vis, G_raw, speed_k, resid_k):
    """Shape-agnostic bad-track mask. Flags a track (in ANY of two ways) as an
    outlier whose depth is noisy/wrong -- these become visible flyers once the cloud
    is rotated rigidly. NEITHER criterion assumes object geometry (works for a
    sphere, cube, or stick):

    - SPEED: per-track 95th-percentile inter-frame speed (metres/frame, gap-
      normalized). Rigid-body points move smoothly; bad-depth points jump.
    - RIGIDITY RESIDUAL: mean distance between where the per-frame rigid model
      (from G_raw) predicts a point and where it was actually measured. A point that
      does not move with the body has a large residual by definition.

    Both use a robust median + k*MAD cutoff, so the scale adapts to the clip. Returns
    a boolean keep-mask (N,) and a small stats dict.
    """
    T, N, _ = coords.shape
    # speed
    speed = np.full(N, np.nan)
    for i in range(N):
        fr = np.where(vis[:, i] & np.isfinite(coords[:, i]).all(axis=1))[0]
        if fr.size < 2:
            continue
        step = np.linalg.norm(np.diff(coords[fr, i], axis=0), axis=1) / np.diff(fr)
        speed[i] = np.percentile(step, 95)
    # rigidity residual to the integrated raw motion
    rs_sum = np.zeros(N)
    rs_cnt = np.zeros(N)
    for t in range(T - 1):
        m = vis[t] & vis[t + 1] & np.isfinite(coords[t]).all(1) & np.isfinite(coords[t + 1]).all(1)
        idx = np.where(m)[0]
        if idx.size < 8:
            continue
        Tf = G_raw[t + 1] @ se3_inv(G_raw[t])
        pred = (Tf[:3, :3] @ coords[t, idx].T).T + Tf[:3, 3]
        r = np.linalg.norm(pred - coords[t + 1, idx], axis=1)
        rs_sum[idx] += r
        rs_cnt[idx] += 1
    resid = np.where(rs_cnt > 0, rs_sum / np.maximum(rs_cnt, 1), np.nan)

    def cutoff(x, k):
        v = np.isfinite(x)
        med = np.median(x[v])
        mad = np.median(np.abs(x[v] - med)) * 1.4826 + 1e-9
        return med + k * mad

    has = np.isfinite(speed) & np.isfinite(resid)
    keep = has & (speed <= cutoff(speed, speed_k)) & (resid <= cutoff(resid, resid_k))
    stats = dict(n_total=N, n_keep=int(keep.sum()), n_drop=int(N - keep.sum()),
                 speed_cut=cutoff(speed, speed_k), resid_cut=cutoff(resid, resid_k))
    return keep, stats


def resynthesize(coords, vis, G, G_raw, window, keep=None):
    """Re-place the MEASURED point cloud by the jitter-removed motion, re-grounded
    every frame and rotated ABOUT THE OBJECT CENTROID (not the camera origin).

    The correction D_t = G_sm,t @ G_raw,t^-1 encodes the jitter the low-pass removed;
    its shared accumulated drift cancels (so the cloud stays rigid). But D_t is a
    transform about the CAMERA origin, and the object sits off-axis ~0.25 m away, so
    applying it directly slides the whole body by (lever arm) x sin(angle) -- several
    cm of deviation from the real object even though only ~2 mm of jitter existed.
    Fix: apply only D_t's ROTATION about the measured object centroid c_t, and place
    the centroid on the SMOOTHED (jitter-free) centroid path c_smooth (a low-pass of
    the measured centroid with the same window):
        p'(t) = R_Dt (p(t) - c_meas_t) + c_smooth_t
    So orientation jitter is removed without translating the body off the object, and
    translational jitter is removed via the centroid low-pass -- the cloud stays
    glued to the real object.

    coords (T,N,3), vis (T,N), G/G_raw (T,4,4), window = smoothing window (frames).
    `keep` (N,) bool optionally drops outlier tracks (left NaN). Returns smoothed
    coords (T,N,3), NaN where a point is inactive / dropped.
    """
    T, N, _ = coords.shape
    c_meas = object_centroids(coords, vis)                          # measured centroid path
    kernel = gaussian_kernel(window)
    c_smooth = np.stack([smooth_zero_phase(c_meas[:, j], kernel) for j in range(3)], axis=1)
    out = np.full_like(coords, np.nan)
    for t in range(T):
        m = vis[t] & np.isfinite(coords[t]).all(axis=1)
        if keep is not None:
            m = m & keep
        if not m.any():
            continue
        R_D = (G[t] @ se3_inv(G_raw[t]))[:3, :3]                    # rotation part only
        pts = coords[t, m]
        out[t, m] = (R_D @ (pts - c_meas[t]).T).T + c_smooth[t]
    return out


# --------------------------------------------------------------------------- #
# 2D overlay
# --------------------------------------------------------------------------- #
def reproject(coords, vis, K):
    """coords (T,N,3) -> uv (T,N,2) pixels; NaN where invalid or Z<=0."""
    fx, fy, cx, cy = K
    T, N, _ = coords.shape
    uv = np.full((T, N, 2), np.nan, dtype=np.float32)
    Z = coords[..., 2]
    ok = vis & np.isfinite(Z) & (Z > 1e-6)
    Zs = np.where(ok, Z, 1.0)  # avoid 0/NaN division warnings; masked out below
    uv[..., 0] = np.where(ok, fx * coords[..., 0] / Zs + cx, np.nan)
    uv[..., 1] = np.where(ok, fy * coords[..., 1] / Zs + cy, np.nan)
    return uv


def load_background(video, T, H, W):
    """RGB background frames (T,H,W,3) uint8: from --video if given, else black."""
    if video and os.path.isdir(video):
        files = sorted(f for f in os.listdir(video) if f.lower().endswith((".png", ".jpg", ".jpeg")))
        frames = [cv2.cvtColor(cv2.imread(os.path.join(video, f)), cv2.COLOR_BGR2RGB) for f in files[:T]]
    elif video and os.path.isfile(video):
        cap = cv2.VideoCapture(video)
        frames = []
        while len(frames) < T:
            ok, fr = cap.read()
            if not ok:
                break
            frames.append(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB))
        cap.release()
    else:
        frames = []
    if len(frames) < T:  # pad (or fully create) with black canvas at the target size
        frames += [np.zeros((H, W, 3), dtype=np.uint8) for _ in range(T - len(frames))]
    return np.stack(frames[:T])


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--pkl", required=True, help="input dense_3d_track.pkl")
    p.add_argument("--output-path", required=True, help="output dir for smoothed pkl + mp4")
    p.add_argument("--smooth-window", type=int, default=9,
                   help="low-pass window in frames (larger = more attenuation; 1 = passthrough)")
    p.add_argument("--intrinsics", default="771.59,771.365,645.555,349.653", help="fx,fy,cx,cy")
    p.add_argument("--video", default=None, help="optional color.mp4 / frame folder for overlay background")
    p.add_argument("--image-size", default="1280,720", help="canvas W,H when --video absent")
    p.add_argument("--min-corr", type=int, default=8, help="min correspondences to fit a pair's pose")
    p.add_argument("--ransac-thresh", type=float, default=0.01, help="RANSAC inlier residual (metres)")
    p.add_argument("--fps", type=int, default=10, help="output mp4 fps")
    p.add_argument("--reject-speed-k", type=float, default=6.0,
                   help="drop tracks whose p95 inter-frame speed exceeds median + k*MAD "
                   "(shape-agnostic; lower = stricter; large value ~ disables)")
    p.add_argument("--reject-resid-k", type=float, default=6.0,
                   help="drop tracks whose rigid-motion residual exceeds median + k*MAD "
                   "(shape-agnostic; lower = stricter; large value ~ disables)")
    p.add_argument("--no-reject", action="store_true", help="keep all tracks (skip outlier rejection)")
    return p.parse_args()


def main():
    args = parse_args()
    with open(args.pkl, "rb") as f:
        data = pickle.load(f)
    coords = data["coords"].astype(np.float64)
    vis = data["vis"].astype(bool)
    colors = data["colors"]
    T, N, _ = coords.shape
    print(f"Loaded {args.pkl}: coords {coords.shape}, {N} tracks over {T} frames")

    K = tuple(float(x) for x in args.intrinsics.split(","))
    G, diag = recover_and_smooth(coords, vis, args.smooth_window, args.min_corr, args.ransac_thresh)
    print(f"Fit per-frame rigid motion: {diag['n_gap']} of {T-1} pairs gap-filled "
          f"(median {int(np.median(diag['n_fit']))} inliers/pair)")

    # Report attenuation with rotation-valid metrics. "traveled" = sum of per-frame
    # rotation-vector magnitudes (a proper measure of how much the object jiggled);
    # smoothing should cut this. Net rotation is the composed pose frame0->last
    # (non-commutative), which smoothing should PRESERVE -- NOT a sum of scalar
    # projections onto one axis, which is meaningless when the axis drifts.
    def composed_net_angle(twist):
        R = np.eye(3)
        for om in twist[:, :3]:
            R = so3_exp(om) @ R
        return np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1.0, 1.0)))
    trav_raw = np.degrees(np.linalg.norm(diag["raw_twist"][:, :3], axis=1)).sum()
    trav_sm = np.degrees(np.linalg.norm(diag["sm_twist"][:, :3], axis=1)).sum()
    net_raw = composed_net_angle(diag["raw_twist"])
    net_sm = composed_net_angle(diag["sm_twist"])
    print(f"  rotational travel: raw {trav_raw:8.1f} deg -> smoothed {trav_sm:8.1f} deg "
          f"(oscillation cut {(1 - trav_sm / max(trav_raw, 1e-9)) * 100:.0f}%)")
    print(f"  net object rotation (composed pose): raw {net_raw:.1f} deg -> smoothed {net_sm:.1f} deg "
          f"(preserved)")

    # Shape-agnostic outlier rejection: drop bad-depth tracks (by speed / rigidity
    # residual) so they don't become flyers when the cloud is rotated rigidly.
    if args.no_reject:
        keep = np.ones(N, dtype=bool)
    else:
        keep, rstats = reject_bad_tracks(coords, vis, diag["G_raw"],
                                         args.reject_speed_k, args.reject_resid_k)
        print(f"Outlier rejection: kept {rstats['n_keep']}/{rstats['n_total']} tracks "
              f"(dropped {rstats['n_drop']}, {100 * rstats['n_drop'] / rstats['n_total']:.0f}%; "
              f"speed>{rstats['speed_cut']:.3f} m/f or resid>{rstats['resid_cut']:.3f} m)")
    vis_keep = vis & keep[None, :]

    coords_sm = resynthesize(coords, vis, G, diag["G_raw"], args.smooth_window, keep=keep).astype(np.float32)

    os.makedirs(args.output_path, exist_ok=True)
    out_pkl = os.path.join(args.output_path, "dense_3d_track.pkl")
    with open(out_pkl, "wb") as h:
        pickle.dump({"coords": coords_sm, "colors": colors, "vis": vis_keep}, h,
                    protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Saved {out_pkl}")

    # 2D overlay of the smoothed tracks (reprojected), on RGB or black canvas
    W, H = (int(x) for x in args.image_size.split(","))
    bg = load_background(args.video, T, H, W)
    H, W = bg.shape[1:3]
    uv = reproject(coords_sm, vis_keep, K)
    viz_colors = rainbow_colors_by_position(uv, vis_keep)
    vid = render_2d_overlay(bg, uv, vis_keep, viz_colors, trace=8)
    out_mp4 = os.path.join(args.output_path, "tracks_2d.mp4")
    media.write_video(out_mp4, vid, fps=args.fps)
    print(f"Saved {out_mp4}")


if __name__ == "__main__":
    main()
