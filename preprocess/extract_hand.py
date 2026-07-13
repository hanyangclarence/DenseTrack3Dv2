#!/usr/bin/env python3
"""Extract hand data from a ZED/teleop mcap, aligned to the camera frames.

Two hand streams are recorded alongside the ZED RGB-D:
    /manus_glove_{side}                  manus_ros2_msgs/msg/ManusGlove   (~120 Hz)
        the raw Manus glove readout: 20 named ergonomics joint angles + a 25-node
        hand skeleton (each node a 6-DOF pose).
    /dg5f_{side}/lj_dg_pospid/reference  control_msgs/msg/MultiDOFCommand (~120 Hz)
        the RETARGETED hand joints (20 DOF reference values) driving the robot hand;
        this is the action signal for the forward-dynamics (Model B) object-flow model.

Only one side is enabled per capture (normally left); the other side's topics carry
zero messages. Both streams run ~4x the 30 Hz camera rate, so we nearest-timestamp
resample them onto the exact camera-frame timeline (the same frames extract_mcap_rgbd.py
writes to color.mp4) -> every saved array is row-aligned with the object flow: row t
corresponds to camera frame t. Output is <output-dir>/hand.pkl (see save_hand for keys).

Runs in the track4world env (has rosbags). Re-reads the mcap independently of the RGB-D
extractor; the camera timeline is recovered by reading RGB-topic message timestamps only
(no image decode).
"""
import argparse
import os
import pickle
from pathlib import Path

import numpy as np

RGB_TOPIC = "/dag/zed/compressed"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mcap", required=True, help="path to the .mcap file")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--side", choices=["left", "right", "auto"], default="left",
                   help="which hand is enabled (default left); 'auto' picks the non-empty side")
    p.add_argument("--rgb-topic", default=RGB_TOPIC, help="camera topic for the frame timeline")
    p.add_argument("--glove-topic", default=None, help="override /manus_glove_{side}")
    p.add_argument("--retarget-topic", default=None, help="override /dg5f_{side}/lj_dg_pospid/reference")
    return p.parse_args()


def glove_topic(side):
    return f"/manus_glove_{side}"


def retarget_topic(side):
    return f"/dg5f_{side}/lj_dg_pospid/reference"


def resolve_side(reader, side):
    """Return the side to use. For 'auto', pick whichever side's glove topic has
    messages; error if none, warn if both. For an explicit side, verify it is non-empty."""
    counts = {}
    for s in ("left", "right"):
        conns = [c for c in reader.connections if c.topic == glove_topic(s)]
        counts[s] = sum(c.msgcount for c in conns)
    print(f"glove message counts: left={counts['left']}, right={counts['right']}")
    if side == "auto":
        nonempty = [s for s in ("left", "right") if counts[s] > 0]
        if not nonempty:
            raise ValueError("No glove messages on either side; nothing to extract.")
        if len(nonempty) == 2:
            print("WARNING: both sides have glove messages; picking 'left'.")
            return "left"
        return nonempty[0]
    if counts[side] == 0:
        raise ValueError(f"--side {side} requested but /manus_glove_{side} has 0 messages "
                         f"(the other side has {counts['right' if side == 'left' else 'left']}). "
                         f"Use --side auto to auto-detect.")
    return side


def nearest_idx(src_ts, query_ts):
    """For each query timestamp, index of the nearest src timestamp (src sorted asc)."""
    pos = np.searchsorted(src_ts, query_ts)
    pos = np.clip(pos, 1, len(src_ts) - 1)
    left = src_ts[pos - 1]
    right = src_ts[pos]
    choose_left = (query_ts - left) <= (right - query_ts)
    idx = np.where(choose_left, pos - 1, pos)
    return idx


def read_frame_timestamps(reader, rgb_topic):
    """Camera-frame timeline: RGB-topic message timestamps, sorted (no image decode)."""
    conns = [c for c in reader.connections if c.topic == rgb_topic]
    if not conns:
        raise ValueError(f"RGB topic {rgb_topic} not found in mcap.")
    ts = [t for _, t, _ in reader.messages(connections=conns)]
    return np.array(sorted(ts), dtype=np.int64)


def read_glove(reader, topic):
    """Read the ManusGlove stream. Returns timestamps (M,) int64 and per-message
    ergonomics (M,20), raw_node_pose (M,25,7), plus the stable names/topology from
    the first message."""
    conns = [c for c in reader.connections if c.topic == topic]
    ts, ergo, nodes = [], [], []
    ergo_names = node_names = node_parent = None
    for conn, t, raw in reader.messages(connections=conns):
        m = reader.deserialize(raw, conn.msgtype)
        if ergo_names is None:
            ergo_names = [e.type for e in m.ergonomics]
            node_names = [f"{n.chain_type}/{n.joint_type}" for n in m.raw_nodes]
            node_parent = np.array([n.parent_node_id for n in m.raw_nodes], dtype=np.int32)
        ts.append(t)
        ergo.append([e.value for e in m.ergonomics])
        nodes.append([[n.pose.position.x, n.pose.position.y, n.pose.position.z,
                       n.pose.orientation.x, n.pose.orientation.y,
                       n.pose.orientation.z, n.pose.orientation.w] for n in m.raw_nodes])
    return (np.array(ts, dtype=np.int64), np.array(ergo, dtype=np.float32),
            np.array(nodes, dtype=np.float32), ergo_names, node_names, node_parent)


def read_retarget(reader, topic):
    """Read the MultiDOFCommand stream. Returns timestamps (M,) int64, values (M,20),
    and the stable dof_names from the first message."""
    conns = [c for c in reader.connections if c.topic == topic]
    ts, vals = [], []
    dof_names = None
    for conn, t, raw in reader.messages(connections=conns):
        m = reader.deserialize(raw, conn.msgtype)
        if dof_names is None:
            dof_names = list(m.dof_names)
        ts.append(t)
        vals.append(np.asarray(m.values, dtype=np.float32))
    return np.array(ts, dtype=np.int64), np.array(vals, dtype=np.float32), dof_names


def save_hand(path, data):
    with open(path, "wb") as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)


def main():
    from rosbags.highlevel import AnyReader

    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    mcap = Path(args.mcap)

    with AnyReader([mcap.parent]) as reader:
        side = resolve_side(reader, args.side)
        g_topic = args.glove_topic or glove_topic(side)
        r_topic = args.retarget_topic or retarget_topic(side)
        print(f"Using side={side}: glove={g_topic}, retarget={r_topic}")

        frame_ts = read_frame_timestamps(reader, args.rgb_topic)
        F = len(frame_ts)
        print(f"Camera timeline: {F} frames")

        g_ts, ergo, nodes, ergo_names, node_names, node_parent = read_glove(reader, g_topic)
        r_ts, rvals, dof_names = read_retarget(reader, r_topic)
        print(f"Glove: {len(g_ts)} msgs, ergonomics {ergo.shape}, nodes {nodes.shape}")
        print(f"Retarget: {len(r_ts)} msgs, values {rvals.shape}")

    # nearest-timestamp resample both streams onto the camera timeline
    gi = nearest_idx(g_ts, frame_ts)
    ri = nearest_idx(r_ts, frame_ts)
    max_off_ms = max(np.abs(g_ts[gi] - frame_ts).max(),
                     np.abs(r_ts[ri] - frame_ts).max()) / 1e6

    data = {
        "side": side,
        "frame_timestamps_ns": frame_ts,
        "retarget_dof_names": dof_names,
        "retarget_values": rvals[ri],                 # (F, 20)
        "ergonomics_names": ergo_names,
        "ergonomics": ergo[gi],                        # (F, 20)
        "raw_node_names": node_names,
        "raw_node_parent": node_parent,               # (25,)
        "raw_node_pose": nodes[gi],                    # (F, 25, 7)
        "max_align_offset_ms": float(max_off_ms),
    }
    out_path = os.path.join(args.output_dir, "hand.pkl")
    save_hand(out_path, data)
    print(f"Wrote {out_path}: side={side}, F={F}, "
          f"retarget_values {data['retarget_values'].shape}, "
          f"ergonomics {data['ergonomics'].shape}, "
          f"raw_node_pose {data['raw_node_pose'].shape}, "
          f"max align offset {max_off_ms:.2f} ms")


if __name__ == "__main__":
    main()
