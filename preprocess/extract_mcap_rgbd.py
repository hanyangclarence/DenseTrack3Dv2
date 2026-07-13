#!/usr/bin/env python3
"""Extract ZED RGB-D from a ROS 2 mcap into compact, training-ready artifacts.

Reads two topics from an mcap recording and writes:
    <output-dir>/color.mp4      RGB (JPEG frames re-encoded to mp4)
    <output-dir>/depth.mkv      16-bit millimetre depth, FFV1 / gray16le (LOSSLESS)
    <output-dir>/intrinsics.txt effective fx,fy,cx,cy for the written frames (if --intrinsics given)

With --crop-width, color and depth are horizontally center-cropped to that width
(full height kept) so every downstream stage sees only the interaction region. A
horizontal crop shifts the principal point: cx' = cx - x0 (x0 = (W - Wc)//2), while
fx, fy, cy are unchanged. We record the effective fx,fy,cx',cy in intrinsics.txt so
tracking/smoothing unproject correctly -- crop and intrinsics stay co-located here.

Depth in the mcap is 32FC1 (float32 metres, NaN/inf = invalid). We convert to
uint16 millimetres (invalid -> 0) and store it as a lossless video. This is far
smaller than a folder of 16-bit PNGs and decodes bit-exact -- BUT only through a
raw ffmpeg pipe: cv2.VideoWriter/VideoCapture and imageio's high-level API both
silently truncate 16-bit to 8-bit. write_depth_video/read_depth_video below are
the one true codec path; track_windowed.py imports read_depth_video.

RGB and depth messages are paired by NEAREST timestamp (they are 1:1 and <=9 ms
apart in practice, but nearest-match is robust to drift / unequal counts).

Runs in the track4world env (has rosbags + OpenCV).
"""
import argparse
import os
import subprocess

import cv2
import numpy as np
from pathlib import Path
# NOTE: `rosbags` is imported lazily inside main() so that read_depth_video /
# write_depth_video (which need only subprocess + numpy) can be imported from
# the tracker's env (densetrack3d), which does not have rosbags installed.


def write_depth_video(path, depth_frames, H, W, fps):
    """Encode uint16 mm depth frames to FFV1/gray16le at `path` (lossless).

    depth_frames: iterable of (H, W) uint16 arrays. Written via a raw ffmpeg
    pipe -- do NOT substitute cv2/imageio (they drop to 8-bit).
    """
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-f", "rawvideo", "-pix_fmt", "gray16le", "-s", f"{W}x{H}", "-r", str(fps),
        "-i", "pipe:0", "-c:v", "ffv1", "-level", "3", "-pix_fmt", "gray16le", path,
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    for d in depth_frames:
        proc.stdin.write(np.ascontiguousarray(d, dtype="<u2").tobytes())
    proc.stdin.close()
    if proc.wait() != 0:
        raise RuntimeError(f"ffmpeg depth encode failed for {path}")


def read_depth_video(path):
    """Decode an FFV1/gray16le depth video back to (T, H, W) uint16 (bit-exact)."""
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=width,height", "-of", "csv=p=0:s=x", path],
        stdout=subprocess.PIPE, check=True, text=True,
    ).stdout.strip()
    W, H = (int(x) for x in probe.split("x"))
    raw = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", path, "-f", "rawvideo", "-pix_fmt", "gray16le", "pipe:1"],
        stdout=subprocess.PIPE, check=True,
    ).stdout
    arr = np.frombuffer(raw, dtype="<u2")
    return arr.reshape(-1, H, W)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mcap", required=True, help="path to the .mcap file")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--rgb-topic", default="/dag/zed/compressed")
    p.add_argument("--depth-topic", default="/dag/zed/depth")
    p.add_argument("--depth-scale", type=float, default=1000.0, help="metres -> raw units (1000 = mm)")
    p.add_argument("--fps", type=float, default=7.5)
    p.add_argument("--crop-width", type=int, default=0,
                   help="horizontally center-crop color+depth to this width, keeping full "
                        "height (0 or >= source width = no crop)")
    p.add_argument("--intrinsics", default=None,
                   help="fx,fy,cx,cy at source resolution; if given, the crop-adjusted "
                        "fx,fy,cx-x0,cy is written to <output-dir>/intrinsics.txt")
    return p.parse_args()


def main():
    from rosbags.highlevel import AnyReader

    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    mcap = Path(args.mcap)

    # --- read all RGB and depth messages with timestamps --------------------
    rgb_msgs, dep_msgs = [], []  # each: (timestamp_ns, raw_bytes)
    with AnyReader([mcap.parent]) as reader:
        conns_rgb = [c for c in reader.connections if c.topic == args.rgb_topic]
        conns_dep = [c for c in reader.connections if c.topic == args.depth_topic]
        if not conns_rgb or not conns_dep:
            raise ValueError(f"Topics not found. rgb={bool(conns_rgb)} depth={bool(conns_dep)}")
        for conn, ts, raw in reader.messages(connections=conns_rgb):
            rgb_msgs.append((ts, reader.deserialize(raw, conn.msgtype)))
        for conn, ts, raw in reader.messages(connections=conns_dep):
            dep_msgs.append((ts, reader.deserialize(raw, conn.msgtype)))
    rgb_msgs.sort(key=lambda x: x[0])
    dep_msgs.sort(key=lambda x: x[0])
    dep_ts = np.array([t for t, _ in dep_msgs])
    print(f"Read {len(rgb_msgs)} RGB and {len(dep_msgs)} depth messages")

    # --- decode RGB, build depth stack, pair by nearest timestamp -----------
    color_path = os.path.join(args.output_dir, "color.mp4")
    depth_path = os.path.join(args.output_dir, "depth.mkv")

    H = W = None
    Wc = x0 = None       # cropped width and left offset (set on first frame)
    writer = None
    depth_frames = []
    max_off_ms = 0.0
    for ts, rgb in rgb_msgs:
        bgr = cv2.imdecode(np.frombuffer(rgb.data, np.uint8), cv2.IMREAD_COLOR)
        if H is None:
            H, W = bgr.shape[:2]
            # horizontal center-crop: keep full height, take Wc px from the middle
            Wc = W if args.crop_width <= 0 or args.crop_width >= W else args.crop_width
            x0 = (W - Wc) // 2
            writer = cv2.VideoWriter(color_path, cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (Wc, H))
            if Wc != W:
                print(f"Cropping width {W} -> {Wc} (x0={x0}), height {H} kept")
        writer.write(bgr[:, x0:x0 + Wc])

        j = int(np.argmin(np.abs(dep_ts - ts)))
        max_off_ms = max(max_off_ms, abs(dep_ts[j] - ts) / 1e6)
        dmsg = dep_msgs[j][1]
        d_m = np.frombuffer(dmsg.data, dtype="<f4").reshape(dmsg.height, dmsg.width)
        d_m = np.nan_to_num(d_m, nan=0.0, posinf=0.0, neginf=0.0)
        if (dmsg.height, dmsg.width) != (H, W):
            d_m = cv2.resize(d_m, (W, H), interpolation=cv2.INTER_NEAREST)
        d_mm = np.clip(d_m * args.depth_scale, 0, 65535).astype(np.uint16)
        depth_frames.append(d_mm[:, x0:x0 + Wc])   # crop identically to color
    writer.release()

    write_depth_video(depth_path, depth_frames, H, Wc, args.fps)

    # effective intrinsics for the (possibly cropped) frames: only cx shifts by x0
    if args.intrinsics is not None:
        fx, fy, cx, cy = (float(v) for v in args.intrinsics.split(","))
        cx_eff = cx - x0
        intr_path = os.path.join(args.output_dir, "intrinsics.txt")
        with open(intr_path, "w") as f:
            f.write(f"{fx},{fy},{cx_eff},{cy}\n")
        print(f"Wrote {intr_path}: fx,fy,cx',cy = {fx},{fy},{cx_eff},{cy} (cx {cx} - x0 {x0})")

    stack = np.stack(depth_frames)
    valid = stack > 0
    print(
        f"Wrote {len(depth_frames)} frames at {Wc}x{H}, {args.fps} fps\n"
        f"  color.mp4 {os.path.getsize(color_path)/1e6:.1f} MB\n"
        f"  depth.mkv {os.path.getsize(depth_path)/1e6:.1f} MB "
        f"(valid depth {100*valid.mean():.1f}%, "
        f"max RGB->depth pairing offset {max_off_ms:.1f} ms)"
    )


if __name__ == "__main__":
    main()
