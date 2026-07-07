"""Record3D visualizer

Parse and stream record3d captures. To get the demo data, see `./assets/download_record3d_dance.sh`.
"""

import os
import sys


sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import logging
import pickle
import time
from pathlib import Path
from typing import Optional

# viser's websocket server logs a full traceback for every stray/incomplete
# connection (browser preconnects, the share proxy, port probes). These are
# harmless and don't affect playback, so quiet the logger to stop the spam.
logging.getLogger("websockets").setLevel(logging.CRITICAL)

import cv2
import imageio
import mediapy as media
import numpy as onp
import numpy as np
import tyro
import viser
import viser.extras
import viser.transforms as tf
from densetrack3d.datasets.custom_data import read_data
from tqdm.auto import tqdm


def main(
    filepath: str = "results/demo/yellow-duck/dense_3d_track.pkl",
    video_path: Optional[str] = None,
    mask_path: Optional[str] = None,
    mask_reso: tuple = (384, 512),
    downsample_factor: int = 1,
    max_frames: int = 100,
    share: bool = True,
    port: int = 8080,
    point_size: float = 0.001,
    min_depth: float = 0.0,
    max_depth: float = 0.5,
) -> None:
    server = viser.ViserServer(port=port)
    if share:
        server.request_share_url()

    print("Loading frames!")

    # Load the dense 3D tracks from the given pkl file.
    with open(filepath, "rb") as handle:
        trajs_3d_dict = pickle.load(handle)

    coords = trajs_3d_dict["coords"].astype(np.float32)  # T N 3
    colors = trajs_3d_dict["colors"].astype(np.float32) / 255.0  # N 3
    vis = trajs_3d_dict["vis"].astype(np.float32)  # T N
    # trajs = trajs_data[:, :, :3] # T N 3
    # trajs[..., :2] *= trajs[..., 2:3]

    num_frames, num_points = coords.shape[:2]
    print(f"Num frames {num_frames}, Num points {num_points}")

    # Optionally restrict to a first-frame object mask. A full pkl has one track
    # per pixel of the model grid (mask_reso, row-major), so a mask resized
    # (nearest) to that grid indexes straight onto the N axis.
    if mask_path is not None:
        H_m, W_m = mask_reso
        if num_points != H_m * W_m:
            raise ValueError(
                f"pkl has {num_points} tracks but mask_reso {mask_reso} implies {H_m * W_m}. "
                "This pkl was likely already masked, or produced at a different resolution."
            )
        mask = np.load(mask_path) if mask_path.endswith(".npy") else media.read_image(mask_path)
        mask = np.asarray(mask)
        if mask.ndim == 3:
            mask = mask[..., :3].max(axis=-1)
        mask_bin = (mask > 0).astype(np.uint8)
        mask_model = cv2.resize(mask_bin, (W_m, H_m), interpolation=cv2.INTER_NEAREST).astype(bool)
        keep = mask_model.reshape(-1)
        coords, colors, vis = coords[:, keep], colors[keep], vis[:, keep]
        num_points = coords.shape[1]
        print(f"Applied object mask {mask_path}: {num_points} tracks kept")

    # Optionally load the source RGB video to show a camera frustum per frame.
    if video_path is not None:
        video, _ = read_data(full_path=video_path)
    else:
        video = None
    # Add playback UI.
    with server.gui.add_folder("Playback"):
        gui_timestep = server.gui.add_slider(
            "Timestep",
            min=0,
            max=num_frames - 1,
            step=1,
            initial_value=0,
            disabled=True,
        )
        gui_next_frame = server.gui.add_button("Next Frame", disabled=True)
        gui_prev_frame = server.gui.add_button("Prev Frame", disabled=True)
        gui_playing = server.gui.add_checkbox("Playing", True)
        gui_framerate = server.gui.add_slider("FPS", min=1, max=60, step=0.1, initial_value=12)
        gui_framerate_options = server.gui.add_button_group("FPS options", ("10", "20", "30", "60"))

    # Frame step buttons.
    @gui_next_frame.on_click
    def _(_) -> None:
        gui_timestep.value = (gui_timestep.value + 1) % num_frames

    @gui_prev_frame.on_click
    def _(_) -> None:
        gui_timestep.value = (gui_timestep.value - 1) % num_frames

    # Disable frame controls when we're playing.
    @gui_playing.on_update
    def _(_) -> None:
        gui_timestep.disabled = gui_playing.value
        gui_next_frame.disabled = gui_playing.value
        gui_prev_frame.disabled = gui_playing.value

    # Set the framerate when we click one of the options.
    @gui_framerate_options.on_click
    def _(_) -> None:
        gui_framerate.value = int(gui_framerate_options.value)

    prev_timestep = gui_timestep.value

    # Toggle frame visibility when the timestep slider changes.
    @gui_timestep.on_update
    def _(_) -> None:
        nonlocal prev_timestep
        current_timestep = gui_timestep.value
        with server.atomic():
            frame_nodes[current_timestep].visible = True
            frame_nodes[prev_timestep].visible = False
        prev_timestep = current_timestep
        server.flush()  # Optional!

    # Load in frames.
    server.scene.add_frame(
        "/frames",
        wxyz=(1.0, 0.0, 0.0, 0.0),
        position=(0, 0, 0),
        show_axes=False,
    )

    frame_nodes: list[viser.FrameHandle] = []
    for i in tqdm(range(num_frames)):
        frame_nodes.append(server.scene.add_frame(f"/frames/t{i}", show_axes=False))

        # Keep only visible points whose depth (Z) is within [min_depth, max_depth].
        # This drops the collapsed invalid-depth points that otherwise blow up the scene bounds.
        z = coords[i][:, 2]
        mask = (vis[i] > 0.5) & (z >= min_depth) & (z <= max_depth)

        # Place the point cloud in the frame.
        server.scene.add_point_cloud(
            name=f"/frames/t{i}/pos",
            points=coords[i][mask],
            colors=colors[mask],
            point_size=point_size,
            point_shape="rounded",
            wxyz=(1.0, 0.0, 0.0, 0.0),
            position=(0.0, 0, 0),
        )

        if video is not None:
            img_i = video[i]
            img_h, img_w = img_i.shape[:2]
            # Place the frustum.
            fov = 2 * onp.arctan2(img_h / 2, img_w)
            aspect = img_w / img_h

            server.scene.add_camera_frustum(
                f"/frames/t{i}/frustum",
                fov=fov,
                aspect=aspect,
                scale=0.5,
                image=img_i,
                wxyz=(1.0, 0.0, 0.0, 0.0),
                position=(0.0, 0.0, -2.0),
            )

    # Hide all but the current frame.
    for i, frame_node in enumerate(frame_nodes):
        frame_node.visible = i == gui_timestep.value

    # Playback update loop.
    prev_timestep = gui_timestep.value
    while True:
        if gui_playing.value:
            gui_timestep.value = (gui_timestep.value + 1) % num_frames

        time.sleep(1.0 / gui_framerate.value)


if __name__ == "__main__":
    tyro.cli(main)
