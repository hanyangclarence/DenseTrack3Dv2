# DenseTrack3Dv2 — in-hand manipulation data processing

Turns raw ROS 2 `.mcap` teleop recordings into a training dataset: per-episode object 3D
flow, hand state, and RGB-D. A fork of [DELTAv2](https://snap-research.github.io/DELTAv2/),
used here as a tracking library.

**Data processing only.** The intent model that consumes this data lives in a separate repo
(`~/yanghan/code/intent-model/`). The boundary is files on disk, not imports.

Per episode the pipeline runs: mcap → RGB-D → segmentation → windowed 3D tracking →
trajectory smoothing, then an offline cloud precompute. Output:

| File | Contents |
|---|---|
| `color.mp4` | RGB, center-cropped to the interaction region |
| `depth.mkv` | 16-bit mm depth, FFV1 / `gray16le`, lossless |
| `intrinsics.txt` | `fx,fy,cx,cy`, crop-adjusted — use these downstream |
| `hand.pkl` | `ergonomics (F,20)`, `retarget_values (F,20)`, `raw_node_pose (F,25,7)`, `wrist_quat (F,4)` |
| `seg/mask/*.png` | dense per-frame object masks |
| `object_flow.pkl` | `coords (T,N,3)` metres, `colors (N,3)`, `vis (T,N)` |
| `clouds.npz` | `clouds (T,P,3)`, `n_valid (T,)`, `intrinsics (4,)` |

All 3D data is in the camera-optical frame (+X right, +Y down, +Z into scene), in metres.

## Installation

One repo, two conda envs — the segmentation stack and the tracker want different
Python/torch builds. Grounded-SAM-2 is vendored under `submodules/`, so nothing else to clone.

```bash
git clone --recursive <this-repo>
cd DenseTrack3Dv2
```

### `densetrack3d` — tracking, smoothing, label precompute

```bash
conda create -n densetrack3d python=3.10 cmake=3.14.0 -y
conda activate densetrack3d
conda install pytorch==2.5.1 torchvision==0.20.1 pytorch-cuda=12.1 -c pytorch -c nvidia -y
conda install mkl=2024.0.0 -c conda-forge -y
conda install ffmpeg -c conda-forge -y

pip install pip==24.0
pip install -r requirements.txt
pip install opencv-python-headless==4.10.0.84 einops jaxtyping mediapy viser
```

Stay on torch 2.5.1 — the model uses `torch.nn.attention` (2.3+) and `flex_attention`
(2.5+) unguarded. Two pins matter: `mkl` must be `2024.0.0` (2025.0.0 breaks `import torch`
with `undefined symbol: iJIT_NotifyEvent`), and OpenCV must stay on 4.x (5.x pulls numpy
2.x and breaks torch). Don't re-run `requirements.txt` against a working env — it pins
`triton==3.2.0` and fights the torch install.

### `track4world` — mcap reading, segmentation

```bash
conda create -n track4world python=3.11 -y
conda activate track4world
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 \
    --index-url https://download.pytorch.org/whl/cu121

cd submodules/Grounded-SAM-2
pip install -e .
pip install --no-build-isolation -e grounding_dino
cd ../..

pip install rosbags supervision transformers opencv-python-headless==4.10.0.84 tqdm
```

`rosbags` is a pure-Python mcap reader — no ROS install needed. Grounding-DINO weights are
pulled from HuggingFace on first run.

### Checkpoints

```bash
mkdir -p ./checkpoints/
gdown --fuzzy https://drive.google.com/file/d/1Qa9YFAjBFIzrrHHWf8NZMln5YkLfw4Qa/view -O ./checkpoints/
wget https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt \
     -O ./checkpoints/sam2.1_hiera_large.pt
```

### Verify

```bash
conda activate densetrack3d
python -c "
import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())
from densetrack3d.models.densetrack3d.densetrack3dv2 import DenseTrack3DV2
from preprocess.extract_mcap_rgbd import read_depth_video
print('ok')"
python preprocess/_verify_stitch.py
```

## Running the pipeline

### 1. Process a folder of episodes

```bash
./run_batch.sh \
  --data-dir    ~/data/inhand_manipulation/0718_cube_small \
  --text-prompt "A blue and orange checkered cube." \
  --output-dir  ~/yanghan/data/inhand_manipulation/0718_cube_small
```

Expects `episode_<N>/mcaps/*.mcap` under `--data-dir`. One prompt per folder (all episodes
share the target object). Episodes that already have `object_flow.pkl` are skipped;
failures are logged to `<output-dir>/batch.log` and skipped so the batch runs unattended.

Tuning flags are forwarded to the per-episode driver, defaults tuned for 30 fps ZED capture:
`--win 15` (tracking window), `--stride 5` (window step), `--grid-size 40` (query density),
`--smooth-window 50` (trajectory low-pass), `--crop-width 640`, `--side left` (Manus glove),
`--no-viz`, `--force`.

### 2. Precompute the object clouds

Not called by the batch — run it afterwards, before training downstream.

```bash
conda activate densetrack3d
python scripts/gen_flow_labels.py --data-root ~/yanghan/data/inhand_manipulation
# or --clip <clip-dir> / --episode <episode-dir>
```

For every frame it masks the depth, back-projects to the camera frame using
`intrinsics.txt`, and subsamples to a fixed `P` points → `clouds.npz`. This is the observed
per-frame point cloud (unordered, no cross-frame correspondence), as opposed to
`object_flow.pkl`, which holds tracked point identities over time. It exists purely as a
cache: too slow to redo every epoch, tiny on disk (~7 MB per episode at `P=512`). Frames
with an empty mask get an all-NaN row, which the dataset skips.

Flags: `-P 512`, `--subsample random|fps`, `--min-pts 16`, `--seed 0`, `--force`.

## Configuration

`run_pipeline.sh` (invoked per episode by `run_batch.sh`) has a fixed-config block to edit
for a new host or camera — `REPO`, `SAM2_CKPT`, `INTRINSICS` (ZED @ 1280×720 native,
pre-crop), `DEPTH_SCALE`, `FPS`. `run_batch.sh` has its own `REPO`. Topic names are
per-script argparse defaults.

## Inspecting results

```bash
python visualizer/vis_densetrack3d_trails.py --filepath <ep>/object_flow.pkl --smooth_sigma 3.0
conda run -n track4world python preprocess/viz_hand_cloud_live.py --folder <ep>   # hand + cloud in Genesis frame P
python preprocess/viz_goal_flow.py      --folder <ep>   # 1s goal-pose target
python preprocess/viz_velocity_field.py --folder <ep>   # 3D velocity field
```

## Notes

- Raw mcaps live at `~/data/inhand_manipulation/`, processed episodes at
  `~/yanghan/data/inhand_manipulation/`.
- `preprocess/hand_frame_transforms.py` is a vendored copy; the canonical version is
  `data/hand_frame_transforms.py` in the intent-model repo. Change one, change both.
- `preprocess/run_dino_sam2.py` is ported from `Track4World/scripts/run_dino_sam2.py`; the
  `--max-detections` / `--exclude-prompt` / `--frame-stride` flags are local additions.
- `postprocess/smooth_object_motion.py` is unused — the pose-chain route drifted over long
  clips and added rotational jitter. `smooth_trajectory.py` low-passes each point's path
  directly instead, keeping the object only approximately rigid (~1% distance change).
- Segmentation runs on every frame: the tracker only needs masks at window starts, but
  `gen_flow_labels.py` needs one per frame.
- `scripts/train/`, `scripts/eval/`, and `data/kubric/` are inherited from DELTAv2 and are
  not part of this pipeline.

## Acknowledgements

Tracker: [DELTAv2](https://snap-research.github.io/DELTAv2/). Segmentation:
[Grounded-SAM-2](https://github.com/IDEA-Research/Grounded-SAM-2), driven by a script
ported from [Track4World](https://github.com/TencentARC/Track4World). Visualization:
[Viser](https://viser.studio/main/), [Open3D](https://www.open3d.org/).

```bibtex
@article{ngo2024delta,
  author    = {Ngo, Tuan Duc and Mirzaei, Ashkan and Qian, Gordon and Liang, Hanwen and Gan, Chuang and Kalogerakis, Evangelos and Wonka, Peter and Wang, Chaoyang},
  title     = {DELTAv2: Dense Efficient Long-range 3D Tracking for Any video},
  journal   = {arXiv preprint arXiv:2508.01170},
  year      = {2025}
}
```
