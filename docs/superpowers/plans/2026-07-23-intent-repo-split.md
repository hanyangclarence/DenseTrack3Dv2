# Intent-Model Repo Split Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extract the object-flow intent model from `DenseTrack3Dv2` into a new standalone repo at `/home/labeng/yanghan/code/intent-model/` running on Python 3.12 (for `rclpy` + `torch`), leaving `DenseTrack3Dv2` as the data-processing pipeline.

**Architecture:** The two repos communicate only through files on disk (`object_flow.pkl` / `hand.pkl`) — no cross-repo code import. Intent files MOVE to the new repo; a small set of DenseTrack model primitives and two viz helpers are COPIED (vendored); `hand_frame_transforms.py` moves canonical to the new repo with a vendored copy left behind for a data-side viewer.

**Tech Stack:** Python 3.12, torch 2.5.1, PyTorch Lightning, einops, jaxtyping, viser, rclpy (from ROS Jazzy `/opt/ros/jazzy`, sourced not pip-installed).

## Global Constraints

- Source repo root: `/home/labeng/yanghan/code/vision/DenseTrack3Dv2` (referred to as `$SRC`).
- New repo root: `/home/labeng/yanghan/code/intent-model` (referred to as `$NEW`).
- New-repo Python env: conda env `intent`, Python 3.12. Until it exists, syntax checks use the system `python3` (3.12) at `/usr/bin/python3.12`.
- After the split, the new repo must have **ZERO** imports matching `densetrack3d`, `preprocess.`, or `worldmodel`. This is a hard gate verified by grep.
- **NO git commits** in either repo until the user explicitly approves (standing user rule). `git init` + staging is allowed; committing is not.
- **Do NOT delete** any file from `$SRC` until the new repo passes all verification (Task 9). Moves are done as copy-then-verify-then-delete.
- **Do NOT move checkpoints** now. A separate step (Task 9, gated) copies the baseline `.ckpt` after the currently-running sweep baseline finishes.
- No functional change to the intent model — relocation + import rewrite + vendoring only. Copied primitives must remain byte-for-byte behaviorally identical to their DenseTrack originals (verified by output parity in Task 9).
- Package import scheme in `$NEW`: top-level packages `intent_model`, `data`, `scripts` are importable because each entrypoint inserts the repo root on `sys.path` (mirrors the existing `sys.path.insert(0, dirname(dirname(abspath(__file__))))` idiom already in the source scripts).

---

### Task 1: Scaffold the new repo

**Files:**
- Create: `$NEW/` directory tree
- Create: `$NEW/.gitignore`
- Create: `$NEW/requirements.txt`
- Create: `$NEW/pyproject.toml`
- Create: `$NEW/README.md`
- Create: `$NEW/intent_model/__init__.py`, `$NEW/intent_model/modules/__init__.py`, `$NEW/data/__init__.py`, `$NEW/scripts/__init__.py` (empty package markers where needed)

**Interfaces:**
- Produces: the directory skeleton and `git` repo that all later tasks write into.

- [ ] **Step 1: Create the directory tree**

```bash
mkdir -p /home/labeng/yanghan/code/intent-model/{intent_model/modules,data,scripts,configs,docs/specs,logdirs}
cd /home/labeng/yanghan/code/intent-model
git init
```

- [ ] **Step 2: Write `.gitignore`**

Create `$NEW/.gitignore`:

```gitignore
__pycache__/
*.pyc
*.ckpt
logdirs/
wandb/
*.log
.DS_Store
# large data root lives outside the repo
/home/labeng/yanghan/data/
```

- [ ] **Step 3: Write `requirements.txt`**

Create `$NEW/requirements.txt`:

```
torch==2.5.1
torchvision
lightning
einops
jaxtyping
numpy
wandb
viser
opencv-python
matplotlib
tqdm
# rclpy is NOT pip-installed: `source /opt/ros/jazzy/setup.bash` (ROS Jazzy, py3.12) provides it.
```

- [ ] **Step 4: Write `pyproject.toml`**

Create `$NEW/pyproject.toml`:

```toml
[project]
name = "intent-model"
version = "0.1.0"
description = "Object-flow intent world model (split from DenseTrack3Dv2)"
requires-python = ">=3.12"

[tool.setuptools]
packages = ["intent_model", "intent_model.modules", "data", "scripts"]
```

- [ ] **Step 5: Write a placeholder `README.md`**

Create `$NEW/README.md` with a short description: what the repo is (object-flow intent model), that it was split from DenseTrack3Dv2, the disk contract (`object_flow.pkl`/`hand.pkl` produced by DenseTrack3Dv2), the env setup (conda py3.12 + `source /opt/ros/jazzy/setup.bash`), and the train command `python scripts/train_intent.py fit --config configs/intent.yaml`.

- [ ] **Step 6: Create empty package markers**

```bash
cd /home/labeng/yanghan/code/intent-model
touch intent_model/modules/__init__.py data/__init__.py scripts/__init__.py
```
(`intent_model/__init__.py` is written in Task 3; leave it out here.)

- [ ] **Step 7: Verify the skeleton**

```bash
find /home/labeng/yanghan/code/intent-model -type d -not -path '*/.git/*'
```
Expected: the tree `intent_model/modules`, `data`, `scripts`, `configs`, `docs/specs`, `logdirs`.

---

### Task 2: Vendor the DenseTrack model primitives into `intent_model/modules/`

**Files:**
- Create: `$NEW/intent_model/modules/nn_blocks.py` (from `$SRC/densetrack3d/models/densetrack3d/blocks.py`)
- Create: `$NEW/intent_model/modules/embeddings.py` (from `$SRC/densetrack3d/models/embeddings.py`)
- Create: `$NEW/intent_model/modules/losses.py` (from `$SRC/densetrack3d/models/loss.py` + `model_utils.py`)

**Interfaces:**
- Produces: `from intent_model.modules.nn_blocks import AttnBlock, CrossAttnBlock`; `from intent_model.modules.embeddings import get_3d_embedding`; `from intent_model.modules.losses import balanced_bce_loss`. These are what Tasks 3 rewrite the worldmodel imports to point at.

- [ ] **Step 1: Extract the nn_blocks source ranges**

The four classes needed are `Mlp` (L68–112), `Attention` (L462–553), `AttnBlock` (L934–965), `CrossAttnBlock` (L967–999) in `$SRC/densetrack3d/models/densetrack3d/blocks.py`. `Mlp` uses `to_2tuple`, which is `_ntuple(2)` (helper at L48–65). Extract them:

```bash
cd /home/labeng/yanghan/code/vision/DenseTrack3Dv2
B=densetrack3d/models/densetrack3d/blocks.py
{
  sed -n '48,55p' "$B"      # _ntuple
  echo ""
  echo "to_2tuple = _ntuple(2)"
  echo ""
  sed -n '68,112p' "$B"     # Mlp
  echo ""
  sed -n '462,553p' "$B"    # Attention
  echo ""
  sed -n '934,999p' "$B"    # AttnBlock + CrossAttnBlock (contiguous)
} > /tmp/nn_blocks_body.py
head -5 /tmp/nn_blocks_body.py && echo "..." && wc -l /tmp/nn_blocks_body.py
```
Expected: ~215 lines, starting with `def _ntuple(n):`.

- [ ] **Step 2: Write `nn_blocks.py` with a minimal import header**

Create `$NEW/intent_model/modules/nn_blocks.py`: prepend this exact header, then append the body from `/tmp/nn_blocks_body.py`.

```python
"""Vendored from DenseTrack3Dv2 densetrack3d/models/densetrack3d/blocks.py.
Only the transformer primitives the intent model uses: Mlp, Attention, AttnBlock,
CrossAttnBlock (+ the to_2tuple helper Mlp needs). Copied, not imported, so this repo
has no densetrack3d dependency. Uses SDPA (scaled_dot_product_attention); the original's
flex_attention path and bilinear_sampler are NOT used by these classes and are omitted.
"""
import collections
from functools import partial
from itertools import repeat
from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.attention import SDPBackend
from torch.nn.functional import scaled_dot_product_attention
```

Then:
```bash
cat /tmp/nn_blocks_body.py >> /home/labeng/yanghan/code/intent-model/intent_model/modules/nn_blocks.py
```

- [ ] **Step 3: Verify nn_blocks imports and instantiates**

```bash
cd /home/labeng/yanghan/code/intent-model
/usr/bin/python3.12 -c "import ast; ast.parse(open('intent_model/modules/nn_blocks.py').read()); print('parse ok')"
PYTHONPATH=. /usr/bin/python3.12 -c "
from intent_model.modules.nn_blocks import Mlp, Attention, AttnBlock, CrossAttnBlock
print('import ok:', AttnBlock, CrossAttnBlock)
"
```
Expected: `parse ok` then `import ok: <class ...AttnBlock> <class ...CrossAttnBlock>`.
NOTE: if the env lacks torch, this step is deferred to Task 8's env; still run the `ast.parse` check now.

- [ ] **Step 4: Write `embeddings.py`**

`get_3d_embedding` is L226–253 in `$SRC/densetrack3d/models/embeddings.py`, self-contained (no local-func calls). It uses `torch`. Create `$NEW/intent_model/modules/embeddings.py`:

```python
"""Vendored from DenseTrack3Dv2 densetrack3d/models/embeddings.py: get_3d_embedding only."""
import torch
```
Then append the function body:
```bash
cd /home/labeng/yanghan/code/vision/DenseTrack3Dv2
sed -n '226,253p' densetrack3d/models/embeddings.py >> /home/labeng/yanghan/code/intent-model/intent_model/modules/embeddings.py
```
Verify: `ast.parse` ok, and `get_3d_embedding` present. If the source function references `rearrange` or `math` after extraction, add the matching import to the header (check with `grep -nE "rearrange|math\." $NEW/intent_model/modules/embeddings.py`; the function was confirmed not to use `rearrange`).

- [ ] **Step 5: Write `losses.py`**

`balanced_bce_loss` (L75–98 of `loss.py`) needs `reduce_masked_mean` (L112–159 of `model_utils.py`). Create `$NEW/intent_model/modules/losses.py`:

```python
"""Vendored from DenseTrack3Dv2: balanced_bce_loss (models/loss.py) +
reduce_masked_mean (models/model_utils.py). Copied, not imported."""
import torch
import torch.nn.functional as F
from jaxtyping import Float, Int64
from torch import Tensor
```
Then append `reduce_masked_mean` FIRST (balanced_bce_loss calls it), then `balanced_bce_loss`:
```bash
cd /home/labeng/yanghan/code/vision/DenseTrack3Dv2
{ echo ""; sed -n '112,159p' densetrack3d/models/model_utils.py; echo ""; sed -n '75,98p' densetrack3d/models/loss.py; } \
  >> /home/labeng/yanghan/code/intent-model/intent_model/modules/losses.py
```

- [ ] **Step 6: Verify losses.py**

```bash
cd /home/labeng/yanghan/code/intent-model
/usr/bin/python3.12 -c "import ast; ast.parse(open('intent_model/modules/losses.py').read()); print('parse ok')"
```
Expected: `parse ok`. Confirm `reduce_masked_mean` is defined above `balanced_bce_loss` in the file. If `reduce_masked_mean`'s body references anything not imported (e.g. `EPS`), inline the constant or add the import (check the extracted body).

---

### Task 3: Move the worldmodel package → `intent_model/`

**Files:**
- Create: `$NEW/intent_model/{intent_model,backbone,hand_encoder,heads,point_encoder,types}.py` (copied from `$SRC/densetrack3d/models/worldmodel/`)
- Create: `$NEW/intent_model/__init__.py`

**Interfaces:**
- Consumes: `intent_model.modules.nn_blocks`, `.embeddings`, `.losses` (Task 2).
- Produces: `from intent_model import IntentModel, IntentModelConfig, intent_loss`; `from intent_model.types import FlowItem, IntentBatch, IntentOutput`.

- [ ] **Step 1: Copy the six module files verbatim**

```bash
cd /home/labeng/yanghan/code/vision/DenseTrack3Dv2/densetrack3d/models/worldmodel
cp intent_model.py backbone.py hand_encoder.py heads.py point_encoder.py types.py \
   /home/labeng/yanghan/code/intent-model/intent_model/
```

- [ ] **Step 2: Rewrite imports in `intent_model/intent_model.py`**

In `$NEW/intent_model/intent_model.py`, apply these exact replacements:

| Old line | New line |
|---|---|
| `from densetrack3d.models.loss import balanced_bce_loss` | `from intent_model.modules.losses import balanced_bce_loss` |
| `from densetrack3d.models.worldmodel.backbone import IntentBackbone, init_query_tokens` | `from intent_model.backbone import IntentBackbone, init_query_tokens` |
| `from densetrack3d.models.worldmodel.hand_encoder import HandEncoder` | `from intent_model.hand_encoder import HandEncoder` |
| `from densetrack3d.models.worldmodel.heads import Heads` | `from intent_model.heads import Heads` |
| `from densetrack3d.models.worldmodel.point_encoder import PointEncoder, PosEnc` | `from intent_model.point_encoder import PointEncoder, PosEnc` |
| `from densetrack3d.models.worldmodel.types import IntentBatch, IntentOutput` | `from intent_model.types import IntentBatch, IntentOutput` |

- [ ] **Step 3: Rewrite imports in `backbone.py` and `point_encoder.py`**

In `$NEW/intent_model/backbone.py`:
- `from densetrack3d.models.densetrack3d.blocks import AttnBlock, CrossAttnBlock` → `from intent_model.modules.nn_blocks import AttnBlock, CrossAttnBlock`

In `$NEW/intent_model/point_encoder.py`:
- `from densetrack3d.models.embeddings import get_3d_embedding` → `from intent_model.modules.embeddings import get_3d_embedding`

(`hand_encoder.py`, `heads.py`, `types.py` have no intra-repo imports — no edits.)

- [ ] **Step 4: Write `intent_model/__init__.py`**

Create `$NEW/intent_model/__init__.py`:

```python
"""Object-flow intent world model (split from DenseTrack3Dv2)."""
from intent_model.intent_model import (
    IntentModel,
    IntentModelConfig,
    intent_loss,
)

__all__ = ["IntentModel", "IntentModelConfig", "intent_loss"]
```

- [ ] **Step 5: Verify the package imports with no densetrack3d on path**

```bash
cd /home/labeng/yanghan/code/intent-model
for f in intent_model/*.py; do /usr/bin/python3.12 -c "import ast; ast.parse(open('$f').read())" || echo "PARSE FAIL $f"; done
echo "ast ok"
grep -rnE "densetrack3d|worldmodel|preprocess\." intent_model/ && echo "STALE IMPORTS FOUND" || echo "clean: no stale imports"
```
Expected: `ast ok` and `clean: no stale imports`. (Full runtime import test with torch happens in Task 8.)

---

### Task 4: Move `hand_frame_transforms.py` (canonical) + leave a vendored copy

**Files:**
- Create: `$NEW/data/hand_frame_transforms.py` (canonical, from `$SRC/preprocess/hand_frame_transforms.py`)
- Modify: `$SRC/preprocess/hand_frame_transforms.py` (add sync-note header only)

**Interfaces:**
- Produces: `from data.hand_frame_transforms import wrist_M_rel, placed_hand_camera, recover_joint_axes, repose_skeleton, random_wrist_delta` (used by Task 5), and the viewer-only names `T_HAND_TO_P, transform_cloud_to_P, stabilized_cloud_P, stabilized_hand_P, placed_hand_P`.

- [ ] **Step 1: Copy the file to the new repo**

```bash
cp /home/labeng/yanghan/code/vision/DenseTrack3Dv2/preprocess/hand_frame_transforms.py \
   /home/labeng/yanghan/code/intent-model/data/hand_frame_transforms.py
```
No import rewrites needed (the module imports only `numpy`).

- [ ] **Step 2: Add a sync-note header to the new-repo canonical copy**

At the very top of `$NEW/data/hand_frame_transforms.py` (before the existing docstring), insert:

```python
# CANONICAL COPY. A vendored duplicate lives in DenseTrack3Dv2 at
# preprocess/hand_frame_transforms.py (used by preprocess/viz_hand_cloud_live.py).
# These two files MUST stay in sync: this module defines the training-time hand->camera
# placement (placed_hand_camera is the augmentation choke point). Edit here first.
```

- [ ] **Step 3: Add a matching sync-note header to the DenseTrack3Dv2 vendored copy**

At the very top of `$SRC/preprocess/hand_frame_transforms.py`, insert:

```python
# VENDORED COPY. The CANONICAL version lives in the intent-model repo at
# data/hand_frame_transforms.py. This copy exists only for preprocess/viz_hand_cloud_live.py.
# If you change the transforms, update the canonical copy in intent-model too (must match:
# the intent model trains on placed_hand_camera from the canonical copy).
```

- [ ] **Step 4: Verify both copies parse and are content-identical (below the headers)**

```bash
/usr/bin/python3.12 -c "import ast; ast.parse(open('/home/labeng/yanghan/code/intent-model/data/hand_frame_transforms.py').read()); print('new parse ok')"
/usr/bin/python3.12 -c "import ast; ast.parse(open('/home/labeng/yanghan/code/vision/DenseTrack3Dv2/preprocess/hand_frame_transforms.py').read()); print('old parse ok')"
# bodies (minus the 4-line headers) should be identical:
diff <(tail -n +5 /home/labeng/yanghan/code/intent-model/data/hand_frame_transforms.py) \
     <(tail -n +5 /home/labeng/yanghan/code/vision/DenseTrack3Dv2/preprocess/hand_frame_transforms.py) \
  && echo "bodies identical" || echo "BODIES DIFFER — investigate"
```
Expected: both parse ok, `bodies identical`.

- [ ] **Step 5: Confirm the DenseTrack3Dv2 viewer still resolves its import**

```bash
cd /home/labeng/yanghan/code/vision/DenseTrack3Dv2/preprocess
/usr/bin/python3.12 -c "import ast; ast.parse(open('viz_hand_cloud_live.py').read()); print('viewer parse ok')"
# viz_hand_cloud_live.py uses a bare 'from hand_frame_transforms import ...' relative to preprocess/,
# so the vendored copy must remain at preprocess/hand_frame_transforms.py (it does). Confirm:
test -f hand_frame_transforms.py && echo "vendored copy present"
```
Expected: `viewer parse ok`, `vendored copy present`.

---

### Task 5: Move the `data/` intent files

**Files:**
- Create: `$NEW/data/flow_window_dataset.py` (from `$SRC/data/flow_window_dataset.py`)
- Create: `$NEW/data/viz_flow_window_item.py` (from `$SRC/data/viz_flow_window_item.py`)
- Create: `$NEW/data/flow_stats.npz` (copy of `$SRC/data/flow_stats.npz`)

**Interfaces:**
- Consumes: `data.hand_frame_transforms` (Task 4), `intent_model.types` (Task 3).
- Produces: `from data.flow_window_dataset import FlowWindowDataset`.

- [ ] **Step 1: Copy the files**

```bash
cd /home/labeng/yanghan/code/vision/DenseTrack3Dv2
cp data/flow_window_dataset.py data/viz_flow_window_item.py data/flow_stats.npz \
   /home/labeng/yanghan/code/intent-model/data/
```

- [ ] **Step 2: Rewrite imports in `flow_window_dataset.py`**

In `$NEW/data/flow_window_dataset.py`:
- `from preprocess.hand_frame_transforms import (` → `from data.hand_frame_transforms import (`  (the multi-line import list of `wrist_M_rel, placed_hand_camera, recover_joint_axes, repose_skeleton, random_wrist_delta` is unchanged)
- `from densetrack3d.models.worldmodel.types import FlowItem` → `from intent_model.types import FlowItem`
- The existing `sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))` line stays (it puts `$NEW` root on the path so `data.` and `intent_model.` resolve when run as a script).

- [ ] **Step 3: Rewrite imports in `viz_flow_window_item.py`**

In `$NEW/data/viz_flow_window_item.py`:
- `from data.flow_window_dataset import FlowWindowDataset` → unchanged (still valid).
- `from preprocess.hand_frame_transforms import (` → `from data.hand_frame_transforms import (`

- [ ] **Step 4: Verify**

```bash
cd /home/labeng/yanghan/code/intent-model
for f in data/flow_window_dataset.py data/viz_flow_window_item.py; do
  /usr/bin/python3.12 -c "import ast; ast.parse(open('$f').read())" || echo "PARSE FAIL $f"
done
echo "ast ok"
grep -rnE "densetrack3d|worldmodel|preprocess\." data/*.py && echo "STALE FOUND" || echo "clean"
```
Expected: `ast ok`, `clean`.

---

### Task 6: Copy viz helpers + move the `scripts/`

**Files:**
- Create: `$NEW/scripts/viz_helpers_2d.py` (2 functions copied from `$SRC/preprocess/track_windowed.py`)
- Create: `$NEW/scripts/{train_intent,compute_flow_stats,analyze_intent_ckpt,viz_intent_predictions}.py`
- Create: `$NEW/scripts/sweep_intent.sh`

**Interfaces:**
- Consumes: `intent_model`, `intent_model.types`, `data.flow_window_dataset`, and `scripts.train_intent` (for shared `FlowWindowDataModule`, `collate`, `load_ema_model`, `_cfg_from_dict`).
- Produces: runnable `python scripts/train_intent.py fit --config configs/intent.yaml`.

- [ ] **Step 1: Create `viz_helpers_2d.py` from the two self-contained helpers**

`render_2d_overlay` and `rainbow_colors_by_position` in `$SRC/preprocess/track_windowed.py` are self-contained (need cv2/numpy/matplotlib only). Find their exact line ranges and extract:

```bash
cd /home/labeng/yanghan/code/vision/DenseTrack3Dv2
/usr/bin/python3.12 - <<'EOF'
import re
src=open('preprocess/track_windowed.py').read(); lines=src.split('\n')
starts=[(i,l) for i,l in enumerate(lines) if re.match(r'^(def |class )',l)]
def rng(name):
    idx=next(i for i,(ln,l) in enumerate(starts) if re.match(rf'^def {name}\b',l))
    s=starts[idx][0]; e=starts[idx+1][0] if idx+1<len(starts) else len(lines)
    return s+1, e
print('render_2d_overlay', rng('render_2d_overlay'))
print('rainbow_colors_by_position', rng('rainbow_colors_by_position'))
EOF
```
Create `$NEW/scripts/viz_helpers_2d.py` with this header, then append the two function bodies (in the order printed) using `sed -n 'START,ENDp'`:

```python
"""2D-overlay viz helpers copied from DenseTrack3Dv2 preprocess/track_windowed.py
(render_2d_overlay, rainbow_colors_by_position). Copied so this repo does not import
track_windowed.py, which pulls in the full DenseTrack tracker at module load."""
import cv2
import numpy as np
import matplotlib.pyplot as plt
```
After appending, `grep -nE "^\s*(from|import)" $NEW/scripts/viz_helpers_2d.py` and confirm no other imports are needed by the two functions (add `matplotlib`/`cv2` usages are covered). Run `ast.parse`.

- [ ] **Step 2: Copy the four scripts**

```bash
cd /home/labeng/yanghan/code/vision/DenseTrack3Dv2
cp scripts/train_intent.py scripts/compute_flow_stats.py scripts/analyze_intent_ckpt.py scripts/viz_intent_predictions.py \
   /home/labeng/yanghan/code/intent-model/scripts/
```

- [ ] **Step 3: Rewrite imports in `train_intent.py`**

In `$NEW/scripts/train_intent.py`:
- `from data.flow_window_dataset import FlowWindowDataset` → unchanged.
- `from densetrack3d.models.worldmodel import IntentModel, IntentModelConfig, intent_loss` → `from intent_model import IntentModel, IntentModelConfig, intent_loss`
- `from densetrack3d.models.worldmodel.types import FlowItem, IntentBatch, IntentOutput` → `from intent_model.types import FlowItem, IntentBatch, IntentOutput`
- `sys.path.insert(...)` line stays.

- [ ] **Step 4: Rewrite imports in `compute_flow_stats.py` and `analyze_intent_ckpt.py`**

`compute_flow_stats.py`:
- `from data.flow_window_dataset import FlowWindowDataset` → unchanged.

`analyze_intent_ckpt.py`:
- `from densetrack3d.models.worldmodel import IntentModel, IntentModelConfig, intent_loss` → `from intent_model import IntentModel, IntentModelConfig, intent_loss`
- `from scripts.train_intent import FlowWindowDataModule, collate, _cfg_from_dict` → unchanged.

- [ ] **Step 5: Rewrite imports in `viz_intent_predictions.py`**

In `$NEW/scripts/viz_intent_predictions.py`:
- `from data.flow_window_dataset import FlowWindowDataset` → unchanged.
- `from densetrack3d.models.worldmodel import IntentModel` → `from intent_model import IntentModel`
- `from densetrack3d.models.worldmodel.types import FlowItem` → `from intent_model.types import FlowItem`
- `from preprocess.track_windowed import render_2d_overlay, rainbow_colors_by_position` → `from scripts.viz_helpers_2d import render_2d_overlay, rainbow_colors_by_position`
- `from scripts.train_intent import FlowWindowDataModule, collate, load_ema_model` → unchanged.

- [ ] **Step 6: Move and fix `sweep_intent.sh`**

```bash
cp /home/labeng/yanghan/code/vision/DenseTrack3Dv2/scripts/sweep_intent.sh \
   /home/labeng/yanghan/code/intent-model/scripts/
```
In `$NEW/scripts/sweep_intent.sh` edit the two path constants at the top:
- `cd /home/labeng/yanghan/code/vision/DenseTrack3Dv2` → `cd /home/labeng/yanghan/code/intent-model`
- `PY=/home/labeng/miniconda3/envs/densetrack3d/bin/python` → `PY=/home/labeng/miniconda3/envs/intent/bin/python`
- Leave `export PYTHONPATH=.` and the `logdirs/intent/...` relative paths unchanged.

- [ ] **Step 7: Verify all scripts parse and are import-clean**

```bash
cd /home/labeng/yanghan/code/intent-model
for f in scripts/*.py; do /usr/bin/python3.12 -c "import ast; ast.parse(open('$f').read())" || echo "PARSE FAIL $f"; done
echo "ast ok"
bash -n scripts/sweep_intent.sh && echo "sweep syntax ok"
grep -rnE "densetrack3d|worldmodel|preprocess\.|track_windowed" scripts/*.py && echo "STALE FOUND" || echo "clean"
```
Expected: `ast ok`, `sweep syntax ok`, `clean`.

---

### Task 7: Move configs + docs

**Files:**
- Create: `$NEW/configs/intent.yaml` (from `$SRC/configs/intent.yaml`, path edits)
- Create: `$NEW/docs/specs/{2026-07-17-object-flow-intent-model-design,2026-07-20-object-flow-model-architecture-detail,2026-07-22-hand-joint-jitter-augmentation-design}.md`
- Create: `$NEW/docs/2026-07-22-hand-joint-jitter-augmentation.md`, `$NEW/docs/intent_experiments.md`, `$NEW/docs/intent_experiments_plan.md`

**Interfaces:**
- Produces: `configs/intent.yaml` usable by `train_intent.py` in the new repo.

- [ ] **Step 1: Copy the config and docs**

```bash
cd /home/labeng/yanghan/code/vision/DenseTrack3Dv2
cp configs/intent.yaml /home/labeng/yanghan/code/intent-model/configs/
cp docs/superpowers/specs/2026-07-17-object-flow-intent-model-design.md \
   docs/superpowers/specs/2026-07-20-object-flow-model-architecture-detail.md \
   docs/superpowers/specs/2026-07-22-hand-joint-jitter-augmentation-design.md \
   /home/labeng/yanghan/code/intent-model/docs/specs/
cp docs/superpowers/plans/2026-07-22-hand-joint-jitter-augmentation.md \
   docs/intent_experiments.md docs/intent_experiments_plan.md \
   /home/labeng/yanghan/code/intent-model/docs/
```

- [ ] **Step 2: Fix paths inside `intent.yaml`**

In `$NEW/configs/intent.yaml`, verify/adjust these keys (the data_root is an absolute external path and stays; the stats path is repo-relative and stays as `data/flow_stats.npz`):
- `data.data_root: /home/labeng/yanghan/data/inhand_manipulation` → unchanged (external data root).
- `data.stats: data/flow_stats.npz` → unchanged (now resolves to `$NEW/data/flow_stats.npz`).
- `trainer.default_root_dir: logdirs/intent` and the `ModelCheckpoint dirpath: logdirs/intent/ckpts` → unchanged (repo-relative; `$NEW/logdirs` exists and is gitignored).
- No `densetrack3d` paths exist in the YAML — confirm with `grep -n densetrack /home/labeng/yanghan/code/intent-model/configs/intent.yaml` returning nothing.

- [ ] **Step 3: Verify**

```bash
cd /home/labeng/yanghan/code/intent-model
/usr/bin/python3.12 -c "import yaml; yaml.safe_load(open('configs/intent.yaml')); print('yaml ok')"
ls docs/specs/ docs/*.md
grep -n densetrack configs/intent.yaml && echo "PATH LEAK" || echo "clean"
```
Expected: `yaml ok`, the 3 specs + 3 docs present, `clean`.

---

### Task 8: Create the py3.12 env and verify rclpy + torch coexist

**Files:**
- None (environment + smoke tests). Uses `$NEW/requirements.txt` from Task 1.

**Interfaces:**
- Produces: conda env `intent` (py3.12) in which the full runtime import graph resolves.

- [ ] **Step 1: Create the env and install deps**

```bash
/home/labeng/miniconda3/bin/conda create -y -n intent python=3.12
/home/labeng/miniconda3/envs/intent/bin/pip install -r /home/labeng/yanghan/code/intent-model/requirements.txt
```
Expected: torch 2.5.1 (cp312 wheel) installs without error.

- [ ] **Step 2: Verify rclpy + torch import together under py3.12**

```bash
source /opt/ros/jazzy/setup.bash
/home/labeng/miniconda3/envs/intent/bin/python -c "import torch; print('torch', torch.__version__)"
# rclpy comes from Jazzy's site-packages; add it to the path for this check:
PYTHONPATH="/opt/ros/jazzy/lib/python3.12/site-packages:$PYTHONPATH" \
  /home/labeng/miniconda3/envs/intent/bin/python -c "import rclpy, torch; print('rclpy+torch coexist ok')"
```
Expected: `torch 2.5.1...` and `rclpy+torch coexist ok`. If `rclpy` is not importable this way, document the exact `source`/`PYTHONPATH` incantation in the README (Task 1 Step 5) — the goal is proving they can share one interpreter, not a permanent path fix.

- [ ] **Step 3: Full runtime import of the intent package (no densetrack3d on path)**

```bash
cd /home/labeng/yanghan/code/intent-model
PYTHONPATH=. /home/labeng/miniconda3/envs/intent/bin/python -c "
from intent_model import IntentModel, IntentModelConfig, intent_loss
from intent_model.types import FlowItem, IntentBatch, IntentOutput
from data.flow_window_dataset import FlowWindowDataset
m = IntentModel(IntentModelConfig())
print('full import + model construct ok; params:', sum(p.numel() for p in m.parameters()))
"
```
Expected: prints a param count (~34.7M) with no ImportError.

---

### Task 9: End-to-end verification, checkpoint move (gated), and source cleanup

**Files:**
- Create (gated): `$NEW/logdirs/intent/ckpts/` baseline `.ckpt` (copied after the sweep baseline finishes)
- Delete (final, gated on all checks passing): the moved intent files from `$SRC` (see list)

**Interfaces:**
- Produces: a verified standalone new repo; a trimmed `DenseTrack3Dv2`.

- [ ] **Step 1: Hard gate — no cross-repo imports anywhere in the new repo**

```bash
cd /home/labeng/yanghan/code/intent-model
grep -rnE "densetrack3d|worldmodel|from preprocess|import preprocess|track_windowed" \
  intent_model/ data/ scripts/ configs/ | grep -v "Vendored\|VENDORED\|CANONICAL\|copied from\|split from" \
  && echo "STALE IMPORTS — STOP" || echo "GATE PASS: no cross-repo imports"
```
Expected: `GATE PASS`. (The grep excludes provenance comments.)

- [ ] **Step 2: Behavior-parity test (gated on baseline checkpoint availability)**

The sweep baseline run (`w_goal=0 w_vel=0`) writes to `$SRC/logdirs/intent/ckpts_sweep_g0.0_v0.0/`. Once it exists, copy the best checkpoint into the new repo and compare `predict_trajectory` output between the two repos on one deterministic batch:

```bash
# copy baseline ckpt (best epoch file) into the new repo
mkdir -p /home/labeng/yanghan/code/intent-model/logdirs/intent/ckpts
cp /home/labeng/yanghan/code/vision/DenseTrack3Dv2/logdirs/intent/ckpts_sweep_g0.0_v0.0/epoch*.ckpt \
   /home/labeng/yanghan/code/intent-model/logdirs/intent/ckpts/
```
Then run a parity script that, in EACH repo (old with densetrack env, new with intent env), loads the same checkpoint via `load_ema_model`, builds one fixed batch (seed 0) from `FlowWindowDataset`, runs `model.predict_trajectory`, and prints the summed abs value of `x_pred`. The two numbers must match to <1e-5. If they differ, the vendored primitives diverged — investigate before deleting anything.

- [ ] **Step 3: DenseTrack3Dv2 regression check**

```bash
cd /home/labeng/yanghan/code/vision/DenseTrack3Dv2
# gen_flow_labels.py (stays) and viz_hand_cloud_live.py (stays) must still parse/import
/usr/bin/python3.12 -c "import ast; ast.parse(open('scripts/gen_flow_labels.py').read()); print('gen_flow_labels ok')"
/usr/bin/python3.12 -c "import ast; ast.parse(open('preprocess/viz_hand_cloud_live.py').read()); print('viz_hand_cloud_live ok')"
```
Expected: both `ok`. These are the only data-side files touching the split boundary.

- [ ] **Step 4: Stage the new repo (do NOT commit)**

```bash
cd /home/labeng/yanghan/code/intent-model
git add -A
git status
```
Expected: all new files staged. STOP — do not commit; the user must approve first (Global Constraint).

- [ ] **Step 5: Delete moved files from `$SRC` (gated on Steps 1–3 passing + user approval)**

Only after Steps 1–3 pass AND the user approves the deletion, remove the now-migrated intent files from `DenseTrack3Dv2`:

```bash
cd /home/labeng/yanghan/code/vision/DenseTrack3Dv2
rm -r densetrack3d/models/worldmodel/
rm data/flow_window_dataset.py data/viz_flow_window_item.py
rm scripts/train_intent.py scripts/compute_flow_stats.py scripts/analyze_intent_ckpt.py scripts/viz_intent_predictions.py scripts/sweep_intent.sh
rm docs/superpowers/specs/2026-07-17-object-flow-intent-model-design.md \
   docs/superpowers/specs/2026-07-20-object-flow-model-architecture-detail.md \
   docs/superpowers/specs/2026-07-22-hand-joint-jitter-augmentation-design.md \
   docs/superpowers/plans/2026-07-22-hand-joint-jitter-augmentation.md \
   docs/intent_experiments.md docs/intent_experiments_plan.md
# KEEP: scripts/gen_flow_labels.py, preprocess/hand_frame_transforms.py (vendored),
#       preprocess/track_windowed.py, all other preprocess/*, configs/intent.yaml? -> intent.yaml MOVES, so:
rm configs/intent.yaml
# data/flow_stats.npz was COPIED (regenerable); safe to keep or remove — leave it (harmless).
```
NOTE: `densetrack3d/models/{loss.py, embeddings.py, model_utils.py, densetrack3d/blocks.py}` STAY — DenseTrack itself uses them; they were copied (not moved).

- [ ] **Step 6: Final confirmation**

```bash
cd /home/labeng/yanghan/code/vision/DenseTrack3Dv2
grep -rlnE "worldmodel|flow_window_dataset|train_intent" --include=*.py . | grep -v __pycache__ \
  && echo "residual intent refs (check they are data-side only)" || echo "no residual intent refs"
```
Expected: only `scripts/gen_flow_labels.py` may reference flow labels (it produces them) — no import of the moved model code.

---

## Notes for the executor

- The sweep baseline run is training in `$SRC` concurrently; it writes only to `$SRC/logdirs/`. It does not touch any file this plan moves. Task 9 Step 2 depends on its output but nothing else does — Tasks 1–8 can run while it trains.
- If the `intent` conda env creation is slow or blocked, Tasks 1–7 are fully verifiable with `/usr/bin/python3.12` (ast/yaml/grep checks); only Task 8 and Task 9 Step 2 need torch.
- Every "move" is copy-then-verify; deletion from `$SRC` happens only in Task 9 Step 5, gated on user approval per the standing no-destructive-change rule.
