#!/usr/bin/env python3
"""Train the object-flow intent model with PyTorch Lightning (spec §11 gate 3 + §13).

    x_{n, t:t+L_pred} = f( P_t, q_{t-T_hist:t}, q_{t:t+L_pred} )

Everything is driven from a YAML config via LightningCLI:

  /home/labeng/miniconda3/envs/densetrack3d/bin/python scripts/train_intent.py fit \
      --config configs/intent.yaml
  # gate 3 (overfit one clip):  ... fit --config configs/intent.yaml --data.overfit_one_clip true \
  #                                 --data.clip <episode_or_clip_dir>

Pieces this ties together (all already built):
  - data   : data/flow_window_dataset.FlowWindowDataset  (+ episode-level split)
  - stats  : scripts/compute_flow_stats.py -> data/flow_stats.npz  (per-channel norm)
  - model  : densetrack3d.models.worldmodel.IntentModel / intent_loss / predict_trajectory

Optimization (spec §6.5): AdamW, cosine 1e-4 -> 1e-6, 5 warmup epochs, wd 0.01,
grad-clip 1.0 (trainer.gradient_clip_val), EMA 0.999 for eval (EMACallback).
"""
import copy
import math
import os
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader

import lightning.pytorch as pl
from lightning.pytorch.cli import LightningCLI

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.flow_window_dataset import FlowWindowDataset
from densetrack3d.models.worldmodel import IntentModel, IntentModelConfig, intent_loss
from densetrack3d.models.worldmodel.types import FlowItem, IntentBatch, IntentOutput


# --------------------------------------------------------------------------- #
# Batch assembly
# --------------------------------------------------------------------------- #
# The Dataset emits numpy dicts. Stack the array fields into a torch batch
_FLOAT_KEYS = ("cloud", "x0", "target", "q_hist", "q_future", "K", "dxyz_mean", "dxyz_std")


def _compute_d_q(articulation: str, use_wrist: bool, wrist_repr: str) -> int:
    """Hand-feature width the encoder must accept -- mirrors Dataset._hand_features.

    articulation dim (ergonomics 20 / raw_node_pose or camera_node_pose 24 keypoints x3 = 72)
    + wrist rotation (none, 6D=6, or matrix=9). Linked into model.model_cfg.d_q so the two can't disagree.
    """
    art = 20 if articulation == "ergonomics" else 72
    if articulation == "camera_node_pose":
        return art                                                   # wrist implicit in positions
    wr = 0 if not use_wrist else (6 if wrist_repr == "6d" else 9)
    return art + wr


def _cfg_from_dict(d: dict) -> IntentModelConfig:
    """Build an IntentModelConfig from a saved dict, dropping keys the dataclass no longer has.

    Old checkpoints (pre point_out removal) carry a `point_out` key; feeding it to
    IntentModelConfig(**d) would raise. Filter to current fields so old ckpts still load
    (the scene-token width is now always C, so a stored point_out is safely ignorable)."""
    import dataclasses
    known = {f.name for f in dataclasses.fields(IntentModelConfig)}
    dropped = set(d) - known
    if dropped:
        print(f"[_cfg_from_dict] ignoring stale config keys from checkpoint: {sorted(dropped)}")
    return IntentModelConfig(**{k: v for k, v in d.items() if k in known})


def collate(items: list[FlowItem]) -> IntentBatch:
    batch = {}
    for k in _FLOAT_KEYS:
        if k in items[0]:
            batch[k] = torch.from_numpy(np.stack([it[k] for it in items])).float()
    batch["target_vis"] = torch.from_numpy(np.stack([it["target_vis"] for it in items]))   # bool
    batch["frame_meta"] = [it["frame_meta"] for it in items]
    return batch


# --------------------------------------------------------------------------- #
# LR schedule: linear warmup -> cosine decay to lr_min (spec §6.5)
# --------------------------------------------------------------------------- #
def build_scheduler(
        optimizer: torch.optim.Optimizer, warmup_steps: int, total_steps: int,
        lr_min_ratio: float
        ) -> torch.optim.lr_scheduler.LambdaLR:
    """LambdaLR: linear 0->1 over warmup_steps, then cosine 1->lr_min_ratio."""
    def fn(step: int) -> float:
        if step < warmup_steps:
            return (step + 1) / max(1, warmup_steps)
        prog = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        cos = 0.5 * (1 + math.cos(math.pi * min(prog, 1.0)))
        return lr_min_ratio + (1 - lr_min_ratio) * cos
    return torch.optim.lr_scheduler.LambdaLR(optimizer, fn)


# --------------------------------------------------------------------------- #
# EMA callback  (spec §13 -- the hazard that MUST survive into code)
# --------------------------------------------------------------------------- #
class EMACallback(pl.Callback):
    """EMA of model weights (decay 0.999); swapped in for validation, restored after.

    SPEC §13 HAZARD -- DO NOT "SIMPLIFY" TO PARAMETERS-ONLY. SetAbstraction uses
    nn.BatchNorm2d, whose running_mean/running_var/num_batches_tracked are BUFFERS,
    not parameters (27 such buffers in the default model). A params-only EMA leaves
    the shadow's BN running stats at random init -> eval silently produces garbage
    while training metrics look fine. So: EMA the params, and copy BUFFERS VERBATIM
    from the live model each update (running stats are already a moving average).
    """

    def __init__(self, decay: float = 0.999):
        super().__init__()
        self.decay = decay
        self.shadow = None        # nn.Module mirror of pl_module.model
        self._backup = None       # live state_dict stashed during validation
        self._loaded = None       # shadow state_dict restored from a checkpoint

    def on_fit_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        self.shadow = copy.deepcopy(pl_module.model).eval()
        for p in self.shadow.parameters():
            p.requires_grad_(False)
        if self._loaded is not None:                     # resumed run
            self.shadow.load_state_dict(self._loaded)
            self._loaded = None

    @torch.no_grad()
    def on_train_batch_end(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule, *args, **kwargs
        ) -> None:
        d = self.decay
        for s, p in zip(self.shadow.parameters(), pl_module.model.parameters()):
            s.mul_(d).add_(p, alpha=1 - d)
        for s, b in zip(self.shadow.buffers(), pl_module.model.buffers()):
            s.copy_(b)                                   # verbatim: running stats already averaged

    @torch.no_grad()
    def on_train_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        # Spec §13 guard WITH TEETH: the copy loop above must keep the shadow's BN running
        # stats in lockstep with the live model. Assert that invariant independently -- if a
        # future edit drops the buffer-copy loop, the live BN stats move off init while the
        # shadow's stay frozen, so this diverges and fires (a mere count-equal assert on a
        # deepcopy shadow is tautological and would NOT catch that regression).
        for s, b in zip(self.shadow.buffers(), pl_module.model.buffers()):
            if not torch.equal(s, b):
                raise AssertionError(
                    "EMA shadow buffers diverged from the live model -- the on_train_batch_end "
                    "buffer-copy loop is missing or broken (BN running stats would be lost at eval)")

    def on_validation_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        # Standalone `trainer.validate --ckpt_path` never calls on_fit_start, so build the
        # shadow lazily from the restored callback state -- else validation would silently
        # run on the RAW weights and its ADE would not match training-time validation.
        if self.shadow is None and self._loaded is not None:
            self.shadow = copy.deepcopy(pl_module.model).eval()
            self.shadow.load_state_dict(self._loaded)
            self._loaded = None
        if self.shadow is None:                          # no EMA available (e.g. cold validate)
            return
        self._backup = {k: v.detach().clone() for k, v in pl_module.model.state_dict().items()}
        pl_module.model.load_state_dict(self.shadow.state_dict())

    def on_validation_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        if self._backup is not None:
            pl_module.model.load_state_dict(self._backup)
            self._backup = None

    # persist the shadow in Lightning checkpoints so resume keeps the EMA
    def state_dict(self) -> dict:
        return {"decay": self.decay,
                "shadow": None if self.shadow is None else self.shadow.state_dict()}

    def load_state_dict(self, sd: dict) -> None:
        self.decay = sd.get("decay", self.decay)
        self._loaded = sd.get("shadow")                  # applied in on_fit_start


# --------------------------------------------------------------------------- #
# LightningModule
# --------------------------------------------------------------------------- #
class IntentLitModule(pl.LightningModule):
    """Wraps IntentModel + intent_loss; AdamW + warmup-cosine (spec §6.5)."""

    def __init__(self, model_cfg: IntentModelConfig = IntentModelConfig(),
                 lr: float = 1e-4, lr_min: float = 1e-6, warmup_steps: int = 2000,
                 weight_decay: float = 0.01):
        super().__init__()
        self.save_hyperparameters()
        self.cfg = model_cfg
        self.model = IntentModel(model_cfg)

    def training_step(self, batch: IntentBatch, batch_idx: int) -> torch.Tensor:
        out = self.model(batch)
        total, parts = intent_loss(out, batch, w_vis=self.cfg.w_vis, reg=self.cfg.reg)
        self.log_dict({"train/loss": total, "train/reg": parts["reg"], "train/vis": parts["vis"]},
                      prog_bar=True, on_step=True, on_epoch=False, batch_size=batch["cloud"].shape[0])
        return total

    def validation_step(self, batch: IntentBatch, batch_idx: int) -> torch.Tensor:
        # EMACallback has swapped the EMA weights into self.model for eval.
        out = self.model(batch)
        total, parts = intent_loss(out, batch, w_vis=self.cfg.w_vis, reg=self.cfg.reg)
        ade, fde = self._endpoint_errors(batch, out)
        bs = batch["cloud"].shape[0]
        # ade/fde are per-visible-step means WITHIN a batch; the epoch value is a batch-size-
        # weighted mean-of-means (on_epoch=True), not a global per-step mean. Negligible at
        # fixed batch size with drop_last=False on val -- expect third-decimal wiggle only.
        self.log_dict({"val/loss": total, "val/reg": parts["reg"], "val/vis": parts["vis"],
                       "val/ade_m": ade, "val/fde_m": fde},
                      prog_bar=True, on_step=False, on_epoch=True, batch_size=bs)
        return total

    @torch.no_grad()
    def _endpoint_errors(
        self, batch: IntentBatch, out: IntentOutput
        ) -> tuple[torch.Tensor, torch.Tensor]:
        """Metric-space L2 error over VISIBLE steps: ADE (all steps) + FDE (last step).

        predict_trajectory de-standardizes Delta and composes on metric x0; it may slice
        to cfg.t_pred, so align target/vis to x_pred's horizon. Occluded targets are NaN,
        masked out -- scrub with nan_to_num so NaN*0 can't poison the mean (same as §9)."""
        x_pred, _ = self.model.predict_trajectory(batch, out)      # (B, H, N, 3) metres
        H = x_pred.shape[1]
        target = torch.nan_to_num(batch["target"][:, :H])
        m = batch["target_vis"][:, :H].float()                     # (B, H, N)
        err = torch.linalg.norm(x_pred - target, dim=-1)           # (B, H, N) metres
        ade = (err * m).sum() / m.sum().clamp_min(1.0)
        fde = (err[:, -1] * m[:, -1]).sum() / m[:, -1].sum().clamp_min(1.0)
        return ade, fde

    def configure_optimizers(self) -> dict:
        # Two-group AdamW: exclude biases, norm affines (BN gamma/beta in SetAbstraction),
        # and positional/embedding tables (step_emb) from weight decay -- decaying those
        # toward zero is a conventional mis-default, not what wd is for.
        decay, no_decay = [], []
        for name, p in self.named_parameters():
            if not p.requires_grad:
                continue
            if p.ndim <= 1 or name.endswith("step_emb"):     # biases, norm scales, 1-D tables
                no_decay.append(p)
            else:
                decay.append(p)
        opt = torch.optim.AdamW(
            [{"params": decay, "weight_decay": self.hparams.weight_decay},
             {"params": no_decay, "weight_decay": 0.0}], lr=self.hparams.lr)
        # estimated_stepping_batches already folds in max_epochs, accumulation, and devices.
        # Warmup is step-denominated: with stride_win=1 an epoch is ~9k steps, so epoch-scaled
        # warmup silently balloons (warmup_epochs=5 -> ~45k steps, far past the usual 1-3k).
        total_steps = int(self.trainer.estimated_stepping_batches)
        lr_min_ratio = self.hparams.lr_min / self.hparams.lr
        sched = build_scheduler(opt, self.hparams.warmup_steps, total_steps, lr_min_ratio)
        return {"optimizer": opt,
                "lr_scheduler": {"scheduler": sched, "interval": "step"}}


# --------------------------------------------------------------------------- #
# DataModule
# --------------------------------------------------------------------------- #
class FlowWindowDataModule(pl.LightningDataModule):
    """Episode-level train/eval split over precomputed windows (spec §5.2)."""

    def __init__(self, data_root: str = "/home/labeng/yanghan/data/inhand_manipulation",
                 clip: str = None, stats: str = "data/flow_stats.npz",
                 stride_hz: int = 4, t_pred: int = 8, t_hist: int = 4, pred_pad: int = 0,
                 n_query: int = 16, val_frac: float = 0.15, split_seed: int = 0,
                 batch_size: int = 16, num_workers: int = 4, overfit_one_clip: bool = False,
                 articulation: str = "ergonomics", use_wrist: bool = True,
                 wrist_repr: str = "6d"):
        super().__init__()
        self.save_hyperparameters()
        self.train_ds = None
        self.val_ds = None

    def setup(self, stage: str = None) -> None:
        if self.train_ds is not None:                       # setup() is called per stage (fit, validate)
            return
        h = self.hparams
        # dxyz stats are rate-specific; a stride_hz mismatch silently mis-scales the target.
        assert int(np.load(h.stats)["stride_hz"]) == h.stride_hz, \
            f"stats stride_hz != {h.stride_hz}; recompute flow_stats with --stride-hz {h.stride_hz}"
        source = h.clip or h.data_root
        common = dict(stats=h.stats, stride_hz=h.stride_hz,
                      t_pred=h.t_pred, t_hist=h.t_hist, pred_pad=h.pred_pad, n_query=h.n_query,
                      articulation=h.articulation, use_wrist=h.use_wrist,
                      wrist_repr=h.wrist_repr)
        if h.overfit_one_clip:                              # gate 3: memorize one clip, no held-out
            assert h.clip is not None, \
                "overfit_one_clip=True needs --data.clip <dir>; else it overfits the WHOLE dataset"
            self.train_ds = FlowWindowDataset(source, split="all", **common)
            self.val_ds = self.train_ds
        else:
            split = dict(val_frac=h.val_frac, split_seed=h.split_seed)
            self.train_ds = FlowWindowDataset(source, split="train", **common, **split)
            self.val_ds = FlowWindowDataset(source, split="eval", **common, **split)
        # train loader uses drop_last=True; fewer windows than a batch -> silently empty loader.
        assert len(self.train_ds) >= h.batch_size, \
            f"train set has {len(self.train_ds)} windows < batch_size={h.batch_size} " \
            "(drop_last would empty the loader); lower batch_size or add data"
        print(self.train_ds.coverage_summary())

    def _loader(self, ds: FlowWindowDataset, shuffle: bool) -> DataLoader:
        return DataLoader(ds, batch_size=self.hparams.batch_size, shuffle=shuffle,
                          num_workers=self.hparams.num_workers, collate_fn=collate,
                          persistent_workers=self.hparams.num_workers > 0,
                          pin_memory=True, drop_last=shuffle)

    def train_dataloader(self) -> DataLoader:
        return self._loader(self.train_ds, shuffle=True)

    def val_dataloader(self) -> DataLoader:
        return self._loader(self.val_ds, shuffle=False)


# --------------------------------------------------------------------------- #
# Canonical inference loader
# --------------------------------------------------------------------------- #
def load_ema_model(ckpt_path: str, device: str = "cpu") -> IntentModel:
    """Load the EMA weights that EARNED a checkpoint's val metric -- the inference entry point.

    Lightning saves training state in ckpt["state_dict"] (raw weights) but ranks/names
    checkpoints by val/ade_m, which is measured on the EMA shadow (EMACallback swaps it in
    for validation, then restores the raw weights BEFORE ModelCheckpoint saves). So
    `IntentModel.load_state_dict(ckpt["state_dict"])` reproduces the WRONG number. The EMA
    weights live under ckpt["callbacks"][<EMACallback key>]["shadow"]; this reads them and
    rebuilds the model from the saved hyper_parameters. Falls back to the raw weights (with a
    warning) only if no shadow is stored (e.g. a run without the EMA callback).
    """
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ck["hyper_parameters"]["model_cfg"]
    if not isinstance(cfg, IntentModelConfig):
        cfg = _cfg_from_dict(cfg)                              # LightningCLI stores it as a dict
    model = IntentModel(cfg)
    ema_key = next((k for k in ck.get("callbacks", {}) if "EMACallback" in k), None)
    shadow = ck["callbacks"][ema_key]["shadow"] if ema_key else None
    if shadow is not None:
        model.load_state_dict(shadow)
    else:
        print(f"[load_ema_model] WARNING: no EMA shadow in {ckpt_path}; "
              "loading raw state_dict -- val metric in the filename will NOT reproduce.")
        model.load_state_dict({k[len("model."):]: v for k, v in ck["state_dict"].items()
                               if k.startswith("model.")})
    return model.to(device).eval()


# --------------------------------------------------------------------------- #
# CLI: link data<->model dims so they can't drift, register EMA as a callback
# --------------------------------------------------------------------------- #
class IntentCLI(LightningCLI):
    def add_arguments_to_parser(self, parser) -> None:
        parser.add_lightning_class_args(EMACallback, "ema")     # -> trainer.callbacks
        # the dataset is the source of truth for the window shape; the model follows it,
        # so a config can't set n_query=16 on data and 32 on the model (spec §10 shapes).
        parser.link_arguments("data.n_query", "model.model_cfg.n_query")
        parser.link_arguments("data.t_pred", "model.model_cfg.t_pred")
        parser.link_arguments(("data.t_pred", "data.pred_pad"), "model.model_cfg.l_pred",
                              compute_fn=lambda tp, pp: tp + pp)
        # the hand encoder's input width d_q is fully determined by the data-side hand
        # representation; derive it so a config can't set ergonomics on data and d_q=72 on
        # the model (would silently mis-shape the encoder). Also mirror articulation.
        parser.link_arguments("data.articulation", "model.model_cfg.articulation")
        parser.link_arguments(("data.articulation", "data.use_wrist", "data.wrist_repr"),
                              "model.model_cfg.d_q", compute_fn=_compute_d_q)


def cli_main() -> None:
    IntentCLI(model_class=IntentLitModule, datamodule_class=FlowWindowDataModule,
              save_config_kwargs={"overwrite": True})


if __name__ == "__main__":
    cli_main()
