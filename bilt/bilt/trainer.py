# BILT (Because I Like Twice) - A PyTorch-based object detection library
# Copyright (C) 2026 Rikiza89
# Licensed under the Apache License, Version 2.0

"""
BILT training engine.

Handles the full training loop:
  - Head warm-up with frozen backbone (epochs 0–warmup_epochs), allowing
    the detection head to stabilise before the backbone starts learning.
  - Optional linear LR warm-up ramp (epochs 0–lr_warmup_epochs): learning
    rate starts at 10% of the target value and ramps linearly to full LR.
  - Backbone unfreeze (epoch warmup_epochs+) — full end-to-end training.
  - AdamW optimiser + cosine LR annealing.
  - Gradient clipping.
  - Optional Exponential Moving Average (EMA) of model weights: produces a
    smoother parameter trajectory and typically improves generalisation.
    EMA weights are used when saving the best checkpoint.
  - Best-checkpoint saving (lowest validation loss).
"""

import copy
import math
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from .core import DetectionModel, get_lr_scheduler, get_optimizer, get_optimizer_differential
from .dataset import create_dataloader
from .loss import BILTLoss
from .utils import get_logger
from .variants import DEFAULT_VARIANT

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Exponential Moving Average helper
# ---------------------------------------------------------------------------

class ModelEMA:
    """
    Exponential Moving Average of model weights.

    After each optimiser step call ``update()``.  The smoothed weights are
    stored in ``self.shadow`` and are applied to the model temporarily when
    the best checkpoint is saved, then immediately restored.

    Parameters
    ----------
    model : nn.Module   The model whose parameters are tracked.
    decay : float       EMA decay (0.999 – 0.9999 typical; higher = smoother).
    """

    def __init__(self, model: nn.Module, decay: float = 0.99):
        self.decay = decay
        self.steps = 0          # used to compute adaptive warm-up decay
        # Track ALL named parameters (including frozen backbone params) AND
        # floating-point buffers (BatchNorm running_mean / running_var) so
        # that the saved EMA checkpoint is fully consistent: weights AND BN
        # statistics all come from the same smoothed trajectory.
        self.shadow: Dict[str, torch.Tensor] = {}
        for name, param in model.named_parameters():
            self.shadow[name] = param.data.clone().float().cpu()
        for name, buf in model.named_buffers():
            if buf.is_floating_point():
                self.shadow[name] = buf.data.clone().float().cpu()

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        """Blend current model weights and buffers into the shadow copy.

        Adaptive decay: starts near 0 and ramps toward self.decay so that
        the shadow converges in tens of steps regardless of the chosen max
        decay.  Formula matches YOLOv5's EMA warm-up strategy:
            d = min(decay, (1 + steps) / (10 + steps))
        """
        self.steps += 1
        d = min(self.decay, (1.0 + self.steps) / (10.0 + self.steps))
        # Trainable parameters
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.shadow[name] = (
                    d * self.shadow[name]
                    + (1.0 - d) * param.data.float().cpu()
                )
        # Floating-point buffers (BN running_mean, running_var, …)
        for name, buf in model.named_buffers():
            if buf.is_floating_point() and name in self.shadow:
                self.shadow[name] = (
                    d * self.shadow[name]
                    + (1.0 - d) * buf.data.float().cpu()
                )

    def apply_to(self, model: nn.Module) -> Dict[str, torch.Tensor]:
        """
        Overwrite model parameters and buffers with EMA values.

        Returns the original values so they can be restored afterwards.
        """
        backup: Dict[str, torch.Tensor] = {}
        for name, param in model.named_parameters():
            if name in self.shadow:
                backup[name] = param.data.clone()
                param.data.copy_(self.shadow[name].to(param.dtype).to(param.device))
        for name, buf in model.named_buffers():
            if name in self.shadow:
                backup[name] = buf.data.clone()
                buf.data.copy_(self.shadow[name].to(buf.dtype).to(buf.device))
        return backup

    @staticmethod
    def restore(model: nn.Module, backup: Dict[str, torch.Tensor]) -> None:
        """Restore model parameters and buffers from a backup dict."""
        for name, param in model.named_parameters():
            if name in backup:
                param.data.copy_(backup[name])
        for name, buf in model.named_buffers():
            if name in backup:
                buf.data.copy_(backup[name])


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class Trainer:
    """
    Training engine for BILT detectors.

    Parameters
    ----------
    dataset_path    : Path   Root dataset directory (must contain train/ and val/).
    num_classes     : int    Number of object categories.
    class_names     : list   Human-readable names for the categories.
    batch_size      : int    Images per batch (min 2).
    learning_rate   : float  Initial AdamW learning rate.
    num_epochs      : int    Total training epochs.
    num_workers     : int    DataLoader worker processes (0 = main process).
    input_size      : int    Image resolution; None = variant default.
    device          : torch.device
    variant         : str    BILT model variant (spark/flash/core/pro/max).
    warmup_epochs   : int    Epochs to freeze the backbone (0 = no freeze).
    lr_warmup_epochs: int    Epochs for linear LR ramp-up (0 = no ramp).
                             LR starts at 10 % of learning_rate and ramps to
                             learning_rate linearly over these epochs.
    backbone_lr_mult: float  Backbone LR multiplier after unfreeze.
    weight_decay    : float  AdamW weight decay.
    cos_lr_min      : float  Cosine scheduler minimum LR.
    grad_clip       : float  Max gradient norm.
    focal_alpha     : float  Focal loss alpha.
    focal_gamma     : float  Focal loss gamma.
    box_loss_weight : float  Regression loss weight.
    use_ciou        : bool   CIoU box loss instead of Smooth-L1.
    augment         : bool   Enable training augmentation.
    flip_prob       : float  Horizontal flip probability.
    color_jitter    : tuple  (brightness, contrast, saturation, hue) jitter.
    cache_images    : bool   Pre-load all images into RAM (recommended for
                             small datasets — eliminates disk I/O after ep 1).
    mosaic          : bool   Enable mosaic 4-image augmentation (train only).
    mosaic_prob     : float  Probability of applying mosaic per sample.
    use_ema         : bool   Enable Exponential Moving Average of weights.
    ema_decay       : float  EMA decay factor (higher = slower update).
                             Note: an auto-decay derived from dataset size is
                             used in practice; this value is the upper bound.
    """

    def __init__(
        self,
        dataset_path: Path,
        num_classes: int,
        class_names: list,
        batch_size: int = 4,
        learning_rate: float = 2e-3,
        num_epochs: int = 50,
        num_workers: int = 0,
        input_size: Optional[int] = None,
        device: Optional[torch.device] = None,
        variant: str = DEFAULT_VARIANT,
        # Training loop
        warmup_epochs: int = 3,
        lr_warmup_epochs: int = 0,
        backbone_lr_mult: float = 0.1,
        weight_decay: float = 1e-4,
        cos_lr_min: float = 1e-6,
        grad_clip: float = 5.0,
        # Loss
        focal_alpha: float = 0.25,
        focal_gamma: float = 2.0,
        box_loss_weight: float = 1.0,
        use_ciou: bool = False,
        # Augmentation
        augment: bool = True,
        flip_prob: float = 0.5,
        color_jitter: Optional[Tuple[float, float, float, float]] = (0.4, 0.4, 0.4, 0.1),
        cache_images: bool = False,
        mosaic: bool = False,
        mosaic_prob: float = 0.5,
        # EMA
        use_ema: bool = False,
        ema_decay: float = 0.99,
    ):
        self.dataset_path = Path(dataset_path)
        self.num_classes = num_classes
        self.class_names = class_names
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.num_workers = num_workers
        self.variant = variant
        self.warmup_epochs = warmup_epochs
        self.lr_warmup_epochs = lr_warmup_epochs
        self.backbone_lr_mult = backbone_lr_mult
        self.weight_decay = weight_decay
        self.cos_lr_min = cos_lr_min
        self.grad_clip = grad_clip
        self.use_ema = use_ema
        self.ema_decay = ema_decay

        if device is not None:
            self.device = device
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
        logger.info(f"Trainer using device: {self.device}")

        # Resolve input size from variant default if not provided
        if input_size is None:
            from .variants import get_variant_config
            input_size = get_variant_config(variant)["input_size"]
        self.input_size = input_size

        # ------------------------------------------------------------------ #
        # Data loaders                                                        #
        # ------------------------------------------------------------------ #
        _pin = self.device.type == "cuda"
        logger.info("Building training dataloader …")
        self.train_loader, _ = create_dataloader(
            images_dir=self.dataset_path / "train" / "images",
            labels_dir=self.dataset_path / "train" / "labels",
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=True,
            input_size=self.input_size,
            training=True,
            pin_memory=_pin,
            augment=augment,
            flip_prob=flip_prob,
            color_jitter=color_jitter,
            cache_images=cache_images,
            mosaic=mosaic,
            mosaic_prob=mosaic_prob,
        )

        logger.info("Building validation dataloader …")
        self.val_loader, _ = create_dataloader(
            images_dir=self.dataset_path / "val" / "images",
            labels_dir=self.dataset_path / "val" / "labels",
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=False,
            input_size=self.input_size,
            training=False,
            pin_memory=_pin,
            cache_images=cache_images,   # cache val images too — they're tiny
        )

        # ------------------------------------------------------------------ #
        # Model                                                               #
        # ------------------------------------------------------------------ #
        logger.info(f"Initialising BILT-{variant} for {num_classes} classes …")
        self.detection_model = DetectionModel(
            variant=variant,
            num_classes=num_classes,
            class_names=class_names,
            use_ciou=use_ciou,
        )
        self.detection_model.to(self.device)

        # Override the loss criterion with user-supplied hyperparameters
        self.detection_model.model.criterion = BILTLoss(
            num_classes,
            alpha=focal_alpha,
            gamma=focal_gamma,
            box_weight=box_loss_weight,
            use_ciou=use_ciou,
        )

        # ------------------------------------------------------------------ #
        # EMA                                                                 #
        # ------------------------------------------------------------------ #
        self.ema: Optional[ModelEMA] = None
        if use_ema:
            # Auto-compute a sensible decay from dataset size and batch size.
            # Target: EMA window ≈ 2 epochs → decay = 1 - 1/(2*steps_per_epoch).
            # Clamped to [0.90, 0.9999] so it works for tiny and large datasets.
            steps_per_epoch = max(1, len(self.train_loader.dataset) // batch_size)
            auto_decay = max(0.90, min(0.9999, 1.0 - 1.0 / (2 * steps_per_epoch)))
            if abs(auto_decay - ema_decay) > 0.001:
                logger.info(
                    f"EMA: auto-decay {auto_decay:.4f} used instead of "
                    f"user-specified {ema_decay:.4f} "
                    f"(auto-tuned for dataset={len(self.train_loader.dataset)} "
                    f"batch={batch_size} steps/epoch={steps_per_epoch})"
                )
            else:
                logger.info(
                    f"EMA auto-decay: dataset={len(self.train_loader.dataset)} "
                    f"batch={batch_size} steps/epoch={steps_per_epoch} "
                    f"→ decay={auto_decay:.4f}"
                )
            self.ema = ModelEMA(self.detection_model.model, decay=auto_decay)
            self.ema_decay = auto_decay   # store actual decay used

        # ------------------------------------------------------------------ #
        # Optimiser and LR scheduler                                         #
        # ------------------------------------------------------------------ #
        if warmup_epochs > 0:
            self.optimizer = get_optimizer(
                self.detection_model.model, learning_rate, weight_decay
            )
        else:
            self.optimizer = get_optimizer_differential(
                self.detection_model.model, learning_rate, backbone_lr_mult, weight_decay
            )
        self.scheduler = self._build_scheduler(self.optimizer, num_epochs, lr_warmup_epochs)

        # Training state
        self.current_epoch = 0
        self.training_losses: list = []
        self.validation_losses: list = []

        logger.info("Trainer ready.")

    # ---------------------------------------------------------------------- #
    # Scheduler builder                                                       #
    # ---------------------------------------------------------------------- #

    def _build_scheduler(
        self,
        optimizer: torch.optim.Optimizer,
        total_epochs: int,
        lr_warmup_epochs: int,
    ) -> torch.optim.lr_scheduler.LRScheduler:
        """
        Build a LambdaLR combining linear warmup then cosine annealing.

        Using a single LambdaLR avoids the double-step initialisation issue
        that arises when LinearLR and CosineAnnealingLR are chained via
        SequentialLR — each sub-scheduler calls step() during __init__,
        causing the LR to be overwritten and the warmup to start at full
        learning rate instead of the intended 10%.

        If lr_warmup_epochs == 0 uses pure CosineAnnealingLR (original
        behaviour).
        """
        if lr_warmup_epochs <= 0:
            return get_lr_scheduler(optimizer, total_epochs, self.cos_lr_min)

        cos_epochs   = max(1, total_epochs - lr_warmup_epochs)
        cos_lr_min   = self.cos_lr_min
        # Capture per-group base LRs before any scheduler modifies them
        base_lrs: List[float] = [g['lr'] for g in optimizer.param_groups]

        def _make_lambda(base_lr: float):
            min_factor = (cos_lr_min / base_lr) if base_lr > 0 else 0.0
            def fn(epoch: int) -> float:
                epoch = max(0, epoch)          # guard against -1 on some versions
                if epoch < lr_warmup_epochs:
                    # Linear ramp: 10 % → 100 % over lr_warmup_epochs epochs
                    return 0.1 + 0.9 * epoch / max(1, lr_warmup_epochs)
                # Cosine annealing for remaining epochs
                t = epoch - lr_warmup_epochs
                cosine_val = 0.5 * (1.0 + math.cos(math.pi * t / cos_epochs))
                return max(min_factor, cosine_val)
            return fn

        lambdas = [_make_lambda(b) for b in base_lrs]
        sched   = torch.optim.lr_scheduler.LambdaLR(optimizer, lambdas)

        # Explicitly apply the warmup start factor so that epoch 0 trains at
        # 10 % LR regardless of whether this PyTorch version calls step()
        # inside __init__ or not.
        for group, base_lr in zip(optimizer.param_groups, base_lrs):
            group['lr'] = base_lr * 0.1

        return sched

    # ---------------------------------------------------------------------- #

    def train_one_epoch(self) -> float:
        """Run one forward-backward pass over the training set."""
        self.detection_model.train()
        epoch_loss = 0.0
        num_batches = 0

        _nb = self.device.type == "cuda"
        for batch_idx, (images, targets) in enumerate(self.train_loader):
            images = images.to(self.device, non_blocking=_nb)
            targets = [
                {k: v.to(self.device, non_blocking=_nb) for k, v in t.items()}
                for t in targets
            ]

            loss_dict = self.detection_model(images, targets)
            loss = loss_dict["total"]

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.detection_model.model.parameters(), max_norm=self.grad_clip
            )
            self.optimizer.step()

            # Update EMA shadow weights after each optimiser step
            if self.ema is not None:
                self.ema.update(self.detection_model.model)

            epoch_loss += loss.item()
            num_batches += 1

            if batch_idx % 10 == 0:
                cls_l = loss_dict.get("cls", torch.tensor(0.0)).item()
                box_l = loss_dict.get("box", torch.tensor(0.0)).item()
                logger.info(
                    f"Epoch {self.current_epoch + 1}/{self.num_epochs}  "
                    f"batch {batch_idx}/{len(self.train_loader)}  "
                    f"loss={loss.item():.4f}  "
                    f"cls={cls_l:.4f}  box={box_l:.4f}"
                )

        return epoch_loss / max(num_batches, 1)

    # ---------------------------------------------------------------------- #

    def validate(self) -> float:
        """Compute validation loss without updating model parameters.

        The model is kept in train mode so that BILTDetector.forward() takes
        the loss-computation branch (which requires self.training == True).
        However, all BatchNorm2d layers are switched to eval mode before the
        loop so that their running_mean / running_var are NOT updated by the
        validation batches.

        Without this fix, torch.no_grad() disables gradient computation but
        does NOT stop BatchNorm from updating its running statistics.  On
        ResNet-50/101 (which has ~53/104 BN layers), each validation pass
        overwrites the carefully adapted training-data statistics with noisy
        validation-batch statistics.  The saved best-checkpoint therefore
        contains corrupted BN stats, producing degraded feature maps at
        inference time (eval mode uses running stats) and near-zero
        confidence scores — i.e. no detections.  MobileNet variants have
        only ~15-17 BN layers and are far less sensitive to this effect.
        """
        self.detection_model.train()   # keep training mode for loss branch

        # Freeze all BN running stats for the duration of the validation loop.
        for module in self.detection_model.model.modules():
            if isinstance(module, nn.BatchNorm2d):
                module.eval()

        epoch_loss = 0.0
        num_batches = 0

        _nb = self.device.type == "cuda"
        with torch.no_grad():
            for images, targets in self.val_loader:
                images = images.to(self.device, non_blocking=_nb)
                targets = [
                    {k: v.to(self.device, non_blocking=_nb) for k, v in t.items()}
                    for t in targets
                ]
                loss_dict = self.detection_model(images, targets)
                epoch_loss += loss_dict["total"].item()
                num_batches += 1

        # Restore all modules to train mode for the next training epoch.
        self.detection_model.train()

        return epoch_loss / max(num_batches, 1)

    # ---------------------------------------------------------------------- #

    def _save_best(self, save_path: Path, class_id_mapping: dict) -> None:
        """
        Save the best checkpoint.

        When EMA is enabled, EMA weights are written to disk and then the
        original weights are immediately restored so training continues
        unaffected.
        """
        if self.ema is not None:
            backup = self.ema.apply_to(self.detection_model.model)
            self.detection_model.save(save_path, self.class_names, class_id_mapping)
            ModelEMA.restore(self.detection_model.model, backup)
        else:
            self.detection_model.save(save_path, self.class_names, class_id_mapping)

    # ---------------------------------------------------------------------- #

    def train(
        self,
        save_path: Path,
        last_save_path: Optional[Path] = None,
        callback: Optional[Callable[[Dict[str, Any]], None]] = None,
    ) -> Dict[str, Any]:
        """
        Run the complete training loop.

        Parameters
        ----------
        save_path : Path  Where to write the best checkpoint.
        callback  : optional callable invoked at the end of each epoch with
                    a metrics dict.

        Returns
        -------
        dict with final training metrics.
        """
        logger.info(
            f"Training BILT-{self.variant}  "
            f"epochs={self.num_epochs}  "
            f"input={self.input_size}px  "
            f"device={self.device}  "
            f"ema={'on' if self.ema else 'off'}  "
            f"ciou={'on' if self.detection_model.model.criterion.use_ciou else 'off'}  "
            f"lr_warmup={self.lr_warmup_epochs}ep"
        )
        t0 = time.time()
        best_val_loss = float("inf")

        for epoch in range(self.num_epochs):
            self.current_epoch = epoch

            # ── Backbone freeze / unfreeze ─────────────────────────────────
            if epoch == 0 and self.warmup_epochs > 0:
                logger.info(
                    f"Warming up: backbone frozen for first {self.warmup_epochs} epochs."
                )
                self.detection_model.model.backbone.freeze()

            if self.warmup_epochs > 0 and epoch == self.warmup_epochs:
                logger.info(
                    "Unfreezing backbone with differential LR "
                    f"(backbone={self.learning_rate * self.backbone_lr_mult:.1e}, "
                    f"head={self.learning_rate:.1e})."
                )
                self.detection_model.model.backbone.unfreeze()
                self.optimizer = get_optimizer_differential(
                    self.detection_model.model,
                    self.learning_rate,
                    self.backbone_lr_mult,
                    self.weight_decay,
                )
                # Rebuild scheduler for remaining epochs (LR warmup is done
                # by this point if lr_warmup_epochs <= warmup_epochs).
                remaining = self.num_epochs - epoch
                remaining_warmup = max(0, self.lr_warmup_epochs - epoch)
                self.scheduler = self._build_scheduler(
                    self.optimizer, remaining, remaining_warmup
                )

            # ── Train / validate ───────────────────────────────────────────
            train_loss = self.train_one_epoch()
            self.training_losses.append(train_loss)

            val_loss = self.validate()
            self.validation_losses.append(val_loss)

            self.scheduler.step()
            current_lr = self.optimizer.param_groups[0]["lr"]

            logger.info(
                f"Epoch {epoch + 1}/{self.num_epochs}  "
                f"train_loss={train_loss:.4f}  "
                f"val_loss={val_loss:.4f}  "
                f"lr={current_lr:.2e}"
            )

            if callback:
                should_stop = callback({
                    "epoch":        epoch + 1,
                    "total_epochs": self.num_epochs,
                    "train_loss":   train_loss,
                    "val_loss":     val_loss,
                    "lr":           current_lr,
                })
                if should_stop:
                    break

            # ── Save best checkpoint ───────────────────────────────────────
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                class_id_mapping = {
                    "class_id_to_idx": getattr(
                        self.train_loader.dataset, "class_id_to_idx", None
                    ),
                    "idx_to_class_id": getattr(
                        self.train_loader.dataset, "idx_to_class_id", None
                    ),
                }
                self._save_best(save_path, class_id_mapping)
                ema_note = " (EMA weights)" if self.ema else ""
                logger.info(
                    f"  ↳ New best model saved{ema_note} (val_loss={val_loss:.4f})"
                )

        elapsed = time.time() - t0
        logger.info(f"Training finished in {elapsed:.1f}s")

        # Save the final model state as last.pt (regardless of val loss)
        if last_save_path is not None:
            class_id_mapping = {
                "class_id_to_idx": getattr(
                    self.train_loader.dataset, "class_id_to_idx", None
                ),
                "idx_to_class_id": getattr(
                    self.train_loader.dataset, "idx_to_class_id", None
                ),
            }
            self.detection_model.save(
                last_save_path, self.class_names, class_id_mapping
            )
            logger.info(f"Last checkpoint saved → {last_save_path}")

        return {
            "variant":            self.variant,
            "num_epochs":         self.num_epochs,
            "final_train_loss":   self.training_losses[-1],
            "final_val_loss":     self.validation_losses[-1],
            "best_val_loss":      best_val_loss,
            "training_time":      elapsed,
            "model_path":         str(save_path),
            "training_losses":    list(self.training_losses),
            "validation_losses":  list(self.validation_losses),
            "num_train":          len(self.train_loader.dataset),
            "num_val":            len(self.val_loader.dataset),
        }
