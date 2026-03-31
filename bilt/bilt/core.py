# BILT (Because I Like Twice) - A PyTorch-based object detection library
# Copyright (C) 2026 Rikiza89
# Licensed under the Apache License, Version 2.0

"""
BILT core detector.

BILTDetector assembles the backbone, FPN neck, detection head, anchor
generator and loss into a single nn.Module with a unified forward() that
returns losses during training and decoded predictions during inference.

DetectionModel wraps BILTDetector to add save / load / optimizer helpers
that the Trainer and high-level BILT class depend on.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn

from .backbone import BILTBackbone
from .neck import FPNNeck
from .head import BILTHead
from .anchors import AnchorGenerator, AnchorMatcher, encode_boxes, decode_boxes
from .loss import BILTLoss
from .variants import get_variant_config, DEFAULT_VARIANT

logger = logging.getLogger(__name__)

# FPN strides for levels P3 → P6
_FPN_STRIDES = [8, 16, 32, 64]


# ---------------------------------------------------------------------------
# BILTDetector – the core nn.Module
# ---------------------------------------------------------------------------

class BILTDetector(nn.Module):
    """
    Full BILT object detection model.

    Architecture
    ------------
    Backbone  →  FPN neck  →  Detection head
    [C3,C4,C5]  [P3,P4,P5,P6]  cls_preds + box_preds per level

    The backbone is determined by the chosen variant:
      spark  → MobileNetV2
      flash  → MobileNetV3-Small
      core   → MobileNetV3-Large
      pro    → ResNet-50
      max    → ResNet-101

    Parameters
    ----------
    variant     : str   Model variant name (spark/flash/core/pro/max).
    num_classes : int   Number of object categories.
    """

    def __init__(
        self,
        variant: str,
        num_classes: int,
        use_ciou: bool = False,
    ):
        super().__init__()

        cfg = get_variant_config(variant)

        self.variant = variant
        self.num_classes = num_classes
        self.input_size = cfg["input_size"]

        # ------------------------------------------------------------------ #
        # Backbone                                                            #
        # ------------------------------------------------------------------ #
        self.backbone = BILTBackbone(cfg["backbone"])

        # ------------------------------------------------------------------ #
        # Neck                                                                #
        # ------------------------------------------------------------------ #
        fpn_ch = cfg["fpn_channels"]
        self.neck = FPNNeck(self.backbone.out_channels, fpn_ch)

        # ------------------------------------------------------------------ #
        # Anchor generator                                                    #
        # ------------------------------------------------------------------ #
        _scales = cfg.get("anchor_scales", (1.0, 1.26, 1.587))
        _ratios = cfg["anchor_aspect_ratios"]
        num_anchors = len(_scales) * len(_ratios)
        self.anchor_gen = AnchorGenerator(
            strides=_FPN_STRIDES,
            anchor_sizes=cfg["anchor_sizes"],
            anchor_scales=_scales,
            aspect_ratios=_ratios,
        )

        # ------------------------------------------------------------------ #
        # Detection head                                                      #
        # ------------------------------------------------------------------ #
        self.head = BILTHead(
            in_channels=fpn_ch,
            num_classes=num_classes,
            num_anchors=num_anchors,
            num_convs=cfg["head_num_convs"],
        )

        # ------------------------------------------------------------------ #
        # Loss and matcher (used only during training)                        #
        # ------------------------------------------------------------------ #
        self.criterion = BILTLoss(num_classes, use_ciou=use_ciou)
        self.matcher = AnchorMatcher()

    # ---------------------------------------------------------------------- #

    def forward(
        self,
        images: torch.Tensor,
        targets: Optional[List[Dict[str, torch.Tensor]]] = None,
    ) -> Union[Dict[str, torch.Tensor], List[Dict[str, torch.Tensor]]]:
        """
        Parameters
        ----------
        images  : (B, 3, H, W)  normalised input
        targets : optional list of dicts (one per image), each containing:
                    "boxes"  : (G, 4)  [x1, y1, x2, y2] in pixel coords
                    "labels" : (G,)    class ids (1-indexed, no background)

        Returns
        -------
        Training  (targets provided)
            dict  {"total": scalar, "cls": scalar, "box": scalar}

        Inference (no targets)
            list of dicts  {"boxes": (N,4), "scores": (N,), "labels": (N,)}
        """
        backbone_feats = self.backbone(images)          # [C3, C4, C5]
        fpn_feats = self.neck(backbone_feats)           # [P3, P4, P5, P6]
        cls_preds, box_preds = self.head(fpn_feats)     # (B,A,C), (B,A,4)
        anchors = self.anchor_gen(fpn_feats)            # (total_A, 4)

        if self.training and targets is not None:
            return self._compute_loss(cls_preds, box_preds, anchors, targets)

        return self._decode_predictions(
            cls_preds, box_preds, anchors, images.shape[-2:]
        )

    # ---------------------------------------------------------------------- #
    # Training helpers                                                        #
    # ---------------------------------------------------------------------- #

    def _compute_loss(
        self,
        cls_preds: torch.Tensor,
        box_preds: torch.Tensor,
        anchors: torch.Tensor,
        targets: List[Dict[str, torch.Tensor]],
    ) -> Dict[str, torch.Tensor]:

        B = cls_preds.shape[0]
        A = anchors.shape[0]
        device = anchors.device

        all_cls_targets: List[torch.Tensor] = []
        all_box_targets: List[torch.Tensor] = []
        all_pos_masks: List[torch.Tensor] = []

        for i in range(B):
            gt_boxes = targets[i]["boxes"].to(device)
            gt_labels = targets[i]["labels"].to(device)

            matched_cls, matched_boxes = self.matcher(
                anchors, gt_boxes, gt_labels
            )

            pos_mask = matched_cls > 0

            box_targets = torch.zeros(A, 4, device=device)
            if pos_mask.any():
                box_targets[pos_mask] = encode_boxes(
                    anchors[pos_mask], matched_boxes[pos_mask]
                )

            all_cls_targets.append(matched_cls)
            all_box_targets.append(box_targets)
            all_pos_masks.append(pos_mask)

        cls_targets = torch.stack(all_cls_targets)   # (B, A)
        box_targets = torch.stack(all_box_targets)   # (B, A, 4)
        pos_masks = torch.stack(all_pos_masks)        # (B, A)

        return self.criterion(
            cls_preds, box_preds, cls_targets, box_targets, pos_masks,
            anchors=anchors,
        )

    # ---------------------------------------------------------------------- #
    # Inference helpers                                                       #
    # ---------------------------------------------------------------------- #

    def _decode_predictions(
        self,
        cls_preds: torch.Tensor,
        box_preds: torch.Tensor,
        anchors: torch.Tensor,
        image_shape: Tuple[int, int],
        score_threshold: float = 0.01,
        nms_iou_threshold: float = 0.5,
        max_detections: int = 300,
    ) -> List[Dict[str, torch.Tensor]]:

        results: List[Dict[str, torch.Tensor]] = []
        H, W = image_shape

        for i in range(cls_preds.shape[0]):
            scores = cls_preds[i].sigmoid()           # (A, num_classes)
            boxes = decode_boxes(anchors, box_preds[i])  # (A, 4)

            # Clip to image bounds
            boxes[:, [0, 2]] = boxes[:, [0, 2]].clamp(0.0, float(W))
            boxes[:, [1, 3]] = boxes[:, [1, 3]].clamp(0.0, float(H))

            # Best class per anchor
            max_scores, labels = scores.max(dim=1)
            keep = max_scores > score_threshold

            if not keep.any():
                results.append({
                    "boxes":  torch.zeros(0, 4),
                    "scores": torch.zeros(0),
                    "labels": torch.zeros(0, dtype=torch.long),
                })
                continue

            boxes = boxes[keep]
            max_scores = max_scores[keep]
            labels = labels[keep]

            # Per-class NMS
            keep_idx: List[torch.Tensor] = []
            for cls in labels.unique():
                cls_mask = labels == cls
                nms_keep = torch.ops.torchvision.nms(
                    boxes[cls_mask], max_scores[cls_mask], nms_iou_threshold
                )
                orig = cls_mask.nonzero(as_tuple=False).squeeze(1)
                keep_idx.append(orig[nms_keep])

            if keep_idx:
                keep_all = torch.cat(keep_idx)
                top = max_scores[keep_all].argsort(descending=True)[:max_detections]
                keep_all = keep_all[top]
                results.append({
                    "boxes":  boxes[keep_all],
                    "scores": max_scores[keep_all],
                    "labels": labels[keep_all] + 1,  # 1-indexed for consistency
                })
            else:
                results.append({
                    "boxes":  torch.zeros(0, 4),
                    "scores": torch.zeros(0),
                    "labels": torch.zeros(0, dtype=torch.long),
                })

        return results


# ---------------------------------------------------------------------------
# DetectionModel – save / load / training utilities
# ---------------------------------------------------------------------------

class DetectionModel:
    """
    Convenience wrapper around BILTDetector that adds:
      - Model save / load (checkpoint with metadata)
      - Optimizer and scheduler factories
      - train() / eval() delegation

    Parameters
    ----------
    variant      : str           Variant name (spark / flash / core / pro / max).
    num_classes  : int           Number of object categories.
    class_names  : list of str   Human-readable class names.
    """

    def __init__(
        self,
        variant: str = DEFAULT_VARIANT,
        num_classes: int = 80,
        class_names: Optional[List[str]] = None,
        use_ciou: bool = False,
    ):
        self.variant = variant
        self.num_classes = num_classes
        self.class_names = class_names or [str(i) for i in range(1, num_classes + 1)]
        self.model = BILTDetector(variant, num_classes, use_ciou=use_ciou)

    # ---------------------------------------------------------------------- #
    # Delegation                                                              #
    # ---------------------------------------------------------------------- #

    def __call__(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def train(self) -> "DetectionModel":
        self.model.train()
        return self

    def eval(self) -> "DetectionModel":
        self.model.eval()
        return self

    def to(self, device) -> "DetectionModel":
        self.model = self.model.to(device)
        return self

    def parameters(self):
        return self.model.parameters()

    # ---------------------------------------------------------------------- #
    # Persistence                                                             #
    # ---------------------------------------------------------------------- #

    def save(
        self,
        save_path: Union[str, Path],
        class_names: Optional[List[str]] = None,
        class_id_mapping: Optional[dict] = None,
    ) -> None:
        """Save model checkpoint to *save_path*.

        Weights are stored in float16 to halve disk usage.  They are
        transparently upcast back to float32 by :meth:`load`.
        """
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        # Downcast floating-point tensors to float16 — this cuts file size in
        # half with no meaningful loss of precision for stored weights.
        half_sd = {
            k: v.half() if v.is_floating_point() else v
            for k, v in self.model.state_dict().items()
        }

        checkpoint = {
            "model_state_dict": half_sd,
            "storage_dtype":    "float16",
            "num_classes":      self.num_classes,
            "class_names":      class_names or self.class_names,
            "variant":          self.variant,
            "input_size":       self.model.input_size,
            "class_id_mapping": class_id_mapping,
            "architecture":     "bilt_fpn",
        }
        torch.save(checkpoint, save_path)
        logger.info(f"Model saved to {save_path}")

    @staticmethod
    def load(
        model_path: Union[str, Path],
        device: str = "cpu",
    ) -> Tuple[nn.Module, List[str]]:
        """
        Load a saved BILTDetector checkpoint.

        Returns
        -------
        (nn.Module, class_names)   the raw BILTDetector and class name list.
        """
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")

        logger.info(f"Loading model from {model_path}")
        checkpoint = torch.load(model_path, map_location=device)

        variant = checkpoint.get("variant", DEFAULT_VARIANT)
        num_classes = checkpoint["num_classes"]
        class_names = checkpoint.get("class_names", [])

        # Upcast float16 storage weights back to float32 for inference.
        # Old checkpoints without the flag are loaded as-is (float32).
        sd = checkpoint["model_state_dict"]
        if checkpoint.get("storage_dtype") == "float16":
            sd = {
                k: v.float() if v.is_floating_point() else v
                for k, v in sd.items()
            }

        detector = BILTDetector(variant, num_classes)
        detector.load_state_dict(sd)
        detector.eval()

        logger.info(
            f"Loaded BILT-{variant} with {num_classes} classes from {model_path}"
        )
        return detector, class_names


# ---------------------------------------------------------------------------
# Optimizer / scheduler helpers (used by Trainer)
# ---------------------------------------------------------------------------

def get_optimizer(
    model: nn.Module,
    learning_rate: float = 5e-4,
    weight_decay: float = 1e-4,
) -> torch.optim.Optimizer:
    """AdamW optimizer for detection training."""
    params = [p for p in model.parameters() if p.requires_grad]
    return torch.optim.AdamW(params, lr=learning_rate, weight_decay=weight_decay)


def get_optimizer_differential(
    model: nn.Module,
    learning_rate: float = 5e-4,
    backbone_lr_multiplier: float = 0.1,
    weight_decay: float = 1e-4,
) -> torch.optim.Optimizer:
    """AdamW with a lower learning rate for the pretrained backbone.

    The backbone already carries ImageNet features; training it at the same
    rate as the randomly-initialised head risks destroying those features.
    A 10× lower rate lets the head drive learning while the backbone adapts
    gently to the new domain.
    """
    backbone_ids = {id(p) for p in model.backbone.parameters()}
    backbone_params = [p for p in model.backbone.parameters() if p.requires_grad]
    head_params = [
        p for p in model.parameters()
        if p.requires_grad and id(p) not in backbone_ids
    ]
    return torch.optim.AdamW(
        [
            {"params": head_params,     "lr": learning_rate},
            {"params": backbone_params, "lr": learning_rate * backbone_lr_multiplier},
        ],
        weight_decay=weight_decay,
    )


def get_lr_scheduler(
    optimizer: torch.optim.Optimizer,
    num_epochs: int,
    eta_min: float = 1e-6,
) -> torch.optim.lr_scheduler.LRScheduler:
    """Cosine annealing LR decay."""
    return torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=num_epochs, eta_min=eta_min
    )
