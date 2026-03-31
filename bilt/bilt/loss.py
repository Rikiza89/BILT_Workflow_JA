# BILT (Because I Like Twice) - A PyTorch-based object detection library
# Copyright (C) 2026 Rikiza89
# Licensed under the Apache License, Version 2.0

"""
BILT training losses.

BILTLoss combines:
  - Focal loss  (classification)  — focuses training on hard examples by
    down-weighting well-classified ones.  Introduced in RetinaNet
    (Lin et al. 2017, https://arxiv.org/abs/1708.02002).
  - Box regression loss — either Smooth-L1 or CIoU.

    Smooth-L1  (default): less sensitive to outliers than L1/L2, operates
    on encoded deltas (fast, no anchor decoding required).

    CIoU (Complete IoU, Zheng et al. 2020): directly optimises the IoU
    between predicted and ground-truth boxes, adding a centre-distance
    penalty and an aspect-ratio consistency term.  Produces tighter boxes
    and higher mAP, especially on small datasets.

Both losses are normalised by the number of positive anchors in the batch so
that the scale remains roughly constant across different batch sizes and
image densities.
"""

import math
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Focal loss
# ---------------------------------------------------------------------------

def sigmoid_focal_loss(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    alpha: float = 0.25,
    gamma: float = 2.0,
) -> torch.Tensor:
    """
    Binary sigmoid focal loss (sum reduction).

    Parameters
    ----------
    inputs  : (N, C)  raw logits
    targets : (N, C)  binary targets {0, 1}
    alpha   : float   class-balance weighting factor
    gamma   : float   focusing parameter (0 = cross-entropy)

    Returns
    -------
    loss : scalar  (sum over all elements)
    """
    p = torch.sigmoid(inputs)
    ce = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")

    p_t = p * targets + (1.0 - p) * (1.0 - targets)
    loss = ce * ((1.0 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1.0 - alpha) * (1.0 - targets)
        loss = alpha_t * loss

    return loss.sum()


# ---------------------------------------------------------------------------
# Smooth-L1 loss
# ---------------------------------------------------------------------------

def smooth_l1_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    beta: float = 0.1,
) -> torch.Tensor:
    """
    Smooth-L1 (Huber) loss with customisable transition point.

        L(x) = 0.5 * x^2 / beta   if |x| < beta
               |x| - 0.5 * beta   otherwise

    Parameters
    ----------
    pred   : (N, 4)
    target : (N, 4)
    beta   : float  transition from quadratic to linear

    Returns
    -------
    loss : scalar  (sum over all elements)
    """
    diff = (pred - target).abs()
    loss = torch.where(
        diff < beta,
        0.5 * diff ** 2 / beta,
        diff - 0.5 * beta,
    )
    return loss.sum()


# ---------------------------------------------------------------------------
# CIoU loss
# ---------------------------------------------------------------------------

def ciou_loss(
    pred_boxes: torch.Tensor,
    target_boxes: torch.Tensor,
    eps: float = 1e-7,
) -> torch.Tensor:
    """
    Complete IoU loss (Zheng et al. 2020, https://arxiv.org/abs/1911.08287).

    Unlike Smooth-L1 which works on encoded coordinate deltas, CIoU directly
    optimises three geometric properties:
      1. Overlap area  (IoU term)
      2. Centre-point distance  (ρ² / c² penalty)
      3. Aspect-ratio consistency  (α × v penalty)

    Parameters
    ----------
    pred_boxes   : (N, 4)  predicted boxes  [x1, y1, x2, y2]  pixel coords
    target_boxes : (N, 4)  ground-truth boxes  [x1, y1, x2, y2]  pixel coords
    eps          : float   small constant for numerical stability

    Returns
    -------
    loss : scalar  mean CIoU loss  (1 − CIoU, averaged over N)
    """
    # ── Intersection ──────────────────────────────────────────────────────────
    inter_x1 = torch.max(pred_boxes[:, 0], target_boxes[:, 0])
    inter_y1 = torch.max(pred_boxes[:, 1], target_boxes[:, 1])
    inter_x2 = torch.min(pred_boxes[:, 2], target_boxes[:, 2])
    inter_y2 = torch.min(pred_boxes[:, 3], target_boxes[:, 3])

    inter_w = (inter_x2 - inter_x1).clamp(min=0)
    inter_h = (inter_y2 - inter_y1).clamp(min=0)
    inter_area = inter_w * inter_h

    # ── Union ─────────────────────────────────────────────────────────────────
    pred_w   = (pred_boxes[:, 2]   - pred_boxes[:, 0]).clamp(min=0)
    pred_h   = (pred_boxes[:, 3]   - pred_boxes[:, 1]).clamp(min=0)
    target_w = (target_boxes[:, 2] - target_boxes[:, 0]).clamp(min=0)
    target_h = (target_boxes[:, 3] - target_boxes[:, 1]).clamp(min=0)

    pred_area   = pred_w   * pred_h
    target_area = target_w * target_h
    union_area  = pred_area + target_area - inter_area + eps

    iou = inter_area / union_area   # (N,)

    # ── Smallest enclosing box diagonal² ─────────────────────────────────────
    encl_x1 = torch.min(pred_boxes[:, 0], target_boxes[:, 0])
    encl_y1 = torch.min(pred_boxes[:, 1], target_boxes[:, 1])
    encl_x2 = torch.max(pred_boxes[:, 2], target_boxes[:, 2])
    encl_y2 = torch.max(pred_boxes[:, 3], target_boxes[:, 3])

    c2 = (encl_x2 - encl_x1) ** 2 + (encl_y2 - encl_y1) ** 2 + eps

    # ── Centre-distance penalty ────────────────────────────────────────────
    pred_cx   = (pred_boxes[:, 0]   + pred_boxes[:, 2])   / 2
    pred_cy   = (pred_boxes[:, 1]   + pred_boxes[:, 3])   / 2
    target_cx = (target_boxes[:, 0] + target_boxes[:, 2]) / 2
    target_cy = (target_boxes[:, 1] + target_boxes[:, 3]) / 2

    rho2 = (pred_cx - target_cx) ** 2 + (pred_cy - target_cy) ** 2

    # ── Aspect-ratio consistency term ─────────────────────────────────────────
    v = (4.0 / (math.pi ** 2)) * (
        torch.atan(target_w / (target_h + eps))
        - torch.atan(pred_w  / (pred_h   + eps))
    ) ** 2

    with torch.no_grad():
        alpha_ciou = v / (1.0 - iou + v + eps)

    ciou = (iou - rho2 / c2 - alpha_ciou * v).clamp(-1.0, 1.0)

    # loss = 1 − CIoU, summed (caller divides by num_pos)
    return (1.0 - ciou).sum()


# ---------------------------------------------------------------------------
# Combined loss module
# ---------------------------------------------------------------------------

class BILTLoss(nn.Module):
    """
    Joint focal + box regression detection loss.

    Parameters
    ----------
    num_classes : int    Number of object classes (excluding background).
    alpha       : float  Focal loss alpha (class-balance weight).
    gamma       : float  Focal loss gamma (focusing strength).
    box_weight  : float  Relative weight of regression loss vs. cls loss.
    use_ciou    : bool   Use CIoU box loss instead of Smooth-L1.
                         Requires *anchors* to be passed in forward().
    """

    def __init__(
        self,
        num_classes: int,
        alpha: float = 0.25,
        gamma: float = 2.0,
        box_weight: float = 1.0,
        use_ciou: bool = False,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.alpha = alpha
        self.gamma = gamma
        self.box_weight = box_weight
        self.use_ciou = use_ciou

    def forward(
        self,
        cls_preds: torch.Tensor,   # (B, A, num_classes)
        box_preds: torch.Tensor,   # (B, A, 4)
        cls_targets: torch.Tensor, # (B, A)  -1=ignore, 0=bg, >0=class
        box_targets: torch.Tensor, # (B, A, 4) encoded deltas
        pos_mask: torch.Tensor,    # (B, A)  bool
        anchors: Optional[torch.Tensor] = None,  # (A, 4) — required for CIoU
    ) -> Dict[str, torch.Tensor]:
        """
        Parameters
        ----------
        anchors : (A, 4) anchor boxes in [x1,y1,x2,y2] pixel coords.
                  Required when use_ciou=True; ignored for Smooth-L1.

        Returns
        -------
        dict with keys 'total', 'cls', 'box'
        """
        num_pos = pos_mask.sum().clamp(min=1).float()

        # ------------------------------------------------------------------ #
        # Classification loss (focal, applied to all non-ignored anchors)    #
        # ------------------------------------------------------------------ #
        valid_mask = cls_targets >= 0                       # exclude ignore

        # Build one-hot targets (background = all-zeros)
        cls_targets_oh = torch.zeros_like(cls_preds)        # (B, A, C)
        if pos_mask.any():
            pos_classes = (cls_targets[pos_mask] - 1).clamp(
                0, self.num_classes - 1
            )
            flat_pos = pos_mask.reshape(-1).nonzero(as_tuple=False).squeeze(1)
            cls_targets_oh.reshape(-1, self.num_classes)[flat_pos, pos_classes] = 1.0

        cls_loss = sigmoid_focal_loss(
            cls_preds[valid_mask],
            cls_targets_oh[valid_mask],
            alpha=self.alpha,
            gamma=self.gamma,
        ) / num_pos

        # ------------------------------------------------------------------ #
        # Regression loss (positive anchors only)                            #
        # ------------------------------------------------------------------ #
        if pos_mask.any():
            if self.use_ciou and anchors is not None:
                # Decode predictions and targets to absolute [x1,y1,x2,y2]
                # and compute geometry-aware CIoU loss.
                from .anchors import decode_boxes

                # Gather anchors for each positive (same anchors across batch)
                B, A = pos_mask.shape
                anchor_idx = (
                    torch.arange(A, device=pos_mask.device)
                    .unsqueeze(0).expand(B, -1)        # (B, A)
                    [pos_mask]                          # (N_pos,)
                )
                pos_anchors  = anchors[anchor_idx]         # (N_pos, 4)
                pred_decoded = decode_boxes(pos_anchors, box_preds[pos_mask])
                tgt_decoded  = decode_boxes(pos_anchors, box_targets[pos_mask])

                box_loss = ciou_loss(pred_decoded, tgt_decoded) / num_pos
            else:
                box_loss = smooth_l1_loss(
                    box_preds[pos_mask],
                    box_targets[pos_mask],
                ) / num_pos
        else:
            box_loss = cls_preds.new_tensor(0.0)

        total = cls_loss + self.box_weight * box_loss

        return {
            "total": total,
            "cls":   cls_loss.detach(),
            "box":   box_loss.detach(),
        }
