# BILT (Because I Like Twice) - A PyTorch-based object detection library
# Copyright (C) 2026 Rikiza89
# Licensed under the Apache License, Version 2.0

"""
BILT anchor utilities.

Provides:
  AnchorGenerator  – generates anchor boxes for all FPN levels
  AnchorMatcher   – assigns GT boxes to anchors for training
  encode_boxes     – convert GT boxes to regression deltas
  decode_boxes     – convert predicted deltas back to boxes
  box_iou          – pairwise IoU between two sets of boxes
"""

import math
import torch
from typing import List, Tuple


# ---------------------------------------------------------------------------
# IoU helper
# ---------------------------------------------------------------------------

def box_iou(boxes_a: torch.Tensor, boxes_b: torch.Tensor) -> torch.Tensor:
    """
    Compute pairwise IoU between two sets of axis-aligned boxes.

    Parameters
    ----------
    boxes_a : (N, 4)  [x1, y1, x2, y2]
    boxes_b : (M, 4)  [x1, y1, x2, y2]

    Returns
    -------
    iou : (N, M)
    """
    area_a = (boxes_a[:, 2] - boxes_a[:, 0]).clamp(0) * \
             (boxes_a[:, 3] - boxes_a[:, 1]).clamp(0)
    area_b = (boxes_b[:, 2] - boxes_b[:, 0]).clamp(0) * \
             (boxes_b[:, 3] - boxes_b[:, 1]).clamp(0)

    inter_x1 = torch.max(boxes_a[:, None, 0], boxes_b[None, :, 0])
    inter_y1 = torch.max(boxes_a[:, None, 1], boxes_b[None, :, 1])
    inter_x2 = torch.min(boxes_a[:, None, 2], boxes_b[None, :, 2])
    inter_y2 = torch.min(boxes_a[:, None, 3], boxes_b[None, :, 3])

    inter = (inter_x2 - inter_x1).clamp(0) * (inter_y2 - inter_y1).clamp(0)
    union = area_a[:, None] + area_b[None, :] - inter

    return inter / (union + 1e-7)


# ---------------------------------------------------------------------------
# Delta encoding / decoding
# ---------------------------------------------------------------------------

def encode_boxes(
    anchors: torch.Tensor,
    gt_boxes: torch.Tensor,
) -> torch.Tensor:
    """
    Encode ground-truth boxes as deltas relative to anchors.

    Delta format: [dx, dy, dw, dh]
      dx = (gx - ax) / aw
      dy = (gy - ay) / ah
      dw = log(gw / aw)
      dh = log(gh / ah)

    Parameters
    ----------
    anchors  : (N, 4)  [x1, y1, x2, y2]
    gt_boxes : (N, 4)  [x1, y1, x2, y2]

    Returns
    -------
    deltas : (N, 4)
    """
    aw = (anchors[:, 2] - anchors[:, 0]).clamp(min=1.0)
    ah = (anchors[:, 3] - anchors[:, 1]).clamp(min=1.0)
    ax = anchors[:, 0] + 0.5 * aw
    ay = anchors[:, 1] + 0.5 * ah

    gw = (gt_boxes[:, 2] - gt_boxes[:, 0]).clamp(min=1.0)
    gh = (gt_boxes[:, 3] - gt_boxes[:, 1]).clamp(min=1.0)
    gx = gt_boxes[:, 0] + 0.5 * gw
    gy = gt_boxes[:, 1] + 0.5 * gh

    dx = (gx - ax) / aw
    dy = (gy - ay) / ah
    dw = torch.log(gw / aw)
    dh = torch.log(gh / ah)

    return torch.stack([dx, dy, dw, dh], dim=-1)


def decode_boxes(
    anchors: torch.Tensor,
    deltas: torch.Tensor,
) -> torch.Tensor:
    """
    Decode predicted deltas back to absolute box coordinates.

    Parameters
    ----------
    anchors : (N, 4)  [x1, y1, x2, y2]
    deltas  : (N, 4)  [dx, dy, dw, dh]

    Returns
    -------
    boxes : (N, 4)  [x1, y1, x2, y2]
    """
    aw = (anchors[:, 2] - anchors[:, 0]).clamp(min=1.0)
    ah = (anchors[:, 3] - anchors[:, 1]).clamp(min=1.0)
    ax = anchors[:, 0] + 0.5 * aw
    ay = anchors[:, 1] + 0.5 * ah

    dx, dy, dw, dh = (
        deltas[:, 0], deltas[:, 1],
        deltas[:, 2].clamp(max=4.0),
        deltas[:, 3].clamp(max=4.0),
    )

    gx = ax + dx * aw
    gy = ay + dy * ah
    gw = aw * torch.exp(dw)
    gh = ah * torch.exp(dh)

    x1 = gx - 0.5 * gw
    y1 = gy - 0.5 * gh
    x2 = gx + 0.5 * gw
    y2 = gy + 0.5 * gh

    return torch.stack([x1, y1, x2, y2], dim=-1)


# ---------------------------------------------------------------------------
# Anchor generator
# ---------------------------------------------------------------------------

class AnchorGenerator:
    """
    Generates anchor boxes for each FPN level.

    Each FPN level has one base size, S octave scales and R aspect ratios,
    giving S×R anchors per spatial location.  The octave scales (default
    1.0, 1.26, 1.587 — i.e. ×∛2 steps within each octave) ensure that
    every object size is covered by at least one anchor with IoU > 0.5,
    which is essential for getting positive matches in small datasets.

    Parameters
    ----------
    strides       : list of int    Spatial stride for each FPN level.
    anchor_sizes  : list of int    Base anchor size for each level.
    anchor_scales : tuple of float Octave scale multipliers (default RetinaNet style).
    aspect_ratios : tuple of float Width/height ratios, e.g. (0.5, 1.0, 2.0).
    """

    def __init__(
        self,
        strides: List[int],
        anchor_sizes: List[int],
        anchor_scales: Tuple[float, ...] = (1.0, 1.26, 1.587),
        aspect_ratios: Tuple[float, ...] = (0.5, 1.0, 2.0),
    ):
        assert len(strides) == len(anchor_sizes), (
            "strides and anchor_sizes must have the same length"
        )
        self.strides = strides
        self.anchor_sizes = anchor_sizes
        self.anchor_scales = anchor_scales
        self.aspect_ratios = aspect_ratios
        self.num_anchors = len(anchor_scales) * len(aspect_ratios)

    def _base_anchors(self, size: int, device: torch.device) -> torch.Tensor:
        """Build (num_anchors, 4) base anchors centred at origin.

        Iterates over all (scale, ratio) combinations so that each level
        covers a continuous range of object sizes within the octave.
        """
        anchors = []
        for scale in self.anchor_scales:
            scaled = size * scale
            for ratio in self.aspect_ratios:
                w = scaled * math.sqrt(ratio)
                h = scaled / math.sqrt(ratio)
                anchors.append([-w / 2, -h / 2, w / 2, h / 2])
        return torch.tensor(anchors, dtype=torch.float32, device=device)

    def __call__(
        self, feature_maps: List[torch.Tensor]
    ) -> torch.Tensor:
        """
        Generate all anchors for the given feature maps.

        Parameters
        ----------
        feature_maps : list of (B, C, H, W) tensors (one per FPN level)

        Returns
        -------
        anchors : (total_anchors, 4)  [x1, y1, x2, y2] in image pixel space
        """
        all_anchors: List[torch.Tensor] = []

        for feat, stride, size in zip(
            feature_maps, self.strides, self.anchor_sizes
        ):
            H, W = feat.shape[-2:]
            device = feat.device

            # Grid of anchor centres
            shift_x = (torch.arange(W, device=device, dtype=torch.float32)
                       + 0.5) * stride
            shift_y = (torch.arange(H, device=device, dtype=torch.float32)
                       + 0.5) * stride
            grid_y, grid_x = torch.meshgrid(shift_y, shift_x, indexing="ij")
            shifts = torch.stack(
                [grid_x, grid_y, grid_x, grid_y], dim=-1
            ).reshape(-1, 4)                          # (H*W, 4)

            base = self._base_anchors(size, device)   # (A, 4)

            # Broadcast and sum: (H*W, 1, 4) + (1, A, 4) → (H*W*A, 4)
            anchors = (shifts[:, None, :] + base[None, :, :]).reshape(-1, 4)
            all_anchors.append(anchors)

        return torch.cat(all_anchors, dim=0)


# ---------------------------------------------------------------------------
# Anchor matcher
# ---------------------------------------------------------------------------

class AnchorMatcher:
    """
    Assigns ground-truth boxes to anchors based on IoU thresholds.

    Anchors with IoU ≥ pos_thresh are assigned the matching GT class.
    Anchors with IoU < neg_thresh are assigned background (class 0).
    Anchors in between are marked ignore (class -1).
    Every GT box is guaranteed to be matched to at least one anchor
    (the one with the highest IoU to that GT).

    Parameters
    ----------
    pos_thresh : float   IoU threshold to call an anchor positive.
    neg_thresh : float   IoU threshold below which an anchor is negative.
    """

    def __init__(self, pos_thresh: float = 0.35, neg_thresh: float = 0.25):
        self.pos_thresh = pos_thresh
        self.neg_thresh = neg_thresh

    def __call__(
        self,
        anchors: torch.Tensor,
        gt_boxes: torch.Tensor,
        gt_labels: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        anchors   : (A, 4)   all anchor boxes
        gt_boxes  : (G, 4)   ground-truth boxes
        gt_labels : (G,)     ground-truth labels (1-indexed, no background)

        Returns
        -------
        matched_cls  : (A,)    -1 = ignore, 0 = background, >0 = class id
        matched_boxes: (A, 4)  GT box for each positive anchor (zero otherwise)
        """
        A = anchors.shape[0]
        device = anchors.device

        if gt_boxes.numel() == 0:
            return (
                torch.zeros(A, dtype=torch.long, device=device),
                torch.zeros(A, 4, device=device),
            )

        iou = box_iou(anchors, gt_boxes)          # (A, G)
        best_gt_iou, best_gt_idx = iou.max(dim=1) # (A,)

        # Default: ignore
        matched_cls = torch.full(
            (A,), -1, dtype=torch.long, device=device
        )

        # Negatives
        matched_cls[best_gt_iou < self.neg_thresh] = 0

        # Positives
        pos_mask = best_gt_iou >= self.pos_thresh
        matched_cls[pos_mask] = gt_labels[best_gt_idx[pos_mask]]

        # Force every GT to be matched by its best-IoU anchor
        _, best_anchor_per_gt = iou.max(dim=0)    # (G,)
        for g_idx, a_idx in enumerate(best_anchor_per_gt):
            matched_cls[a_idx] = gt_labels[g_idx]

        # Build matched boxes tensor (used only for positive anchors)
        matched_boxes = torch.zeros(A, 4, device=device)
        all_pos = matched_cls > 0
        if all_pos.any():
            matched_boxes[all_pos] = gt_boxes[best_gt_idx[all_pos]]

        # Also fill for forced GT matches
        for g_idx, a_idx in enumerate(best_anchor_per_gt):
            matched_boxes[a_idx] = gt_boxes[g_idx]

        return matched_cls, matched_boxes
