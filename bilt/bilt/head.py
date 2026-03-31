# BILT (Because I Like Twice) - A PyTorch-based object detection library
# Copyright (C) 2026 Rikiza89
# Licensed under the Apache License, Version 2.0

"""
BILT detection head.

A shared convolutional head is applied independently to each FPN level.
It produces two outputs per spatial location per anchor:
  - class logits : (B, A*num_classes, H, W)
  - box deltas   : (B, A*4,           H, W)

The classification branch initialises its final bias using the focal-loss
prior probability trick so the model starts with low confidence rather than
random noise, stabilising early training.
"""

import math
import torch
import torch.nn as nn
from typing import List, Tuple


def _conv_gn_relu(in_ch: int, out_ch: int) -> nn.Sequential:
    """
    Conv 3×3 → GroupNorm → ReLU.

    GroupNorm is used instead of BatchNorm so that the head works correctly
    at batch size 1 (inference) without special-casing.
    """
    num_groups = min(32, out_ch // 4)
    num_groups = max(1, num_groups)
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
        nn.GroupNorm(num_groups, out_ch),
        nn.ReLU(inplace=True),
    )


class BILTHead(nn.Module):
    """
    Shared anchor-based detection head applied to all FPN levels.

    Parameters
    ----------
    in_channels  : int   Number of channels in each FPN feature map.
    num_classes  : int   Number of object categories (excluding background).
    num_anchors  : int   Anchors per spatial location.
    num_convs    : int   Convolutional layers in each tower (2–4).
    """

    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        num_anchors: int,
        num_convs: int = 3,
    ):
        super().__init__()

        self.num_classes = num_classes
        self.num_anchors = num_anchors

        # ------------------------------------------------------------------ #
        # Classification tower                                                #
        # ------------------------------------------------------------------ #
        cls_layers: List[nn.Module] = []
        for _ in range(num_convs):
            cls_layers.append(_conv_gn_relu(in_channels, in_channels))
        cls_layers.append(
            nn.Conv2d(in_channels, num_anchors * num_classes, 1)
        )
        self.cls_tower = nn.Sequential(*cls_layers)

        # ------------------------------------------------------------------ #
        # Regression tower                                                    #
        # ------------------------------------------------------------------ #
        reg_layers: List[nn.Module] = []
        for _ in range(num_convs):
            reg_layers.append(_conv_gn_relu(in_channels, in_channels))
        reg_layers.append(
            nn.Conv2d(in_channels, num_anchors * 4, 1)
        )
        self.reg_tower = nn.Sequential(*reg_layers)

        self._init_weights()

    # ---------------------------------------------------------------------- #

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        # Focal-loss prior: initialise cls output bias so sigmoid ≈ 0.01
        prior = 0.01
        bias_val = -math.log((1.0 - prior) / prior)
        nn.init.constant_(self.cls_tower[-1].bias, bias_val)

    # ---------------------------------------------------------------------- #

    def forward(
        self, features: List[torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        features : list of FPN tensors (B, C, H_i, W_i) for i in {P3..P6}

        Returns
        -------
        cls_preds : (B, total_anchors, num_classes)
        box_preds : (B, total_anchors, 4)
        """
        cls_preds: List[torch.Tensor] = []
        box_preds: List[torch.Tensor] = []

        for feat in features:
            B, _, H, W = feat.shape

            cls = self.cls_tower(feat)          # (B, A*C, H, W)
            box = self.reg_tower(feat)          # (B, A*4, H, W)

            # Reshape to (B, H*W*A, C or 4)
            cls = cls.permute(0, 2, 3, 1).reshape(B, -1, self.num_classes)
            box = box.permute(0, 2, 3, 1).reshape(B, -1, 4)

            cls_preds.append(cls)
            box_preds.append(box)

        return torch.cat(cls_preds, dim=1), torch.cat(box_preds, dim=1)
