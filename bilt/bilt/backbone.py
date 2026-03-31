# BILT (Because I Like Twice) - A PyTorch-based object detection library
# Copyright (C) 2026 Rikiza89
# Licensed under the Apache License, Version 2.0

"""
BILT backbone feature extractors.

Wraps torchvision backbone architectures to expose three multi-scale feature
maps at 1/8, 1/16 and 1/32 of the input resolution (C3, C4, C5).  Backbones
are initialised with ImageNet pretrained weights by default, which dramatically
improves convergence on small datasets (even 2 images).  The neck (FPN)
consumes these three scales and optionally adds a 1/64 level.

Supported backbones
-------------------
mobilenet_v2        : MobileNetV2  (used by spark)
mobilenet_v3_small  : MobileNetV3-Small (used by flash)
mobilenet_v3_large  : MobileNetV3-Large (used by core)
resnet50            : ResNet-50    (used by pro)
resnet101           : ResNet-101   (used by max)
"""

import torch
import torch.nn as nn
from torchvision import models
from typing import List


def _pretrained(model_fn, **kwargs):
    """Load a torchvision model with ImageNet weights, compatible with both
    the new (torchvision ≥0.13) ``weights="DEFAULT"`` API and the legacy
    ``pretrained=True`` API used by older installs."""
    try:
        return model_fn(weights="DEFAULT", **kwargs)
    except TypeError:
        return model_fn(pretrained=True, **kwargs)

# ---------------------------------------------------------------------------
# Normalisation constants used by the preprocessing pipeline
# ---------------------------------------------------------------------------
NORM_MEAN = [0.485, 0.456, 0.406]
NORM_STD  = [0.229, 0.224, 0.225]


# ---------------------------------------------------------------------------
# Internal backbone builders
# ---------------------------------------------------------------------------

class _MobileNetV2Backbone(nn.Module):
    """
    MobileNetV2 backbone split at three feature scales.

    Output channels: C3=32, C4=96, C5=1280
    Output strides : 8,    16,    32
    """

    def __init__(self):
        super().__init__()
        m = _pretrained(models.mobilenet_v2)
        feats = list(m.features)
        # Split the sequential feature list into three groups
        self.stage1 = nn.Sequential(*feats[:7])    # → 32 ch,   stride 8
        self.stage2 = nn.Sequential(*feats[7:14])  # → 96 ch,   stride 16
        self.stage3 = nn.Sequential(*feats[14:])   # → 1280 ch, stride 32
        self.out_channels = [32, 96, 1280]

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        c3 = self.stage1(x)
        c4 = self.stage2(c3)
        c5 = self.stage3(c4)
        return [c3, c4, c5]


class _MobileNetV3SmallBackbone(nn.Module):
    """
    MobileNetV3-Small backbone split at three feature scales.

    Output channels: C3=24, C4=48, C5=576
    Output strides : 8,    16,    32
    """

    def __init__(self):
        super().__init__()
        m = _pretrained(models.mobilenet_v3_small)
        feats = list(m.features)
        self.stage1 = nn.Sequential(*feats[:4])    # → 24 ch,  stride 8
        self.stage2 = nn.Sequential(*feats[4:9])   # → 48 ch,  stride 16
        self.stage3 = nn.Sequential(*feats[9:])    # → 576 ch, stride 32
        self.out_channels = [24, 48, 576]

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        c3 = self.stage1(x)
        c4 = self.stage2(c3)
        c5 = self.stage3(c4)
        return [c3, c4, c5]


class _MobileNetV3LargeBackbone(nn.Module):
    """
    MobileNetV3-Large backbone split at three feature scales.

    Output channels: C3=40, C4=112, C5=960
    Output strides : 8,     16,     32
    """

    def __init__(self):
        super().__init__()
        m = _pretrained(models.mobilenet_v3_large)
        feats = list(m.features)
        self.stage1 = nn.Sequential(*feats[:7])    # → 40 ch,  stride 8
        self.stage2 = nn.Sequential(*feats[7:13])  # → 112 ch, stride 16
        self.stage3 = nn.Sequential(*feats[13:])   # → 960 ch, stride 32
        self.out_channels = [40, 112, 960]

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        c3 = self.stage1(x)
        c4 = self.stage2(c3)
        c5 = self.stage3(c4)
        return [c3, c4, c5]


class _ResNetBackbone(nn.Module):
    """
    ResNet-50 / ResNet-101 backbone split at three feature scales.

    Output channels: C3=512, C4=1024, C5=2048
    Output strides : 8,      16,      32
    """

    def __init__(self, depth: int = 50):
        super().__init__()
        if depth == 50:
            m = _pretrained(models.resnet50)
        elif depth == 101:
            m = _pretrained(models.resnet101)
        else:
            raise ValueError(f"Unsupported ResNet depth: {depth}. Use 50 or 101.")

        # Stem: brings input from stride-1 to stride-4
        self.stem = nn.Sequential(
            m.conv1, m.bn1, m.relu, m.maxpool, m.layer1
        )
        self.layer2 = m.layer2   # stride 8  → 512 ch
        self.layer3 = m.layer3   # stride 16 → 1024 ch
        self.layer4 = m.layer4   # stride 32 → 2048 ch
        self.out_channels = [512, 1024, 2048]

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        x = self.stem(x)
        c3 = self.layer2(x)
        c4 = self.layer3(c3)
        c5 = self.layer4(c4)
        return [c3, c4, c5]


# ---------------------------------------------------------------------------
# Public factory
# ---------------------------------------------------------------------------

_BACKBONE_MAP = {
    "mobilenet_v2":       lambda: _MobileNetV2Backbone(),
    "mobilenet_v3_small": lambda: _MobileNetV3SmallBackbone(),
    "mobilenet_v3_large": lambda: _MobileNetV3LargeBackbone(),
    "resnet50":           lambda: _ResNetBackbone(depth=50),
    "resnet101":          lambda: _ResNetBackbone(depth=101),
}


class BILTBackbone(nn.Module):
    """
    Unified backbone wrapper used by BILTDetector.

    Backbones are initialised with ImageNet pretrained weights, which
    provides rich low-level and semantic features out of the box.  This
    is the single most important factor for getting good detections with
    very few training images (as few as 2).

    Parameters
    ----------
    backbone_name : str
        One of ``mobilenet_v2``, ``mobilenet_v3_small``,
        ``mobilenet_v3_large``, ``resnet50``, ``resnet101``.
    """

    def __init__(self, backbone_name: str):
        super().__init__()
        if backbone_name not in _BACKBONE_MAP:
            raise ValueError(
                f"Unknown backbone '{backbone_name}'. "
                f"Supported: {list(_BACKBONE_MAP.keys())}"
            )
        self._backbone = _BACKBONE_MAP[backbone_name]()
        self.out_channels: List[int] = self._backbone.out_channels
        self.backbone_name = backbone_name

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        Parameters
        ----------
        x : Tensor  (B, 3, H, W)   normalised input

        Returns
        -------
        list of three tensors: [C3, C4, C5]
        """
        return self._backbone(x)

    def freeze(self) -> None:
        """Freeze all backbone parameters (used during head warmup)."""
        for p in self._backbone.parameters():
            p.requires_grad = False

    def unfreeze(self) -> None:
        """Unfreeze all backbone parameters."""
        for p in self._backbone.parameters():
            p.requires_grad = True
