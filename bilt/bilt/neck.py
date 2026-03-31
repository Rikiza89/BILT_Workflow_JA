# BILT (Because I Like Twice) - A PyTorch-based object detection library
# Copyright (C) 2026 Rikiza89
# Licensed under the Apache License, Version 2.0

"""
BILT Feature Pyramid Network (FPN) neck.

Takes three multi-scale backbone feature maps [C3, C4, C5] and produces
four FPN levels [P3, P4, P5, P6] each with *out_channels* feature channels.

Architecture
------------
Lateral 1×1 convolutions reduce each backbone level to out_channels.
Top-down pathway upsample + adds from the level above.
3×3 output convolutions smooth each merged feature map.
An extra stride-2 conv on C5 produces the P6 level for large objects.

All convolutions use BatchNorm + ReLU (GroupNorm on the output conv to
improve training stability with small batch sizes).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List


def _conv_bn_relu(in_ch: int, out_ch: int, kernel: int = 3,
                  stride: int = 1, padding: int = 1) -> nn.Sequential:
    """Standard Conv-BN-ReLU block."""
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel, stride=stride,
                  padding=padding, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )


class FPNNeck(nn.Module):
    """
    Feature Pyramid Network neck.

    Parameters
    ----------
    in_channels : list of int
        Output channel counts from the backbone: [c3_ch, c4_ch, c5_ch].
    out_channels : int
        Uniform channel width for all FPN output levels.
    """

    def __init__(self, in_channels: List[int], out_channels: int):
        super().__init__()

        c3_ch, c4_ch, c5_ch = in_channels

        # ------------------------------------------------------------------ #
        # Lateral 1×1 projections (reduce backbone channels → out_channels)  #
        # ------------------------------------------------------------------ #
        self.lateral5 = nn.Conv2d(c5_ch, out_channels, 1)
        self.lateral4 = nn.Conv2d(c4_ch, out_channels, 1)
        self.lateral3 = nn.Conv2d(c3_ch, out_channels, 1)

        # ------------------------------------------------------------------ #
        # Output 3×3 convolutions (anti-alias after nearest-neighbour upsamp)#
        # ------------------------------------------------------------------ #
        self.out5 = _conv_bn_relu(out_channels, out_channels)
        self.out4 = _conv_bn_relu(out_channels, out_channels)
        self.out3 = _conv_bn_relu(out_channels, out_channels)

        # ------------------------------------------------------------------ #
        # Extra level P6 (stride 2 from C5 — helps detect large objects)     #
        # ------------------------------------------------------------------ #
        self.extra6 = _conv_bn_relu(c5_ch, out_channels, stride=2)

        self.out_channels = out_channels
        self.num_levels = 4   # P3, P4, P5, P6

        self._init_weights()

    # ---------------------------------------------------------------------- #

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight, mode="fan_out",
                                         nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    # ---------------------------------------------------------------------- #

    def forward(self, features: List[torch.Tensor]) -> List[torch.Tensor]:
        """
        Parameters
        ----------
        features : [C3, C4, C5]   tensors from the backbone

        Returns
        -------
        [P3, P4, P5, P6]   four FPN tensors, all with self.out_channels
        """
        c3, c4, c5 = features

        # Top-down lateral merging
        lat5 = self.lateral5(c5)
        lat4 = self.lateral4(c4) + F.interpolate(
            lat5, size=c4.shape[-2:], mode="nearest"
        )
        lat3 = self.lateral3(c3) + F.interpolate(
            lat4, size=c3.shape[-2:], mode="nearest"
        )

        p5 = self.out5(lat5)
        p4 = self.out4(lat4)
        p3 = self.out3(lat3)
        p6 = self.extra6(c5)

        return [p3, p4, p5, p6]
