# BILT (Because I Like Twice) - A PyTorch-based object detection library
# Copyright (C) 2026 Rikiza89
# Licensed under the Apache License, Version 2.0

"""
BILT — Because I Like Twice

A lightweight, CPU-friendly object detection library built on PyTorch.

Quick start
-----------
    from bilt import BILT

    # --- Inference on a saved model ---
    model = BILT("weights/best.pth")
    detections = model.predict("image.jpg", conf=0.25)

    # --- Train a new model ---
    model = BILT("core")                          # or spark/flash/pro/max
    metrics = model.train(dataset="data/", epochs=50)

    # --- Print available variants ---
    BILT.variants()

Model variants
--------------
    spark  — MobileNetV2,       320 px   (nano, fastest)
    flash  — MobileNetV3-Small, 416 px   (small)
    core   — MobileNetV3-Large, 512 px   (medium, default)
    pro    — ResNet-50,         640 px   (large)
    max    — ResNet-101,        640 px   (xlarge, most accurate)
"""

from .model import BILT, Results
from .variants import list_variants, get_variant_config, VARIANT_CONFIGS
from .utils import set_logging_level

__version__ = "0.2.0"
__author__ = "Rikiza89"
__license__ = "Apache-2.0"

__all__ = [
    "BILT",
    "Results",
    "list_variants",
    "get_variant_config",
    "VARIANT_CONFIGS",
    "set_logging_level",
]
