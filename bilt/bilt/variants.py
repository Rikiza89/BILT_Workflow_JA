# BILT (Because I Like Twice) - A PyTorch-based object detection library
# Copyright (C) 2026 Rikiza89
# Licensed under the Apache License, Version 2.0

"""
BILT model variant configurations.

Each variant defines a different trade-off between speed and accuracy,
using a distinct backbone architecture:

    spark  - nano    : MobileNetV2      | 320px  | fastest, minimal memory
    flash  - small   : MobileNetV3-S    | 416px  | fast with reasonable accuracy
    core   - medium  : MobileNetV3-L    | 512px  | balanced speed / accuracy
    pro    - large   : ResNet-50        | 640px  | high accuracy
    max    - xlarge  : ResNet-101       | 640px  | maximum accuracy
"""

from typing import Dict, Any

# ---------------------------------------------------------------------------
# Variant configurations
# ---------------------------------------------------------------------------

VARIANT_CONFIGS: Dict[str, Dict[str, Any]] = {
    "spark": {
        "backbone": "mobilenet_v2",
        "input_size": 320,
        "fpn_channels": 64,
        "head_num_convs": 2,
        "anchor_sizes": [32, 64, 128, 256],       # per FPN level P3–P6
        "anchor_scales": (1.0, 1.26, 1.587),      # octave scales ×1, ×∛2, ×∛4
        "anchor_aspect_ratios": (0.5, 1.0, 2.0),
        "description": "Nano - fastest inference, minimal memory footprint",
    },
    "flash": {
        "backbone": "mobilenet_v3_small",
        "input_size": 416,
        "fpn_channels": 96,
        "head_num_convs": 3,
        "anchor_sizes": [32, 64, 128, 256],
        "anchor_scales": (1.0, 1.26, 1.587),
        "anchor_aspect_ratios": (0.5, 1.0, 2.0),
        "description": "Small - fast with good accuracy",
    },
    "core": {
        "backbone": "mobilenet_v3_large",
        "input_size": 512,
        "fpn_channels": 128,
        "head_num_convs": 3,
        "anchor_sizes": [32, 64, 128, 256],
        "anchor_scales": (1.0, 1.26, 1.587),
        "anchor_aspect_ratios": (0.5, 1.0, 2.0),
        "description": "Medium - balanced speed and accuracy",
    },
    "pro": {
        "backbone": "resnet50",
        "input_size": 640,
        "fpn_channels": 256,
        "head_num_convs": 4,
        # Anchor sizes scaled 2× relative to spark (320px → 640px) so that
        # the anchor-to-image coverage ratio remains the same across variants.
        "anchor_sizes": [64, 128, 256, 512],
        "anchor_scales": (1.0, 1.26, 1.587),
        "anchor_aspect_ratios": (0.5, 1.0, 2.0),
        "description": "Large - high accuracy",
    },
    "max": {
        "backbone": "resnet101",
        "input_size": 640,
        "fpn_channels": 256,
        "head_num_convs": 4,
        # Same 2× scaling as pro — both use 640px input.
        "anchor_sizes": [64, 128, 256, 512],
        "anchor_scales": (1.0, 1.26, 1.587),
        "anchor_aspect_ratios": (0.5, 1.0, 2.0),
        "description": "XLarge - maximum accuracy",
    },
}

# Short alias → canonical name
VARIANT_ALIASES: Dict[str, str] = {
    "nano":   "spark",
    "n":      "spark",
    "small":  "flash",
    "s":      "flash",
    "medium": "core",
    "m":      "core",
    "large":  "pro",
    "l":      "pro",
    "xlarge": "max",
    "x":      "max",
}

DEFAULT_VARIANT = "core"


def get_variant_config(name: str) -> Dict[str, Any]:
    """
    Return a copy of the configuration dict for *name*.

    Accepts canonical names (spark / flash / core / pro / max) and
    short aliases (n / s / m / l / x, nano / small / medium / large / xlarge).

    Raises
    ------
    ValueError
        If *name* is not a recognised variant or alias.
    """
    key = name.lower().strip()
    key = VARIANT_ALIASES.get(key, key)
    if key not in VARIANT_CONFIGS:
        available = list(VARIANT_CONFIGS.keys())
        raise ValueError(
            f"Unknown BILT variant '{name}'. "
            f"Available variants: {available} "
            f"(aliases: {list(VARIANT_ALIASES.keys())})"
        )
    return dict(VARIANT_CONFIGS[key])


def is_variant_name(value: str) -> bool:
    """Return True if *value* is a recognised variant name or alias."""
    key = value.lower().strip()
    return key in VARIANT_CONFIGS or key in VARIANT_ALIASES


def list_variants() -> None:
    """Print a summary table of all available variants."""
    print(f"\n{'Variant':<8}  {'Backbone':<22}  {'Input':<6}  {'FPN ch':<7}  Description")
    print("-" * 75)
    for name, cfg in VARIANT_CONFIGS.items():
        print(
            f"{name:<8}  {cfg['backbone']:<22}  {cfg['input_size']:<6}  "
            f"{cfg['fpn_channels']:<7}  {cfg['description']}"
        )
    print()
