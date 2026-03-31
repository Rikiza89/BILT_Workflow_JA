# BILT Documentation

Welcome to the BILT documentation.
BILT is a lightweight, CPU-friendly object detection library built on PyTorch
with a fully original FPN-based detection architecture and **ImageNet pretrained
backbones** for fast convergence even on tiny datasets.

---

## Contents

| Guide | Description |
|-------|-------------|
| [Installation](installation.md) | System requirements, pip install, platform notes |
| [Quick Start](quickstart.md) | Get from zero to a running model in 5 minutes |
| [Model Variants](variants.md) | spark / flash / core / pro / max — when to use each |
| [Dataset Preparation](dataset.md) | Label format, directory layout, annotation tools |
| [Training Guide](training.md) | Hyperparameters, augmentation, callbacks, Trainer API |
| [Inference Guide](inference.md) | All input types, thresholds, max_det, batch processing |
| [API Reference](api.md) | Complete reference for every public class and function |
| [Architecture](architecture.md) | Backbone, FPN, detection head, anchor system, losses |

---

## At a glance

```python
from bilt import BILT

# Train — backbones start from ImageNet pretrained weights
model = BILT("core")
model.train(dataset="data/", epochs=50)

# Predict
detections = model.predict("image.jpg", conf=0.15)

# Annotated output
results = model.predict("images/", return_images=True)
results.save("runs/detect/exp")
```

Model sizes:

| Variant | Backbone | Input | File size | Use case |
|---------|----------|-------|-----------|----------|
| `spark` | MobileNetV2 | 320 px | ~9 MB | Edge / real-time |
| `flash` | MobileNetV3-Small | 416 px | ~13 MB | Fast inference |
| `core` | MobileNetV3-Large | 512 px | ~19 MB | General (default) |
| `pro` | ResNet-50 | 640 px | ~55 MB | High accuracy |
| `max` | ResNet-101 | 640 px | ~95 MB | Maximum accuracy |

File sizes reflect float16 checkpoint storage (half the size of float32).

---

## Key features

- **ImageNet pretrained backbones** — converges from very few images (2–5 per class)
- **9 anchors per location** (3 octave scales × 3 aspect ratios) — broad coverage
- **Data augmentation** — random flip + color jitter, bbox-safe
- **Compact checkpoints** — float16 storage, transparent float32 on load
- **Full hyperparameter control** — every training knob exposed via `BILT.train()`
- **Differential learning rate** — backbone at 10× lower LR than head after warmup

---

## License

BILT is copyright © 2026 Rikiza89 and released under the
[Apache License, Version 2.0](../LICENSE).

The detection architecture (FPN neck, detection head, anchor system, focal
loss) is written from scratch. No code from any proprietary or
copyleft-encumbered project is incorporated.
