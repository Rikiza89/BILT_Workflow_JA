# BILT — Because I Like Twice

**BILT** is a lightweight object detection library built entirely on PyTorch.
It provides a clean, high-level API inspired by modern detection frameworks while remaining
fully independent — using an original FPN-based detection architecture rather than any
third-party detection model.

BILT automatically uses a CUDA GPU when one is available. No configuration needed — just
install a CUDA-enabled PyTorch build and BILT picks it up for both training and inference.

> **License:** Apache License 2.0
> **Copyright:** © 2026 Rikiza89

---

## Features

- **Five model sizes** — *spark / flash / core / pro / max* — each with a different backbone so you can choose the right speed/accuracy trade-off
- **Original architecture** — custom FPN neck + anchor-based detection head + focal loss; no dependency on proprietary detection code
- **Five backbone architectures** — MobileNetV2, MobileNetV3-S/L, ResNet-50/101 (via torchvision), all **initialised with ImageNet pretrained weights** for rapid convergence even on tiny datasets
- **Simple API** — `BILT("core")`, `.train()`, `.predict()`, `.evaluate()`, `.save()`, `.load()`
- **GPU-first** — automatically uses CUDA when available; falls back to CPU seamlessly
- **Pin-memory & non-blocking transfers** — DataLoader pins memory and tensor moves
  use `non_blocking=True` on CUDA, overlapping CPU→GPU transfer with the forward pass
- **Edge-friendly** — works on laptops, Raspberry Pi, and any CPU-only device
- **Compact checkpoints** — weights stored in float16, halving file size with no inference accuracy loss
- **Full training control** — every hyperparameter (LR, augmentation, loss, warmup) is exposed and overridable
- **Advanced training** — CIoU loss, Exponential Moving Average (EMA), LR warmup schedule, mosaic augmentation, image caching

---

## Model Variants

| Variant | Backbone | Input | FPN ch | File size (approx) | Best for |
|---------|----------|-------|--------|--------------------|----------|
| `spark` | MobileNetV2 | 320 px | 64 | ~9 MB | Embedded / real-time |
| `flash` | MobileNetV3-Small | 416 px | 96 | ~13 MB | Edge / fast inference |
| `core` | MobileNetV3-Large | 512 px | 128 | ~19 MB | General use (default) |
| `pro` | ResNet-50 | 640 px | 256 | ~55 MB | High accuracy |
| `max` | ResNet-101 | 640 px | 256 | ~95 MB | Maximum accuracy |

Short aliases are supported: `n / s / m / l / x` and `nano / small / medium / large / xlarge`.

```python
from bilt import BILT, list_variants
list_variants()   # prints the table above
```

---

## Installation

```bash
pip install torch torchvision pillow numpy pyyaml
pip install -e .          # development install from source
```

### Requirements

- Python 3.8+
- torch ≥ 1.10
- torchvision ≥ 0.11
- Pillow ≥ 8.0
- numpy ≥ 1.19
- pyyaml ≥ 5.4

---

## Quick Start

### Inference on a saved model

```python
from bilt import BILT

model = BILT("weights/best.pth")
detections = model.predict("image.jpg", conf=0.15)

for det in detections:
    print(f"{det['class_name']}: {det['score']:.2f}  bbox={det['bbox']}")
```

### Training from scratch

BILT auto-detects CUDA — no `device=` argument needed on a GPU machine.
All backbones start from **ImageNet pretrained weights**, so the model learns
meaningful features even from a handful of annotated images.

```python
from bilt import BILT

model = BILT("core")          # MobileNetV3-Large, 512 px

metrics = model.train(
    dataset="datasets/my_dataset",
    epochs=50,
    batch_size=4,
    learning_rate=2e-3,
)
print(metrics)
```

### Training with very few images

BILT is designed to produce working detectors from as few as **2–5 annotated images**
per class, thanks to pretrained backbones, augmentation, and tuned anchor matching.

```python
model = BILT("core")
metrics = model.train(
    dataset="datasets/my_dataset",
    epochs=50,
    batch_size=2,        # minimum for BatchNorm
)
```

### Choosing a different size

```python
model = BILT("spark")   # nano — fastest
model = BILT("flash")   # small
model = BILT("core")    # medium (default)
model = BILT("pro")     # large
model = BILT("max")     # xlarge — most accurate

# Short aliases also work
model = BILT("n")       # same as spark
model = BILT("m")       # same as core
model = BILT("x")       # same as max
```

### Advanced training — ResNet (pro / max)

ResNet models benefit significantly from advanced training features. Recommended settings:

```python
model = BILT("pro")   # or "max"

metrics = model.train(
    dataset="datasets/my_dataset",
    epochs=100,
    batch_size=4,
    learning_rate=1e-3,
    # LR warmup — ramp from 10% → 100% over first N epochs
    lr_warmup_epochs=5,
    # CIoU regression loss — better geometric alignment than Smooth-L1
    use_ciou=True,
    # Exponential Moving Average — smoother weights, better generalisation
    use_ema=True,
    ema_decay=0.99,        # auto-tuned to dataset size; this is the upper cap
    # Mosaic augmentation — strong regularisation for small datasets
    mosaic=True,
    mosaic_prob=0.5,
    # Image caching — load all training images into RAM once (fast GPUs)
    cache_images=True,
)
```

### Advanced training — MobileNet (spark / flash / core)

```python
model = BILT("core")

metrics = model.train(
    dataset="datasets/my_dataset",
    epochs=80,
    batch_size=8,
    learning_rate=2e-3,
    lr_warmup_epochs=3,
    use_ciou=True,
    use_ema=True,
    mosaic=True,
    mosaic_prob=0.5,
)
```

### Batch prediction with annotated images

```python
results = model.predict("images/", conf=0.15, return_images=True)
results.save("runs/detect/exp")   # saves annotated images
results.show()                    # displays with matplotlib
```

### Evaluation

```python
metrics = model.evaluate("datasets/my_dataset")
print(f"Avg detections/image: {metrics['avg_predictions_per_image']:.2f}")
```

---

## GPU Acceleration

### Automatic device selection

BILT checks `torch.cuda.is_available()` at startup in `Trainer`, `Inferencer`,
and `Evaluator`. If a CUDA GPU is present it is used automatically — you do not
need to pass `device="cuda"` anywhere.

```python
model = BILT("core")
model.train(dataset="data/", epochs=50)   # uses GPU if available
```

To verify which device is active:

```python
model = BILT("weights/best.pth")
print(model.device)    # cuda  or  cpu
```

### Forcing a specific device

```python
# Force CPU even on a GPU machine
model = BILT("core", device="cpu")
model.train(dataset="data/", device="cpu")

# Force a specific GPU
model = BILT("core", device="cuda:1")

# Apple Silicon
model = BILT("core", device="mps")
```

### What's optimised on CUDA

| Optimisation | Benefit |
|---|---|
| `pin_memory=True` on DataLoader | Faster CPU→GPU page-locked memory copy |
| `non_blocking=True` on `.to(device)` | GPU forward pass overlaps with next batch transfer |
| `persistent_workers=True` when `workers > 0` | Worker processes stay alive between epochs |

### CUDA install

Install the CUDA-enabled PyTorch wheel that matches your driver:

```bash
# CUDA 11.8
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# CUDA 12.1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

Verify CUDA is visible to PyTorch before running BILT:

```python
import torch
print(torch.cuda.is_available())   # True
print(torch.cuda.get_device_name(0))
```

---

## Dataset Format

BILT uses the standard normalised label format compatible with most annotation
tools (LabelImg, Roboflow, CVAT, etc.):

```
<class_id>  <x_center>  <y_center>  <width>  <height>
```

All five values are normalised to `[0, 1]` relative to the image size.

### Expected directory layout

```
dataset/
├── train/
│   ├── images/   *.jpg / *.png / …
│   └── labels/   *.txt
├── val/
│   ├── images/
│   └── labels/
└── data.yaml     (optional but recommended)
```

### data.yaml

```yaml
nc: 3
names: [cat, dog, person]
```

---

## Architecture

```
Input image (H×W)
      │
  ┌───▼────────────────────────┐
  │  BILTBackbone              │  MobileNet / ResNet (ImageNet pretrained)
  │  C3 (1/8)  C4 (1/16)  C5 (1/32)
  └───┬────────────────────────┘
      │
  ┌───▼────────────────────────┐
  │  FPNNeck  (original)       │
  │  P3    P4    P5    P6      │  all with fpn_channels
  └───┬────────────────────────┘
      │
  ┌───▼────────────────────────┐
  │  BILTHead  (original)      │  shared across all FPN levels
  │  cls_preds  +  box_preds   │  9 anchors/location (3 scales × 3 ratios)
  └───┬────────────────────────┘
      │
  ┌───▼────────────────────────┐    ┌──────────────────────────────┐
  │  Training                  │    │  Inference                   │
  │  Anchor matching           │    │  Box decode + per-class NMS  │
  │  Focal loss + Smooth-L1    │    │  Score filter + top-N cap    │
  │  or CIoU (optional)        │    │  Batch-GPU inference          │
  └────────────────────────────┘    └──────────────────────────────┘
```

**Training losses**

| Loss | Purpose | Enable with |
|------|---------|-------------|
| Focal loss (α=0.25, γ=2.0) | Classification — handles class imbalance | always active |
| Smooth-L1 | Bounding-box regression (default) | `use_ciou=False` |
| CIoU | Complete IoU — geometric penalties on centre distance and aspect ratio | `use_ciou=True` |

**BatchNorm behaviour during validation**

During validation, BILT sets all BatchNorm layers to `eval()` mode while keeping the
parent model in `train()` mode (required for the loss branch). This ensures that
`running_mean` / `running_var` statistics accumulated during training are never
overwritten by validation-batch statistics. The best checkpoint saved after each
validation epoch always reflects clean training-phase statistics — critical for
ResNet models (53+ BN layers) where stat corruption would otherwise prevent detections.

---

## API Reference

### `BILT(weights=None, device=None)`

| Argument | Description |
|----------|-------------|
| `weights` | Variant name (`"spark"` … `"max"`), a `.pth` checkpoint path, or `None` |
| `device`  | `"cpu"`, `"cuda"`, or `None` (auto) |

### `.predict(source, conf, iou, img_size, return_images, max_det)`

| Argument | Default | Description |
|----------|---------|-------------|
| `source` | — | File path, directory, PIL Image, numpy array, or list |
| `conf` | `0.15` | Minimum confidence score |
| `iou` | `0.45` | NMS IoU threshold |
| `img_size` | variant default | Override inference resolution |
| `return_images` | `False` | Return `Results` with annotated images |
| `max_det` | `300` | Maximum detections to return per image |

When `return_images=False` and a list of images is passed, `predict()` uses
GPU-batched inference (`detect_batch()`) for maximum throughput.

### `.train(dataset, epochs, batch_size, ...)`

| Argument | Default | Description |
|----------|---------|-------------|
| `dataset` | — | Path to dataset root |
| `epochs` | `50` | Training epochs |
| `batch_size` | `4` | Images per batch (min 2) |
| `img_size` | variant default | Training resolution |
| `learning_rate` | `2e-3` | AdamW learning rate for the detection head |
| `variant` | `"core"` | Override model variant for this run |
| `save_dir` | `"runs/train"` | Output directory |
| `name` | `"exp"` | Run sub-directory name |
| `workers` | `0` | DataLoader worker processes |
| `warmup_epochs` | `3` | Epochs with backbone frozen |
| `backbone_lr_mult` | `0.1` | Backbone LR = `learning_rate × backbone_lr_mult` |
| `weight_decay` | `1e-4` | AdamW weight decay |
| `cos_lr_min` | `1e-6` | Cosine annealing minimum LR |
| `grad_clip` | `5.0` | Gradient clipping max-norm (0 = disabled) |
| `focal_alpha` | `0.25` | Focal loss alpha |
| `focal_gamma` | `2.0` | Focal loss gamma |
| `box_loss_weight` | `1.0` | Regression loss weight |
| `augment` | `True` | Enable training augmentation |
| `flip_prob` | `0.5` | Random horizontal flip probability |
| `color_jitter` | `(0.4,0.4,0.4,0.1)` | Brightness/contrast/saturation/hue jitter |
| `lr_warmup_epochs` | `0` | Linear LR warmup epochs (0 = disabled). Ramps LR from 10% → 100% |
| `use_ciou` | `False` | Use CIoU regression loss instead of Smooth-L1 |
| `use_ema` | `False` | Enable Exponential Moving Average of model weights |
| `ema_decay` | `0.99` | EMA decay upper cap (auto-tuned down for small datasets) |
| `cache_images` | `False` | Cache all training images in RAM (fast when dataset fits in memory) |
| `mosaic` | `False` | Enable mosaic augmentation (4-image mosaic tiles) |
| `mosaic_prob` | `0.5` | Probability of applying mosaic to each batch |

### `.evaluate(dataset, batch_size, conf)`

Returns a dict with `total_images`, `total_predictions`, `total_ground_truth`,
`avg_predictions_per_image`, `avg_ground_truth_per_image`.

### `.save(path)` / `.load(weights)`

Checkpoints are stored in **float16** to halve disk usage. They are transparently
upcast back to float32 when loaded. The variant name, class names, and input size
are all included so a single file is sufficient to restore a complete model.

### `BILT.variants()`

Prints a summary of all five variants (static method, callable on the class).

---

## Legal

BILT is an original work by **Rikiza89**, released under the
**Apache License, Version 2.0**.

- The detection architecture (FPN neck, detection head, anchor matching,
  focal loss, smooth-L1) is written from scratch and is not derived from any
  other project.
- Backbone architectures (MobileNet, ResNet) are provided by **torchvision** (BSD/MIT licensed); weights are downloaded from torchvision's pretrained model hub (ImageNet) at first use.
- No code from any proprietary or copyleft-encumbered project is incorporated.

See [LICENSE](LICENSE) for the full license text.

---

## Roadmap

- [ ] mAP evaluation (COCO-style)
- [ ] ONNX and TensorRT export
- [ ] Mixed-precision (fp16) training
- [ ] Multi-GPU training
- [ ] Model export — ONNX weights from user-trained checkpoints
- [ ] Web demo (Gradio)
