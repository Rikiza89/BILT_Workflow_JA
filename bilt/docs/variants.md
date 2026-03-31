# Model Variants

BILT provides five model sizes. Each uses a **different backbone architecture**,
so the trade-offs are not just about parameter count — they differ in feature
quality, receptive field, and computational characteristics.

All backbones are initialised with **ImageNet pretrained weights**, so every variant
starts with strong visual features rather than random noise.

---

## Overview

| Variant | Backbone | Input | Params (approx) | File size (fp16) | Best for |
|---------|----------|-------|-----------------|------------------|----------|
| `spark` | MobileNetV2 | 320 px | ~4 M | ~9 MB | Embedded, real-time, Raspberry Pi |
| `flash` | MobileNetV3-Small | 416 px | ~5 M | ~13 MB | Edge devices, fast inference |
| `core` | MobileNetV3-Large | 512 px | ~8 M | ~19 MB | General purpose (default) |
| `pro` | ResNet-50 | 640 px | ~30 M | ~55 MB | High-accuracy production |
| `max` | ResNet-101 | 640 px | ~50 M | ~95 MB | Maximum accuracy |

File sizes shown are for saved checkpoints (weights stored in float16).

---

## Selecting a variant

### By name

```python
from bilt import BILT

model = BILT("spark")   # nano
model = BILT("flash")   # small
model = BILT("core")    # medium  ← default
model = BILT("pro")     # large
model = BILT("max")     # xlarge
```

### Short aliases

All of these are equivalent pairs:

```python
BILT("n")      == BILT("nano")   == BILT("spark")
BILT("s")      == BILT("small")  == BILT("flash")
BILT("m")      == BILT("medium") == BILT("core")
BILT("l")      == BILT("large")  == BILT("pro")
BILT("x")      == BILT("xlarge") == BILT("max")
```

### Overriding during training

```python
model = BILT()                        # no variant yet
model.train(dataset="data/", variant="pro", epochs=50)

# or
model = BILT("pro")
model.train(dataset="data/", epochs=50)
```

---

## spark — nano

```python
model = BILT("spark")
```

**Backbone:** MobileNetV2 (ImageNet pretrained)
**Input:** 320 × 320
**FPN channels:** 64
**Detection head convolutions:** 2
**Checkpoint size:** ~9 MB

MobileNetV2 uses inverted residual blocks with linear bottlenecks. It is
extremely efficient on CPU and edge hardware.  Choose `spark` when:

- You need real-time inference on a Raspberry Pi or mobile CPU.
- Memory is very limited (< 1 GB RAM).
- Speed matters more than accuracy (small objects may be missed).
- You are prototyping and want fast iteration.

Typical training command:

```python
model = BILT("spark")
model.train(
    dataset="data/",
    epochs=50,
    batch_size=4,
    img_size=320,
)
```

---

## flash — small

```python
model = BILT("flash")
```

**Backbone:** MobileNetV3-Small (ImageNet pretrained)
**Input:** 416 × 416
**FPN channels:** 96
**Detection head convolutions:** 3
**Checkpoint size:** ~13 MB

MobileNetV3-Small uses hardware-aware neural architecture search (NAS) and
adds squeeze-and-excitation modules for improved accuracy at the same FLOP
budget. Choose `flash` when:

- You need a step up in accuracy from `spark` with minimal extra cost.
- Target platform is a modern ARM SoC (Cortex-A55/A78 class).
- You want a good default for edge deployment.

```python
model = BILT("flash")
model.train(dataset="data/", epochs=50, batch_size=4)
```

---

## core — medium (default)

```python
model = BILT("core")   # or simply BILT()  then train(variant="core")
```

**Backbone:** MobileNetV3-Large (ImageNet pretrained)
**Input:** 512 × 512
**FPN channels:** 128
**Detection head convolutions:** 3
**Checkpoint size:** ~19 MB

MobileNetV3-Large is the recommended starting point for most projects. It
provides a good balance of speed and accuracy, runs comfortably on a laptop
CPU, and converges reliably. Choose `core` when:

- You are unsure which size to use.
- You are running on a laptop or a mid-range server CPU.
- You want a good baseline before deciding whether to go smaller or larger.

```python
model = BILT("core")
model.train(dataset="data/", epochs=50, batch_size=4)
```

---

## pro — large

```python
model = BILT("pro")
```

**Backbone:** ResNet-50 (ImageNet pretrained)
**Input:** 640 × 640
**FPN channels:** 256
**Detection head convolutions:** 4
**Checkpoint size:** ~55 MB

ResNet-50 offers substantially richer feature representations than MobileNet
due to its greater depth and width. Choose `pro` when:

- Accuracy is more important than speed.
- You have a GPU or a powerful multi-core CPU.
- Your dataset has fine-grained categories or small objects.
- You need a production-quality detector.

```python
model = BILT("pro")
model.train(
    dataset="data/",
    epochs=50,
    batch_size=8,
    device="cuda",    # recommended
)
```

---

## max — xlarge

```python
model = BILT("max")
```

**Backbone:** ResNet-101 (ImageNet pretrained)
**Input:** 640 × 640
**FPN channels:** 256
**Detection head convolutions:** 4
**Checkpoint size:** ~95 MB

ResNet-101 is the largest backbone available in BILT. The extra depth provides
the most powerful feature extraction, at the cost of slower training and
inference. Choose `max` when:

- You are building a high-accuracy offline pipeline (e.g., batch processing).
- Training time is not a concern.
- You have ≥ 8 GB VRAM.

```python
model = BILT("max")
model.train(
    dataset="data/",
    epochs=100,
    batch_size=4,
    device="cuda",
)
```

---

## Comparing variants programmatically

```python
from bilt import get_variant_config, VARIANT_CONFIGS

# Full configuration dict for a single variant
cfg = get_variant_config("pro")
print(cfg["backbone"])          # resnet50
print(cfg["input_size"])        # 640
print(cfg["fpn_channels"])      # 256
print(cfg["anchor_scales"])     # (1.0, 1.26, 1.587)

# Iterate over all variants
for name, cfg in VARIANT_CONFIGS.items():
    print(f"{name}: {cfg['backbone']} @ {cfg['input_size']}px")
```

---

## Checkpoint files include the variant

When you save a model, the variant name is stored in the checkpoint. Loading
the checkpoint automatically restores the correct architecture — you do not
need to specify the variant again:

```python
model = BILT("pro")
model.train(dataset="data/", epochs=50)
model.save("my_pro_model.pth")

# Later, on any machine:
model2 = BILT("my_pro_model.pth")   # automatically uses ResNet-50 architecture
print(model2.variant)               # pro
```

Checkpoints are saved in **float16** to reduce file size. They are
automatically upcast back to float32 on load.

---

## Switching variants after training

Each variant produces an independent checkpoint. You can train multiple
variants on the same dataset and compare them:

```python
from bilt import BILT

for variant in ["spark", "core", "pro"]:
    m = BILT(variant)
    results = m.train(
        dataset="data/",
        epochs=50,
        save_dir="runs/train",
        name=f"compare_{variant}",
    )
    print(f"{variant}: val_loss={results['best_val_loss']:.4f}")
```
