# Training Guide

---

## Basic training

```python
from bilt import BILT

model = BILT("core")          # choose a variant
metrics = model.train(
    dataset="datasets/my_dataset",
    epochs=50,
    batch_size=4,
)
```

BILT automatically:
- Loads train and validation splits from `dataset/train/` and `dataset/val/`.
- Reads class names from `data.yaml` if present.
- Initialises the detector with **ImageNet pretrained backbone weights** — feature extraction starts from a strong base rather than random noise.
- **Selects the best available device** — CUDA GPU if present, otherwise CPU.
- Enables `pin_memory` and `non_blocking` transfers when on CUDA.
- Freezes the backbone for the first `warmup_epochs` (default: 3) so the detection head stabilises before full end-to-end training.
- After warmup, unfreezes the backbone with **differential learning rate** — backbone trains at `learning_rate × backbone_lr_mult` (default 10× lower than the head).
- Applies **data augmentation** (random horizontal flip + color jitter) to training images.
- Saves the best checkpoint (lowest validation loss) to `runs/train/exp/weights/best.pth`.
- Saves checkpoints in **float16** — roughly half the file size of float32 with no accuracy loss.

---

## train() parameters

```python
model.train(
    # Required
    dataset        = "datasets/my_dataset",

    # Basic
    epochs         = 50,
    batch_size     = 4,
    img_size       = None,          # None = use variant default
    learning_rate  = 2e-3,
    device         = None,          # None = auto (cuda if available, else cpu)
    save_dir       = "runs/train",  # parent directory for run outputs
    name           = "exp",         # sub-directory name (auto-incremented)
    variant        = None,          # override the variant for this run
    workers        = 0,             # dataloader worker processes

    # Training loop
    warmup_epochs    = 3,           # epochs with backbone frozen
    backbone_lr_mult = 0.1,         # backbone LR = learning_rate * backbone_lr_mult
    weight_decay     = 1e-4,        # AdamW weight decay
    cos_lr_min       = 1e-6,        # cosine annealing minimum LR
    grad_clip        = 5.0,         # gradient clipping max-norm (0 = disabled)

    # Loss
    focal_alpha      = 0.25,        # focal loss class-balance weight
    focal_gamma      = 2.0,         # focal loss focusing strength
    box_loss_weight  = 1.0,         # regression loss weight

    # Augmentation
    augment          = True,        # enable/disable training augmentation
    flip_prob        = 0.5,         # random horizontal flip probability
    color_jitter     = (0.4, 0.4, 0.4, 0.1),  # (brightness, contrast, saturation, hue)
    mosaic           = False,       # 4-image mosaic augmentation
    mosaic_prob      = 0.5,         # probability of applying mosaic per batch
    cache_images     = False,       # pre-load all training images into RAM

    # Advanced training
    lr_warmup_epochs = 0,           # linear LR ramp from 10%→100% over N epochs (0 = off)
    use_ciou         = False,       # CIoU regression loss instead of Smooth-L1
    use_ema          = False,       # Exponential Moving Average of weights
    ema_decay        = 0.99,        # EMA decay upper cap (auto-tuned to dataset size)
)
```

### Return value

`train()` returns a dict:

```python
{
    "variant":          "core",
    "num_epochs":       50,
    "final_train_loss": 0.0612,
    "final_val_loss":   0.0731,
    "best_val_loss":    0.0473,
    "training_time":    423.1,      # seconds
    "model_path":       "runs/train/exp/weights/best.pth",
}
```

---

## Choosing hyperparameters

### Batch size

| Hardware | Recommended batch size |
|----------|------------------------|
| CPU (8 GB RAM) | 2–4 |
| GPU 4 GB VRAM | 4–8 (`spark`/`flash`) |
| GPU 8 GB VRAM | 8–16 (`core`) or 4–8 (`pro`/`max`) |
| GPU 16+ GB | 16–32 |

Batch size must be ≥ 2 (required for BatchNorm in the backbone). BILT
automatically clamps it to 2 if a lower value is passed.

### Learning rate

The default `2e-3` works well with the pretrained backbone + warmup strategy.
This is the learning rate applied to the **detection head**; the backbone gets
`learning_rate × backbone_lr_mult` (default `2e-4`).

```python
# Default — recommended starting point
model.train(dataset="data/", learning_rate=2e-3)

# More conservative — use if training is unstable
model.train(dataset="data/", learning_rate=5e-4)

# Adjust backbone learning relative to head (default 10×)
model.train(dataset="data/", learning_rate=2e-3, backbone_lr_mult=0.05)
```

### Epochs

| Dataset size | Recommended epochs |
|-------------|-------------------|
| < 10 images | 50–100 |
| 10–100 images | 50–100 |
| 100–500 images | 50–80 |
| 500–2000 images | 80–150 |
| > 2000 images | 50–100 (large datasets converge faster) |

### Warmup epochs

There are two independent warmup mechanisms — it is important not to confuse them:

| Parameter | What it controls |
|-----------|-----------------|
| `warmup_epochs` | Epochs the **backbone is frozen**. Head trains alone; after this the backbone unfreezes at `backbone_lr_mult` × LR. Default: `3`. |
| `lr_warmup_epochs` | Epochs over which the **learning rate ramps** from 10% → 100% of its target value (LambdaLR schedule). Default: `0` (disabled). |

Both can be used together or independently.

```python
# Only backbone freeze warmup (default)
model.train(dataset="data/", warmup_epochs=3)

# Only LR ramp warmup
model.train(dataset="data/", warmup_epochs=0, lr_warmup_epochs=5)

# Both combined — recommended for ResNet models
model.train(dataset="data/", warmup_epochs=3, lr_warmup_epochs=5)

# Skip all warmup
model.train(dataset="data/", warmup_epochs=0, lr_warmup_epochs=0)
```

### Image size

```python
# Use the variant's default (recommended)
model.train(dataset="data/")

# Override — smaller = faster, larger = better small-object detection
model.train(dataset="data/", img_size=320)
model.train(dataset="data/", img_size=640)
```

Image size must be divisible by 32. Common choices: 320, 416, 512, 640.

---

## Augmentation

Training augmentation is **enabled by default** and consists of:

1. **Random horizontal flip** — mirrors the image and corrects box coordinates.
   Controlled by `flip_prob` (default 0.5).
2. **Color jitter** — randomly varies brightness, contrast, saturation, and hue.
   Controlled by `color_jitter=(brightness, contrast, saturation, hue)`.

```python
# Default augmentation
model.train(dataset="data/", augment=True, flip_prob=0.5,
            color_jitter=(0.4, 0.4, 0.4, 0.1))

# Disable color jitter only
model.train(dataset="data/", color_jitter=None)

# Disable all augmentation
model.train(dataset="data/", augment=False)

# Stronger augmentation for small datasets
model.train(dataset="data/", flip_prob=0.5,
            color_jitter=(0.6, 0.6, 0.6, 0.15))
```

Augmentation is automatically disabled for the validation split regardless of
the `augment` setting.

---

## Device selection

BILT checks `torch.cuda.is_available()` inside `Trainer` and uses the GPU
automatically. You only need to pass `device=` when you want to override this.

```python
# Auto-detect — uses CUDA if available, otherwise CPU (recommended)
model = BILT("core")
model.train(dataset="data/")

# Force a specific GPU
model.train(dataset="data/", device="cuda:1")

# Force CPU even on a GPU machine
model.train(dataset="data/", device="cpu")

# Apple Silicon
model.train(dataset="data/", device="mps")
```

You can check which device was selected by looking at the training log:

```
INFO  Trainer using device: cuda
```

Or after training:

```python
print(model.device)   # cuda  or  cpu
```

### GPU performance tips

| Tip | Detail |
|-----|--------|
| Larger batch size | GPU throughput scales well — try 16 or 32 on a 8+ GB card |
| DataLoader workers | Use `workers=2` or `workers=4` on multi-core Linux/macOS |
| Larger variant | `pro` and `max` benefit most from GPU; `spark`/`flash` are already fast on CPU |

---

## Training output structure

```
runs/
└── train/
    ├── exp/                     ← first run
    │   └── weights/
    │       └── best.pth         ← best checkpoint (lowest val loss, float16)
    ├── exp1/                    ← second run (auto-incremented)
    │   └── weights/
    │       └── best.pth
    └── my_experiment/           ← custom name
        └── weights/
            └── best.pth
```

### Custom run names

```python
model.train(
    dataset="data/",
    save_dir="experiments",
    name="resnet50_v2",
)
# saves to experiments/resnet50_v2/weights/best.pth
```

---

## Training callbacks

Use callbacks to monitor training progress, send notifications, or implement
early stopping:

```python
def my_callback(info):
    epoch       = info["epoch"]
    total       = info["total_epochs"]
    train_loss  = info["train_loss"]
    val_loss    = info["val_loss"]
    lr          = info["lr"]

    pct = epoch / total * 100
    print(f"[{pct:5.1f}%] E{epoch}/{total}  "
          f"train={train_loss:.4f}  val={val_loss:.4f}  lr={lr:.2e}")

    # Example: early stopping
    if val_loss < 0.05:
        raise StopIteration("Target reached!")

from bilt.trainer import Trainer
trainer = Trainer(
    dataset_path="data/",
    num_classes=3,
    class_names=["cat", "dog", "person"],
    variant="core",
)
trainer.train(save_path="weights/best.pth", callback=my_callback)
```

---

## Resuming training

BILT does not have built-in resume support, but you can load a checkpoint
and continue training by implementing a custom loop with `Trainer`:

```python
from bilt import BILT

# Load the existing model
model = BILT("runs/train/exp/weights/best.pth")

# Re-train with more epochs (starts from the loaded weights)
metrics = model.train(
    dataset="data/",
    epochs=50,
    name="exp_resumed",
)
```

Note: the optimiser state is not saved in the checkpoint, so the learning
rate schedule restarts from epoch 0. For precise resume behaviour, use the
`Trainer` class directly and manage state yourself.

---

## Multi-variant comparison

```python
from bilt import BILT

results = {}
for variant in ["spark", "core", "pro"]:
    m = BILT(variant)
    r = m.train(
        dataset="data/",
        epochs=50,
        save_dir="runs/compare",
        name=variant,
    )
    results[variant] = r["best_val_loss"]
    print(f"{variant}: {r['best_val_loss']:.4f}")

best = min(results, key=results.get)
print(f"\nBest variant: {best}")
```

---

## Training on a small dataset

BILT is specifically designed to produce working detectors from **very few images**,
including as few as 2–5 images per class. The following features make this possible:

- **ImageNet pretrained backbones** — the model already understands visual features
- **Augmentation** — flip and color jitter multiply effective training data
- **9 anchors per location** (3 octave scales × 3 aspect ratios) — covers a wide range of object sizes
- **Lowered anchor matching thresholds** — IoU ≥ 0.35 for positive matches (industry default is 0.5)
- **Higher default LR** (2e-3) — faster convergence on small datasets
- **Backbone warmup** — head stabilises before pretrained features are touched

Practical recommendations for small datasets:

1. **Use `core` or smaller** — fewer parameters, less overfitting risk.
2. **Use default warmup** (`warmup_epochs=3`) or increase to 5–10 for very small sets.
3. **Do not decrease LR** — 2e-3 is already tuned for few-shot scenarios.
4. **Run enough epochs** — 50 is a good default; go up to 100–200 if validation loss
   is still improving.

```python
model = BILT("core")
model.train(
    dataset="data/",
    epochs=100,
    batch_size=2,
    warmup_epochs=5,
)
```

Healthy loss values (epoch 30–50 with pretrained backbone):

| Metric | Target range |
|--------|-------------|
| train_loss | 0.03 – 0.10 |
| val_loss | 0.04 – 0.12 |

---

## Training loss components

The training log prints three loss values each batch:

```
Epoch 3/50  batch 0/1  loss=0.0432  cls=0.0214  box=0.0218
```

| Component | Description |
|-----------|-------------|
| `loss` | Total loss (cls + box) |
| `cls` | Focal classification loss |
| `box` | Smooth-L1 regression loss |

Both losses are normalised by the number of positive anchors so the scale
remains roughly constant regardless of batch size.

---

## Using the Trainer directly

For full control, bypass `BILT.train()` and use `Trainer` directly:

```python
from bilt.trainer import Trainer
from pathlib import Path

trainer = Trainer(
    dataset_path=Path("data/"),
    num_classes=3,
    class_names=["cat", "dog", "person"],
    batch_size=8,
    learning_rate=2e-3,
    num_epochs=50,
    input_size=512,
    device="cuda",
    variant="core",
    warmup_epochs=3,
    backbone_lr_mult=0.1,
    weight_decay=1e-4,
    cos_lr_min=1e-6,
    grad_clip=5.0,
    focal_alpha=0.25,
    focal_gamma=2.0,
    box_loss_weight=1.0,
    augment=True,
    flip_prob=0.5,
    color_jitter=(0.4, 0.4, 0.4, 0.1),
    mosaic=False,
    mosaic_prob=0.5,
    cache_images=False,
    lr_warmup_epochs=0,
    use_ciou=False,
    use_ema=False,
    ema_decay=0.99,
)

# Train one epoch at a time
for epoch in range(trainer.num_epochs):
    train_loss = trainer.train_one_epoch()
    val_loss   = trainer.validate()
    print(f"E{epoch+1}: train={train_loss:.4f}  val={val_loss:.4f}")

trainer.detection_model.save(Path("weights/final.pth"), trainer.class_names)
```
