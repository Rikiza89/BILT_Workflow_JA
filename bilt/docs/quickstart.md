# Quick Start

Get up and running in five minutes.

---

## 1. Install

```bash
git clone https://github.com/Rikiza89/bilt.git
cd bilt
pip install -e .
```

---

## 2. Check available model sizes

```python
from bilt import BILT

BILT.variants()
```

```
Variant   Backbone                Input   FPN ch   Description
---------------------------------------------------------------------------
spark     mobilenet_v2            320     64       Nano - fastest inference
flash     mobilenet_v3_small      416     96       Small - fast with good accuracy
core      mobilenet_v3_large      512     128      Medium - balanced speed and accuracy
pro       resnet50                640     256      Large - high accuracy
max       resnet101               640     256      XLarge - maximum accuracy
```

All backbones start from **ImageNet pretrained weights** — no training from
scratch. This means BILT can learn to detect objects from **very few images**.

---

## 3. Train on your dataset

```python
from bilt import BILT

model = BILT("core")                # pick a variant

metrics = model.train(
    dataset="datasets/my_dataset",  # path to your data
    epochs=50,
    batch_size=4,
)

print(f"Best val loss : {metrics['best_val_loss']:.4f}")
print(f"Model saved to: {metrics['model_path']}")
```

The model is automatically saved to `runs/train/exp/weights/best.pth`.

> **Dataset layout** — see [Dataset Preparation](dataset.md) for how to
> structure your data and annotation files.

---

## 4. Run inference

```python
from bilt import BILT

model = BILT("runs/train/exp/weights/best.pth")

# Single image → list of detections
detections = model.predict("photo.jpg", conf=0.15)

for det in detections:
    print(det["class_name"], det["score"], det["bbox"])
```

Each detection is a dict:

```python
{
    "class_name": "cat",
    "class_id":   1,
    "score":      0.87,
    "bbox":       [x1, y1, x2, y2]   # absolute pixel coordinates
}
```

---

## 5. Save annotated images

```python
results = model.predict("images/", conf=0.15, return_images=True)
results.save("runs/detect/exp")
```

---

## 6. Evaluate

```python
metrics = model.evaluate("datasets/my_dataset")
print(metrics)
```

---

## Training with very few images

BILT is designed to work even with just **2–5 annotated images per class**.
The pretrained backbone, augmentation pipeline, and tuned anchor matching
make this possible:

```python
model = BILT("core")
metrics = model.train(
    dataset="datasets/my_dataset",
    epochs=50,
    batch_size=2,        # minimum — fine for 2–5 images
)
```

Healthy loss values after 30–50 epochs with a pretrained backbone:
- `train_loss`: 0.03 – 0.10
- `val_loss`: 0.04 – 0.12

---

## Advanced training options

All training hyperparameters are exposed:

```python
model.train(
    dataset="data/",
    epochs=50,
    learning_rate=2e-3,       # head LR; backbone gets 10× less by default
    warmup_epochs=3,           # epochs with backbone frozen
    backbone_lr_mult=0.1,      # backbone LR = learning_rate * backbone_lr_mult
    weight_decay=1e-4,
    grad_clip=5.0,
    focal_alpha=0.25,
    focal_gamma=2.0,
    augment=True,
    flip_prob=0.5,
    color_jitter=(0.4, 0.4, 0.4, 0.1),
)
```

See [Training Guide](training.md) for the full parameter reference.

---

## What next?

| Topic | Guide |
|-------|-------|
| Choosing the right variant | [Model Variants](variants.md) |
| Preparing your dataset | [Dataset Preparation](dataset.md) |
| Training options and tips | [Training Guide](training.md) |
| Inference options | [Inference Guide](inference.md) |
| Full API docs | [API Reference](api.md) |
| Architecture details | [Architecture](architecture.md) |
