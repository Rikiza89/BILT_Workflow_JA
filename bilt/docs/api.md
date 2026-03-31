# API Reference

Complete reference for all public classes and functions.

---

## `bilt.BILT`

Main interface for object detection.

```python
from bilt import BILT

model = BILT(weights=None, device=None)
```

### Constructor

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `weights` | str \| Path \| None | `None` | Variant name (`"spark"`, `"flash"`, `"core"`, `"pro"`, `"max"`, or aliases), a `.pth` checkpoint path, or `None` |
| `device` | str \| None | `None` | `"cpu"`, `"cuda"`, `"cuda:0"`, `"mps"`, or `None` for auto-detect |

**Examples:**

```python
BILT()                          # uninitialized, default device
BILT("core")                    # medium variant, ImageNet pretrained backbone
BILT("m")                       # same as above (alias)
BILT("runs/train/exp/best.pth") # load saved model
BILT("best.pth", device="cuda") # load on GPU
```

---

### `.train()`

```python
metrics = model.train(
    dataset,

    # Basic
    epochs         = 50,
    batch_size     = 4,
    img_size       = None,
    learning_rate  = 2e-3,
    device         = None,
    save_dir       = "runs/train",
    name           = "exp",
    variant        = None,
    workers        = 0,

    # Training loop
    warmup_epochs    = 3,
    backbone_lr_mult = 0.1,
    weight_decay     = 1e-4,
    cos_lr_min       = 1e-6,
    grad_clip        = 5.0,

    # Loss
    focal_alpha      = 0.25,
    focal_gamma      = 2.0,
    box_loss_weight  = 1.0,

    # Augmentation
    augment          = True,
    flip_prob        = 0.5,
    color_jitter     = (0.4, 0.4, 0.4, 0.1),
    mosaic           = False,
    mosaic_prob      = 0.5,
    cache_images     = False,

    # Advanced training
    lr_warmup_epochs = 0,
    use_ciou         = False,
    use_ema          = False,
    ema_decay        = 0.99,
)
```

Train a detector on the user's dataset. The backbone starts from ImageNet
pretrained weights. The backbone is frozen for `warmup_epochs` epochs so the
head can stabilise; after warmup, the backbone is unfrozen and trained at
`learning_rate × backbone_lr_mult`.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `dataset` | str \| Path | **required** | Root directory containing `train/` and `val/` |
| `epochs` | int | `50` | Total training epochs |
| `batch_size` | int | `4` | Images per batch (minimum 2) |
| `img_size` | int \| None | `None` | Square input resolution; `None` uses the variant's default |
| `learning_rate` | float | `2e-3` | Initial AdamW learning rate for the detection head |
| `device` | str \| None | `None` | Override device for this run |
| `save_dir` | str \| Path | `"runs/train"` | Parent directory for outputs |
| `name` | str | `"exp"` | Sub-directory name (auto-incremented if it already exists) |
| `variant` | str \| None | `None` | Override the variant name for this run |
| `workers` | int | `0` | DataLoader worker processes |
| `warmup_epochs` | int | `3` | Epochs to keep backbone frozen (0 = no warmup) |
| `backbone_lr_mult` | float | `0.1` | Backbone LR multiplier relative to head LR |
| `weight_decay` | float | `1e-4` | AdamW weight decay |
| `cos_lr_min` | float | `1e-6` | Cosine annealing minimum learning rate |
| `grad_clip` | float | `5.0` | Gradient clipping max-norm (0 = disabled) |
| `focal_alpha` | float | `0.25` | Focal loss alpha (class-balance weight) |
| `focal_gamma` | float | `2.0` | Focal loss gamma (focusing strength) |
| `box_loss_weight` | float | `1.0` | Regression loss weight relative to classification |
| `augment` | bool | `True` | Enable training augmentation |
| `flip_prob` | float | `0.5` | Random horizontal flip probability (0–1) |
| `color_jitter` | tuple \| None | `(0.4, 0.4, 0.4, 0.1)` | (brightness, contrast, saturation, hue) jitter, or `None` to disable |
| `mosaic` | bool | `False` | Enable 4-image mosaic augmentation |
| `mosaic_prob` | float | `0.5` | Probability of applying mosaic per batch |
| `cache_images` | bool | `False` | Pre-load all training images into RAM |
| `lr_warmup_epochs` | int | `0` | Linear LR ramp from 10%→100% over N epochs (0 = disabled). Independent of backbone `warmup_epochs` |
| `use_ciou` | bool | `False` | Use CIoU regression loss instead of Smooth-L1 |
| `use_ema` | bool | `False` | Enable Exponential Moving Average of model weights |
| `ema_decay` | float | `0.99` | EMA decay upper cap (auto-tuned down for small datasets) |

**Returns:** dict with keys:

| Key | Type | Description |
|-----|------|-------------|
| `variant` | str | Variant name used |
| `num_epochs` | int | Total epochs trained |
| `final_train_loss` | float | Training loss at the last epoch |
| `final_val_loss` | float | Validation loss at the last epoch |
| `best_val_loss` | float | Lowest validation loss achieved |
| `training_time` | float | Total training time in seconds |
| `model_path` | str | Path to the saved best checkpoint |

---

### `.predict()`

```python
result = model.predict(
    source,
    conf=0.15,
    iou=0.45,
    img_size=None,
    return_images=False,
    max_det=300,
)
```

Run object detection on one or more images.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `source` | str \| Path \| PIL.Image \| np.ndarray \| list | **required** | Input image(s) |
| `conf` | float | `0.15` | Minimum confidence threshold in [0, 1] |
| `iou` | float | `0.45` | NMS IoU threshold in [0, 1] |
| `img_size` | int \| None | `None` | Override inference resolution |
| `return_images` | bool | `False` | Return a `Results` object with annotated images |
| `max_det` | int | `300` | Maximum detections to return per image |

**Returns:**

- Single image + `return_images=False` → `List[Dict]`
- Multiple images or `return_images=True` → `Results`

**Detection dict:**

```python
{
    "class_name": str,   # human-readable label
    "class_id":   int,   # 1-indexed class index
    "score":      float, # confidence in [0, 1]
    "bbox":       list,  # [x1, y1, x2, y2] absolute pixels
}
```

---

### `.evaluate()`

```python
metrics = model.evaluate(
    dataset,
    batch_size=4,
    conf=0.25,
)
```

Evaluate the model on a validation split.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `dataset` | str \| Path | **required** | Dataset root or `val/` subdirectory |
| `batch_size` | int | `4` | Images per evaluation batch |
| `conf` | float | `0.25` | Confidence threshold for counting detections |

**Returns:** dict with keys:

| Key | Description |
|-----|-------------|
| `total_images` | Total images evaluated |
| `total_predictions` | Total detections above threshold |
| `total_ground_truth` | Total ground-truth boxes |
| `avg_predictions_per_image` | Mean detections per image |
| `avg_ground_truth_per_image` | Mean GT boxes per image |
| `confidence_threshold` | The threshold used |

---

### `.save()`

```python
model.save(path)
```

Save the model checkpoint (weights + metadata) to `path`.

| Parameter | Type | Description |
|-----------|------|-------------|
| `path` | str \| Path | Output file path (`.pth` extension recommended) |

Weights are stored in **float16** to halve the file size. When the checkpoint
is loaded again via `.load()`, weights are transparently upcast back to float32.
The checkpoint also stores: class names, variant name, input size.

---

### `.load()`

```python
model = model.load(weights)
```

Load a checkpoint produced by `.save()` or by training. Returns `self` for
method chaining.

| Parameter | Type | Description |
|-----------|------|-------------|
| `weights` | str \| Path | Path to a `.pth` checkpoint |

Float16-stored checkpoints are automatically upcast to float32 on load.

---

### Properties

| Property | Type | Description |
|----------|------|-------------|
| `model.variant` | str | Active variant name (`"core"`, etc.) |
| `model.num_classes` | int \| None | Number of detection classes |
| `model.names` | List[str] | Class name list |
| `model.device` | torch.device | Active device |

---

### Static methods

```python
BILT.variants()   # print variant summary table (no instance needed)
```

---

## `bilt.Results`

Container for batch inference results with optional annotated images.
Returned by `model.predict(..., return_images=True)`.

```python
results = model.predict("images/", return_images=True)
```

### Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `results.detections` | List[List[Dict]] | All detection lists |
| `results.images` | List[PIL.Image] | Annotated images |
| `results.class_names` | List[str] | Class name list |

### Methods

```python
len(results)             # number of images
results[i]               # detection list for image i
results.save(save_dir)   # save annotated images to save_dir/
results.show()           # display with matplotlib
```

---

## `bilt.list_variants()`

```python
from bilt import list_variants
list_variants()
```

Prints a formatted table of all available model variants.

---

## `bilt.get_variant_config()`

```python
from bilt import get_variant_config

cfg = get_variant_config("pro")
```

Returns a copy of the configuration dict for the given variant name or alias.

| Key | Type | Description |
|-----|------|-------------|
| `backbone` | str | Backbone model name |
| `input_size` | int | Default input resolution |
| `fpn_channels` | int | FPN output channel width |
| `head_num_convs` | int | Convolutional layers in detection head |
| `anchor_sizes` | list | Anchor base sizes per FPN level |
| `anchor_scales` | tuple | Octave scale multipliers (3 scales → 9 anchors/location) |
| `anchor_aspect_ratios` | tuple | Aspect ratios per anchor |
| `description` | str | Human-readable variant description |

---

## `bilt.VARIANT_CONFIGS`

```python
from bilt import VARIANT_CONFIGS

print(VARIANT_CONFIGS.keys())
# dict_keys(['spark', 'flash', 'core', 'pro', 'max'])
```

Read-only dict of all variant configurations.

---

## `bilt.set_logging_level()`

```python
from bilt import set_logging_level

set_logging_level("WARNING")   # suppress INFO logs
set_logging_level("DEBUG")     # verbose output
set_logging_level("ERROR")     # errors only
```

Valid levels: `DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL`.

---

## Low-level classes

These are used internally but are available for advanced use cases.

### `bilt.core.BILTDetector`

```python
from bilt.core import BILTDetector

detector = BILTDetector(
    variant="core",
    num_classes=3,
)
```

Raw `nn.Module` implementing the full detection pipeline. In training mode,
`forward(images, targets)` returns a loss dict. In eval mode, `forward(images)`
returns a list of detection dicts.

### `bilt.core.DetectionModel`

Convenience wrapper around `BILTDetector` with `save()`, `load()`, `to()`,
`train()`, and `eval()` helpers. Used internally by `Trainer`.

Checkpoints are saved in float16 and loaded back as float32 automatically.

### `bilt.trainer.Trainer`

```python
from bilt.trainer import Trainer

trainer = Trainer(
    dataset_path="data/",
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
metrics = trainer.train(save_path="weights/best.pth", callback=fn)
```

See [Training Guide](training.md) for full usage.

### `bilt.inferencer.Inferencer`

```python
from bilt.inferencer import Inferencer

inf = Inferencer(
    model=detector,          # BILTDetector in eval mode
    class_names=["cat"],
    confidence_threshold=0.15,
    nms_threshold=0.45,
    input_size=512,
    device="cpu",
    max_detections=300,
)
dets = inf.detect(pil_image)
dets = inf.detect_batch([img1, img2])
dets = inf.detect_from_path("photo.jpg")
```

### `bilt.dataset.ObjectDetectionDataset`

```python
from bilt.dataset import ObjectDetectionDataset, get_transforms

ds = ObjectDetectionDataset(
    images_dir="data/train/images",
    labels_dir="data/train/labels",
    transforms=get_transforms(512, training=True),
    input_size=512,
    training=True,
    augment=True,
    flip_prob=0.5,
    color_jitter=(0.4, 0.4, 0.4, 0.1),
)
img_tensor, target = ds[0]
# target = {"boxes": Tensor(N,4), "labels": Tensor(N,)}
```

### `bilt.dataset.read_dataset_info()`

```python
from bilt.dataset import read_dataset_info
from pathlib import Path

num_classes, class_names = read_dataset_info(
    labels_dir=Path("data/train/labels"),
    yaml_path=Path("data/data.yaml"),   # optional
)
```

Lightweight class-info reader that scans only label `.txt` files — no images
are loaded. Used internally by `BILT.train()`.

### `bilt.anchors` module

```python
from bilt.anchors import (
    AnchorGenerator,  # generates anchor boxes for FPN levels
    AnchorMatcher,    # assigns GT boxes to anchors
    encode_boxes,     # GT → delta encoding
    decode_boxes,     # delta → absolute boxes
    box_iou,          # pairwise IoU matrix
)
```

`AnchorGenerator` uses 3 octave scales × 3 aspect ratios = **9 anchors per
location** by default (scales: 1.0, 1.26, 1.587).

`AnchorMatcher` thresholds: positive IoU ≥ 0.35, negative IoU < 0.25.

### `bilt.loss.BILTLoss`

```python
from bilt.loss import BILTLoss, sigmoid_focal_loss, smooth_l1_loss

criterion = BILTLoss(
    num_classes=3,
    alpha=0.25,       # focal loss alpha
    gamma=2.0,        # focal loss gamma
    box_weight=1.0,   # relative weight of regression loss
)
losses = criterion(cls_preds, box_preds, cls_targets, box_targets, pos_mask)
# losses = {"total": scalar, "cls": scalar, "box": scalar}
```
