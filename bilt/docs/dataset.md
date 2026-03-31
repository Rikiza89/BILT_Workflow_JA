# Dataset Preparation

BILT reads datasets from disk in a simple, widely-supported format that is
compatible with LabelImg, Roboflow, CVAT, and most other annotation tools.

---

## Directory layout

```
my_dataset/
├── train/
│   ├── images/          ← training images (.jpg, .jpeg, .png, .bmp)
│   └── labels/          ← training annotations (.txt, one per image)
├── val/
│   ├── images/          ← validation images
│   └── labels/          ← validation annotations
└── data.yaml            ← class names (optional but recommended)
```

The `train/` and `val/` splits are both required. The `data.yaml` file is
optional but strongly recommended — without it, BILT uses auto-generated
class names like `class_0`, `class_1`, etc.

---

## Label file format

Each image has a corresponding `.txt` file with the **same stem** in the
`labels/` directory. Empty label files (no objects in the image) are
allowed.

Each line describes one bounding box:

```
<class_id> <x_center> <y_center> <width> <height>
```

- All five values are separated by spaces.
- `class_id` is a **zero-based integer** (0, 1, 2, …).
- `x_center`, `y_center`, `width`, `height` are **normalised to [0, 1]**
  relative to the image dimensions.

### Example

Image size: 640 × 480. A box covering pixels (100, 80) → (300, 200):

```
width_norm  = (300 - 100) / 640 = 0.3125
height_norm = (200 - 80)  / 480 = 0.25
x_center    = (100 + 300) / 2 / 640 = 0.3125
y_center    = (80  + 200) / 2 / 480 = 0.2917
```

Label line (class 0):

```
0 0.3125 0.2917 0.3125 0.25
```

Multiple objects in one image (one line each):

```
0 0.3125 0.2917 0.3125 0.2500
1 0.7500 0.5000 0.2000 0.3000
0 0.1250 0.8000 0.1000 0.1500
```

---

## data.yaml

```yaml
# Number of classes
nc: 3

# Class names in order (index matches class_id in label files)
names:
  - cat
  - dog
  - person
```

Alternative list syntax:

```yaml
nc: 3
names: [cat, dog, person]
```

Alternative dict syntax (for non-consecutive class IDs):

```yaml
names:
  0: cat
  1: dog
  2: person
```

Place `data.yaml` in the dataset root (same level as `train/` and `val/`).
BILT also looks for `data.yml` and `dataset.yaml` if `data.yaml` is missing.

---

## Annotation tools

### LabelImg

LabelImg can export directly in this format. Select the plain-text
bounding-box format (one `.txt` file per image, values in `[0, 1]`) as the
output format in the application. The output `.txt` files are immediately
compatible with BILT.

### Roboflow

1. Upload your images and annotate them.
2. Export → Format: **"Darknet"** (normalised text labels, one `.txt` per image).
3. The downloaded zip contains `train/`, `valid/` (rename to `val/`), and
   a `data.yaml`.

### CVAT

1. Export annotations → **Darknet 1.1** (normalised text) format.
2. The exported archive includes images and labels in the correct structure.

### Make Sense (makesense.ai)

1. Annotate online.
2. Export → **"Plain text" / normalised bounding-box format**.

---

## Splitting your data

If you have a single pool of labelled images, you need to create the
`train/` and `val/` splits manually. A typical 80/20 split:

```python
import random
import shutil
from pathlib import Path

src_images = Path("all_images")
src_labels = Path("all_labels")

out = Path("my_dataset")
for split, imgs in [("train", train_list), ("val", val_list)]:
    (out / split / "images").mkdir(parents=True, exist_ok=True)
    (out / split / "labels").mkdir(parents=True, exist_ok=True)
    for img in imgs:
        shutil.copy(src_images / img.name, out / split / "images" / img.name)
        lbl = src_labels / img.with_suffix(".txt").name
        if lbl.exists():
            shutil.copy(lbl, out / split / "labels" / lbl.name)

# Randomised 80/20 split
all_imgs = sorted(src_images.glob("*.jpg"))
random.shuffle(all_imgs)
n_train = int(0.8 * len(all_imgs))
train_list = all_imgs[:n_train]
val_list   = all_imgs[n_train:]
```

---

## Validating your dataset

Before training, you can verify the structure is correct:

```python
from bilt.utils import validate_dataset_structure
from pathlib import Path

ok, msg = validate_dataset_structure(Path("my_dataset"))
print(ok, msg)
```

Or simply attempt a dataset load:

```python
from bilt.dataset import ObjectDetectionDataset, get_transforms

ds = ObjectDetectionDataset(
    images_dir="my_dataset/train/images",
    labels_dir="my_dataset/train/labels",
    transforms=get_transforms(512),
)
print(f"{len(ds)} images, {ds.num_classes} classes")
print("Class IDs found:", ds.class_ids)
```

---

## Non-consecutive class IDs

If your label files use class IDs that are not consecutive (e.g., 0, 3, 7),
BILT automatically remaps them to 0, 1, 2, … internally. This remapping is
stored in the checkpoint so inference uses the same mapping.

---

## Training augmentation

BILT applies the following augmentation to training images automatically:

| Transform | Default | Description |
|-----------|---------|-------------|
| Random horizontal flip | prob=0.5 | Mirrors image + corrects box coords |
| Color jitter | (0.4, 0.4, 0.4, 0.1) | Brightness, contrast, saturation, hue |

Validation images are **never augmented**.

To customise or disable augmentation:

```python
# Disable all augmentation
model.train(dataset="data/", augment=False)

# Custom flip probability
model.train(dataset="data/", flip_prob=0.3)

# Disable color jitter only
model.train(dataset="data/", color_jitter=None)

# Stronger jitter for small datasets
model.train(dataset="data/", color_jitter=(0.6, 0.6, 0.6, 0.15))
```

---

## Large datasets

For datasets with many images, increase the `workers` argument to speed up
data loading (not recommended on Windows or Raspberry Pi — use 0 instead):

```python
model.train(dataset="data/", workers=4)
```

---

## Minimum dataset size

BILT is specifically designed to work with **very small datasets**, including
as few as 2 training images, thanks to ImageNet pretrained backbones and
data augmentation. The model can still learn meaningful detectors even from
minimal data.

General guidance:

| Images per class | Expectation |
|-----------------|-------------|
| 2–5 | Works; model may be noisy — use `conf=0.05` to see detections |
| 10–50 | Good results with 50+ epochs |
| 50–200 | Solid detector; most use cases covered |
| 200–500 | High-quality results; try `pro` or `max` |
| 500+ | Excellent results; increase batch size and epochs |

---

## Class imbalance

If some classes have far more examples than others, the focal loss used by
BILT already helps mitigate this. However, for extreme imbalance (e.g., 10:1
ratio), consider oversampling the minority class by duplicating images and
labels into the `train/` directory.

You can also tune the focal loss `alpha` parameter:

```python
# Down-weight easy majority class examples more aggressively
model.train(dataset="data/", focal_alpha=0.5)
```
