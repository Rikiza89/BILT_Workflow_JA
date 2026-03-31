# Inference Guide

---

## Loading a model

```python
from bilt import BILT

model = BILT("runs/train/exp/weights/best.pth")
```

The checkpoint stores the variant name, class names, and input resolution.
Everything is restored automatically — no need to pass extra arguments.
Float16-stored weights are transparently upcast to float32 on load.

### Checking what was loaded

```python
print(model)
# On a GPU machine:   BILT(variant=core, classes=3, device=cuda)
# On a CPU machine:   BILT(variant=core, classes=3, device=cpu)

print(model.variant)         # core
print(model.num_classes)     # 3
print(model.names)           # ['cat', 'dog', 'person']
print(model.device)          # cuda  or  cpu  (auto-detected)
```

---

## Running predictions

### Single image (file path)

```python
detections = model.predict("photo.jpg")
```

### Single image (PIL Image)

```python
from PIL import Image

img = Image.open("photo.jpg")
detections = model.predict(img)
```

### Single image (numpy array)

```python
import numpy as np
from PIL import Image

arr = np.array(Image.open("photo.jpg"))   # H×W×3, uint8
detections = model.predict(arr)
```

### Directory of images

```python
# All .jpg, .jpeg, .png, .bmp files in the directory
all_detections = model.predict("images/")
# Returns a list of detection lists (one per image)
```

### List of mixed inputs

```python
from PIL import Image
import numpy as np

inputs = [
    "img1.jpg",
    Image.open("img2.png"),
    np.array(Image.open("img3.jpg")),
]
all_detections = model.predict(inputs)
```

---

## Detection format

Each call to `predict()` returns a list of detection dicts for each image.

```python
# Single image
detections = model.predict("photo.jpg")

for det in detections:
    print(det)
```

```python
{
    "class_name": "cat",         # human-readable label
    "class_id":   1,             # 1-indexed integer
    "score":      0.8732,        # confidence in [0, 1]
    "bbox":       [142, 67, 310, 289]  # [x1, y1, x2, y2] absolute pixels
}
```

The `bbox` coordinates are in the **original image's pixel space** — BILT
rescales them back from the model's input resolution for you.

---

## Confidence threshold

Controls the minimum score a detection must have to be returned.

```python
# Default — recommended starting point
detections = model.predict("photo.jpg", conf=0.15)

# More detections (lower bar — may include false positives)
detections = model.predict("photo.jpg", conf=0.05)

# High precision (fewer detections, less noise)
detections = model.predict("photo.jpg", conf=0.5)
```

The default `conf=0.15` is tuned for use with pretrained + fine-tuned models.
Raise it to 0.4–0.7 for precision-critical applications; lower it to 0.05–0.10
for recall-critical ones (e.g., safety systems where missing objects is costly).

---

## NMS IoU threshold

Controls how aggressively overlapping boxes are suppressed.

```python
# Aggressive NMS — keeps only the best box per object cluster
detections = model.predict("photo.jpg", iou=0.3)

# Default
detections = model.predict("photo.jpg", iou=0.45)

# Relaxed NMS — allows more overlap (useful for dense scenes)
detections = model.predict("photo.jpg", iou=0.6)
```

---

## Limiting detections per image

Use `max_det` to cap the maximum number of detections returned per image.
Detections are ranked by confidence and the top-N are kept.

```python
# Default: keep up to 300 detections
detections = model.predict("photo.jpg", max_det=300)

# Tight cap — only the top 10 most confident detections
detections = model.predict("photo.jpg", max_det=10)
```

---

## Override inference resolution

```python
# Faster, lower resolution
detections = model.predict("photo.jpg", img_size=320)

# Higher resolution — better for small objects
detections = model.predict("photo.jpg", img_size=640)
```

The default is the variant's training resolution (e.g., 512 for `core`).
You can override without retraining; the backbone will rescale the input.

---

## Returning annotated images

Set `return_images=True` to get a `Results` object that bundles detections
with annotated images.

```python
results = model.predict("images/", conf=0.15, return_images=True)

# Save annotated images to disk
results.save("runs/detect/exp")

# Display in a Jupyter notebook / interactive session
results.show()

# Access raw detections
for i, dets in enumerate(results):
    print(f"Image {i}: {len(dets)} objects")
```

### Results object

```python
len(results)        # number of images
results[0]          # detection list for the first image
results.detections  # list of all detection lists
results.images      # list of PIL Images (annotated)
```

---

## Batch processing

For large numbers of images, process them as a list rather than one by one
— this avoids repeated model reloads:

```python
from pathlib import Path

img_paths = sorted(Path("images/").glob("*.jpg"))

# Process all at once
all_dets = model.predict(list(img_paths), conf=0.15)

for path, dets in zip(img_paths, all_dets):
    print(f"{path.name}: {len(dets)} objects")
```

Or use the `Inferencer` directly for the lowest-overhead batch loop:

```python
from bilt.inferencer import Inferencer
from PIL import Image

inf = Inferencer(
    model=model.model,
    class_names=model.names,
    confidence_threshold=0.15,
    nms_threshold=0.45,
    input_size=512,
    device=model.device,
    max_detections=300,
)

images = [Image.open(p) for p in img_paths]
results = inf.detect_batch(images)
```

---

## GPU inference

`Inferencer` checks `torch.cuda.is_available()` automatically, so loading a
model on a GPU machine requires no extra arguments:

```python
# GPU is used automatically when CUDA is available
model = BILT("weights/best.pth")
detections = model.predict("photo.jpg")
print(model.device)   # cuda
```

To pin to a specific device:

```python
model = BILT("weights/best.pth", device="cuda:1")   # second GPU
model = BILT("weights/best.pth", device="cpu")       # force CPU
model = BILT("weights/best.pth", device="mps")       # Apple Silicon
```

---

## Interpreting results

### Filter by class

```python
detections = model.predict("photo.jpg", conf=0.15)

cats = [d for d in detections if d["class_name"] == "cat"]
dogs = [d for d in detections if d["class_name"] == "dog"]
```

### Filter by score

```python
high_conf = [d for d in detections if d["score"] > 0.8]
```

### Get bounding boxes as tensors

```python
import torch

boxes = torch.tensor([d["bbox"] for d in detections])    # (N, 4)
scores = torch.tensor([d["score"] for d in detections])  # (N,)
```

### Draw boxes manually

```python
from bilt.utils import draw_detections
from PIL import Image

img = Image.open("photo.jpg")
annotated = draw_detections(img, detections)
annotated.save("annotated.jpg")
annotated.show()
```

---

## Common issues

### No detections returned

1. **Confidence too high** — try `conf=0.05` to see if the model is detecting
   anything at all. The default is already `0.15`, which is conservative.
2. **Wrong input resolution** — the model was trained at a specific size. Try
   specifying `img_size` to match training.
3. **Model not trained on these classes** — check `model.names`.
4. **Model not converged** — if trained on very few epochs, try longer training.

### Too many false positives

1. **Confidence too low** — increase to `conf=0.4` or higher.
2. **NMS too relaxed** — decrease `iou=0.3`.
3. **Too many detections** — set `max_det=10` to cap at the top-N most confident.

### Detections look correct but boxes are off-position

This usually means the image was not RGB before being passed in. BILT
converts to RGB automatically for PIL Images and numpy arrays, but check that
your custom pipeline is not feeding BGR (e.g., from OpenCV).

```python
import cv2
from PIL import Image

bgr = cv2.imread("photo.jpg")
rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
img = Image.fromarray(rgb)
detections = model.predict(img)
```
