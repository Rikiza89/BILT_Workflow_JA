# Installation

## Requirements

| Package | Minimum version | Purpose |
|---------|----------------|---------|
| Python | 3.8 | Language runtime |
| torch | 1.10 | Neural network framework |
| torchvision | 0.11 | Backbone architectures, NMS |
| Pillow | 8.0 | Image I/O |
| numpy | 1.19 | Array operations |
| pyyaml | 5.4 | Dataset config files |

Optional:

| Package | Purpose |
|---------|---------|
| matplotlib | `results.show()` visualisation |
| pytest | Running tests |

---

## Install from source (recommended for now)

```bash
git clone https://github.com/Rikiza89/bilt.git
cd bilt
pip install -e .
```

The `-e` flag installs in *editable* mode, so any changes you make to the
source are immediately reflected without re-installing.

---

## Install dependencies only

If you want to manage the package manually:

```bash
pip install torch torchvision pillow numpy pyyaml
```

For visualisation support:

```bash
pip install matplotlib
```

---

## First-run note

On the first `model.train()` call, torchvision will download ImageNet pretrained
weights for the backbone (~10–130 MB depending on variant). These are cached by
torchvision and only downloaded once.

---

## Platform notes

### CPU (any platform)

BILT works out of the box on CPU. No extra steps needed. The `spark` and
`flash` variants are fast enough for real-time use on a modern laptop.

### NVIDIA GPU (CUDA)

Install PyTorch with CUDA support from [pytorch.org](https://pytorch.org/get-started/locally/).
Choose the version that matches your CUDA toolkit (11.8 or 12.x recommended).

```bash
# CUDA 11.8
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# CUDA 12.1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

BILT checks `torch.cuda.is_available()` automatically. Once a CUDA-enabled PyTorch
is installed, BILT uses the GPU for **both training and inference without any extra
configuration**:

```python
model = BILT("core")          # GPU used automatically
model.train(dataset="data/")  # same — no device= needed
```

Verify that CUDA is visible before running:

```python
import torch
print(torch.cuda.is_available())    # True
print(torch.cuda.get_device_name(0))
```

To override and force a specific device, pass `device=` explicitly:

```python
model = BILT("core", device="cuda:1")   # second GPU
model = BILT("core", device="cpu")      # force CPU
```

### Apple Silicon (MPS)

PyTorch supports Apple's Metal Performance Shaders backend:

```bash
pip install torch torchvision   # standard wheels include MPS
```

```python
model = BILT("core", device="mps")
```

### Raspberry Pi / ARM

Use the `spark` or `flash` variant with `device="cpu"`. Limit workers to 0:

```python
metrics = model.train(dataset="data/", epochs=50, batch_size=2, workers=0)
```

Install a lightweight PyTorch build if needed:

```bash
# Raspberry Pi OS (64-bit)
pip install torch torchvision --extra-index-url https://download.pytorch.org/whl/cpu
```

---

## Verify installation

```python
import bilt
print(bilt.__version__)   # 0.2.0

from bilt import BILT, list_variants
list_variants()
```

Expected output:

```
Variant   Backbone                Input   FPN ch   Description
---------------------------------------------------------------------------
spark     mobilenet_v2            320     64       Nano - fastest inference…
flash     mobilenet_v3_small      416     96       Small - fast with good…
core      mobilenet_v3_large      512     128      Medium - balanced speed…
pro       resnet50                640     256      Large - high accuracy
max       resnet101               640     256      XLarge - maximum accuracy
```

---

## Common installation errors

### `ModuleNotFoundError: No module named 'torch'`

PyTorch is not installed. Run:

```bash
pip install torch torchvision
```

### `ImportError: cannot import name 'nms' from 'torchvision.ops'`

Torchvision version is too old. Upgrade:

```bash
pip install -U torchvision
```

### Out-of-memory during training

Reduce batch size or switch to a smaller variant:

```python
model = BILT("spark")
model.train(dataset="data/", batch_size=2)
```

### `RuntimeError: CUDA out of memory`

Lower the batch size or switch to a smaller variant. `spark` with
`batch_size=4` uses about 1.5 GB of VRAM.
