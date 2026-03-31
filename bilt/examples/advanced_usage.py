# BILT (Because I Like Twice) - A PyTorch-based object detection library
# Copyright (C) 2026 Rikiza89
# Licensed under the Apache License, Version 2.0

"""
Advanced BILT usage examples.
"""

from pathlib import Path
from bilt import BILT

# ── 1. Training with a progress callback ────────────────────────────────────
print("=" * 60)
print("Example 1: Training with custom callback")
print("=" * 60)


def training_callback(info):
    pct = info["epoch"] / info["total_epochs"] * 100
    print(
        f"[{pct:5.1f}%] Epoch {info['epoch']}/{info['total_epochs']} | "
        f"train={info['train_loss']:.4f}  val={info['val_loss']:.4f}  "
        f"lr={info['lr']:.2e}"
    )


model = BILT("pro")        # ResNet-50, 640 px
metrics = model.train(
    dataset="datasets/my_dataset",
    epochs=100,
    batch_size=8,
    learning_rate=5e-4,
    save_dir="runs/train",
    name="pro_experiment",
)

# ── 2. Inference with different thresholds ───────────────────────────────────
print("\n" + "=" * 60)
print("Example 2: Custom inference parameters")
print("=" * 60)

model = BILT("weights/best.pth")

# More detections (lower threshold)
r_low  = model.predict("image.jpg", conf=0.1, iou=0.3)
# Fewer, higher-quality detections
r_high = model.predict("image.jpg", conf=0.8, iou=0.5)

print(f"conf=0.1: {len(r_low)}  detections")
print(f"conf=0.8: {len(r_high)} detections")

# ── 3. Save and reload ───────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("Example 3: Save and reload a model")
print("=" * 60)

model = BILT("flash")      # MobileNetV3-Small, 416 px
model.train(dataset="datasets/my_dataset", epochs=10)
model.save("models/my_flash_model.pth")

restored = BILT("models/my_flash_model.pth")
print(restored)            # BILT(variant=flash, classes=N, device=cpu)

# ── 4. Process an entire directory ──────────────────────────────────────────
print("\n" + "=" * 60)
print("Example 4: Process a directory of images")
print("=" * 60)

model = BILT("weights/best.pth")
dataset_dir = Path("datasets/test_images")

summary = []
for img_path in sorted(dataset_dir.glob("*.jpg")):
    dets = model.predict(str(img_path), conf=0.3)
    summary.append({"image": img_path.name, "count": len(dets)})
    print(f"  {img_path.name}: {len(dets)} objects")

total = sum(r["count"] for r in summary)
print(f"\nTotal: {len(summary)} images, {total} detections")

# ── 5. Device selection ──────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("Example 5: Device selection")
print("=" * 60)

model_cpu  = BILT("weights/best.pth", device="cpu")
model_auto = BILT("weights/best.pth")          # auto-detect
print(f"CPU model  : {model_cpu.device}")
print(f"Auto model : {model_auto.device}")

try:
    model_gpu = BILT("weights/best.pth", device="cuda")
    print(f"GPU model  : {model_gpu.device}")
except Exception:
    print("CUDA not available.")

# ── 6. Inspect model properties ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("Example 6: Model properties")
print("=" * 60)

model = BILT("weights/best.pth")
print(model)                                   # repr
print(f"  variant    : {model.variant}")
print(f"  num_classes: {model.num_classes}")
print(f"  class names: {model.names}")
print(f"  device     : {model.device}")
