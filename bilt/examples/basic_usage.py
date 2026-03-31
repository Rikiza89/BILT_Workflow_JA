# BILT (Because I Like Twice) - A PyTorch-based object detection library
# Copyright (C) 2026 Rikiza89
# Licensed under the Apache License, Version 2.0

"""
Basic BILT usage examples.

All five model sizes use different backbone architectures:
    spark  — MobileNetV2,       320 px  (nano / fastest)
    flash  — MobileNetV3-Small, 416 px  (small)
    core   — MobileNetV3-Large, 512 px  (medium / default)
    pro    — ResNet-50,         640 px  (large)
    max    — ResNet-101,        640 px  (xlarge / most accurate)
"""

from bilt import BILT, list_variants

# ── 0. List available variants ──────────────────────────────────────────────
list_variants()

# ── 1. Load a saved model and predict ───────────────────────────────────────
print("=" * 60)
print("Example 1: Load model and predict")
print("=" * 60)

model = BILT("weights/best.pth")
detections = model.predict("images/test.jpg", conf=0.25)

print(f"\nDetected {len(detections)} objects:")
for i, det in enumerate(detections, 1):
    print(f"  {i}. {det['class_name']}: {det['score']:.2f}  bbox={det['bbox']}")

# ── 2. Batch prediction with annotated images ────────────────────────────────
print("\n" + "=" * 60)
print("Example 2: Batch prediction")
print("=" * 60)

results = model.predict("images/", conf=0.3, return_images=True)
print(f"\nProcessed {len(results)} images")
results.save("runs/detect/exp")    # saves annotated images as JPEGs

# ── 3. Train a new model ─────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("Example 3: Train a 'core' model (MobileNetV3-Large, 512 px)")
print("=" * 60)

model = BILT("core")               # pick any variant here
metrics = model.train(
    dataset="datasets/my_dataset",
    epochs=50,
    batch_size=4,
)

print(f"\nTraining complete!")
print(f"  variant        : {metrics['variant']}")
print(f"  final train loss: {metrics['final_train_loss']:.4f}")
print(f"  final val loss  : {metrics['final_val_loss']:.4f}")
print(f"  model saved to  : {metrics['model_path']}")

# ── 4. Evaluate model ────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("Example 4: Evaluate")
print("=" * 60)

m = model.evaluate("datasets/my_dataset", conf=0.25)
print(f"\n  Total images     : {m['total_images']}")
print(f"  Total predictions: {m['total_predictions']}")
print(f"  Avg pred / image : {m['avg_predictions_per_image']:.2f}")

# ── 5. Different input types ─────────────────────────────────────────────────
print("\n" + "=" * 60)
print("Example 5: Multiple input types")
print("=" * 60)

from PIL import Image
import numpy as np

r1 = model.predict("image.jpg")                   # file path
r2 = model.predict(Image.open("image.jpg"))        # PIL Image
r3 = model.predict(np.array(Image.open("image.jpg")))  # numpy array
r4 = model.predict(["img1.jpg", "img2.jpg"])       # list of paths

print("All input types processed successfully.")
