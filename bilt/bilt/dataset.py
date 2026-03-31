# BILT (Because I Like Twice) - A PyTorch-based object detection library
# Copyright (C) 2026 Rikiza89
# Licensed under the Apache License, Version 2.0

"""
BILT dataset loader.

Expects the standard label format used by most annotation tools:

    <class_id>  <x_center>  <y_center>  <width>  <height>

All five values are normalised to [0, 1] relative to the image dimensions.
One label file per image, with the same stem and a ``.txt`` extension,
located in the ``labels/`` directory that mirrors the ``images/`` directory.

Dataset directory layout
------------------------
    <root>/
        train/
            images/   *.jpg / *.png / …
            labels/   *.txt
        val/
            images/
            labels/
        data.yaml     (optional – used for human-readable class names)

data.yaml format
----------------
    names: [cat, dog, person]
    nc: 3
"""

import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from .backbone import NORM_MEAN, NORM_STD
from .utils import get_logger, load_yaml_classes, parse_bilt_label

logger = get_logger(__name__)


class ObjectDetectionDataset(Dataset):
    """
    PyTorch Dataset for object detection.

    Loads images and their corresponding annotation files, remaps class IDs
    to a consecutive zero-indexed range, resizes images to *input_size* and
    applies standard normalisation.

    Parameters
    ----------
    images_dir   : Path   Directory containing image files.
    labels_dir   : Path   Directory containing label ``.txt`` files.
    transforms   : callable, optional  Applied to each PIL image.
    input_size   : int    Target square resolution (only used if no transforms).
    cache_images : bool   Pre-load all images and labels into RAM after first
                          access.  Eliminates disk I/O from epoch 2 onward.
                          Highly recommended for small datasets (< ~1 GB).
    mosaic       : bool   Apply mosaic augmentation during training.
                          Combines 4 random images into one canvas, greatly
                          increasing scene diversity on small datasets.
    mosaic_prob  : float  Probability of applying mosaic per sample (default 0.5).
    """

    def __init__(
        self,
        images_dir: Path,
        labels_dir: Path,
        transforms=None,
        input_size: int = 512,
        training: bool = False,
        augment: Optional[bool] = None,
        flip_prob: float = 0.5,
        color_jitter: Optional[Tuple[float, float, float, float]] = (0.4, 0.4, 0.4, 0.1),
        cache_images: bool = False,
        mosaic: bool = False,
        mosaic_prob: float = 0.5,
    ):
        self.images_dir = Path(images_dir)
        self.labels_dir = Path(labels_dir)
        self.transforms = transforms
        self.input_size = input_size
        self.training = training
        # augment defaults to True for training split, False for val
        self.augment = training if augment is None else augment
        self.flip_prob = flip_prob
        self.color_jitter = color_jitter
        self.cache_images = cache_images
        # Mosaic only makes sense when augmenting and dataset has >= 4 images
        self.mosaic = mosaic and self.augment
        self.mosaic_prob = mosaic_prob

        # Discover image files
        self.image_files: List[Path] = []
        for ext in [".jpg", ".jpeg", ".png", ".bmp"]:
            self.image_files.extend(self.images_dir.glob(f"*{ext}"))
            self.image_files.extend(self.images_dir.glob(f"*{ext.upper()}"))
        self.image_files = sorted(self.image_files)

        if not self.image_files:
            raise ValueError(f"No images found in {self.images_dir}")

        logger.info(f"Dataset: {len(self.image_files)} images in {self.images_dir}")

        # Disable mosaic when dataset is too small to sample 4 distinct images
        if self.mosaic and len(self.image_files) < 4:
            logger.info("Mosaic disabled: need at least 4 images (dataset has fewer).")
            self.mosaic = False

        # Collect unique class IDs present in the label files
        self.class_ids: List[int] = []
        _ids: set = set()
        for img_path in self.image_files:
            lbl = self.labels_dir / f"{img_path.stem}.txt"
            if lbl.exists():
                try:
                    with open(lbl) as f:
                        for line in f:
                            parts = line.strip().split()
                            if len(parts) >= 5:
                                _ids.add(int(parts[0]))
                except Exception as exc:
                    logger.warning(f"Could not read {lbl}: {exc}")

        self.class_ids = sorted(_ids)
        self.num_classes = len(self.class_ids)

        # Map original class IDs to consecutive indices (0-based internally,
        # but the model receives 1-based labels to distinguish from background)
        self.class_id_to_idx: Dict[int, int] = {
            cid: idx for idx, cid in enumerate(self.class_ids)
        }
        self.idx_to_class_id: Dict[int, int] = {
            idx: cid for cid, idx in self.class_id_to_idx.items()
        }

        logger.info(
            f"Found {self.num_classes} classes with IDs: {self.class_ids}"
        )

        # ── RAM cache ──────────────────────────────────────────────────────────
        # Pre-loaded as (PIL.Image, raw_anns) tuples.  Populated lazily on
        # first access so the constructor stays fast.
        self._cache: Dict[int, Tuple] = {}
        if cache_images:
            logger.info("Caching all images and labels into RAM …")
            for i in range(len(self.image_files)):
                self._cache_item(i)
            logger.info(f"Cache complete: {len(self._cache)} items loaded.")

    # ---------------------------------------------------------------------- #
    # Internal helpers
    # ---------------------------------------------------------------------- #

    def _cache_item(self, idx: int):
        """Load image + annotations into self._cache[idx] if not already there."""
        if idx in self._cache:
            return
        img_path = self.image_files[idx]
        lbl_path = self.labels_dir / f"{img_path.stem}.txt"
        try:
            img = Image.open(img_path).convert("RGB")
        except Exception as exc:
            logger.error(f"Cannot load {img_path}: {exc}")
            img = Image.new("RGB", (self.input_size, self.input_size))
        anns = parse_bilt_label(lbl_path, *img.size)
        self._cache[idx] = (img.copy(), anns)

    def _load_raw(self, idx: int) -> Tuple[Image.Image, list]:
        """Return (PIL image, raw annotations) — from cache or disk."""
        if self.cache_images:
            if idx not in self._cache:
                self._cache_item(idx)
            img, anns = self._cache[idx]
            return img.copy(), anns          # copy so augmentation is non-destructive
        # Disk path
        img_path = self.image_files[idx]
        lbl_path = self.labels_dir / f"{img_path.stem}.txt"
        try:
            img = Image.open(img_path).convert("RGB")
        except Exception as exc:
            logger.error(f"Cannot load {img_path}: {exc}")
            img = Image.new("RGB", (self.input_size, self.input_size))
        anns = parse_bilt_label(lbl_path, *img.size)
        return img, anns

    def _ann_to_tensors(self, anns: list, orig_w: int, orig_h: int):
        """Convert raw annotation dicts to (boxes, labels) tensors."""
        if anns:
            boxes = torch.tensor([a["bbox"] for a in anns], dtype=torch.float32)
            labels = torch.tensor(
                [self.class_id_to_idx[a["class_id"]] + 1 for a in anns],
                dtype=torch.int64,
            )
        else:
            boxes  = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,),   dtype=torch.int64)
        return boxes, labels

    def _color_jitter_transform(self) -> T.ColorJitter:
        b, c, s, h = self.color_jitter
        return T.ColorJitter(brightness=b, contrast=c, saturation=s, hue=h)

    # ---------------------------------------------------------------------- #
    # Mosaic augmentation
    # ---------------------------------------------------------------------- #

    def _load_mosaic(self, idx: int) -> Tuple[torch.Tensor, Dict]:
        """
        Build a mosaic from 4 images (the requested one + 3 random others).

        The output is a (C, S, S) normalised tensor where S = self.input_size,
        together with the merged target dict.

        Mosaic layout  (cx, cy = random centre inside [S*0.4, S*0.6]):
            ┌─────────┬─────────┐
            │  img0   │  img1   │
            │ (top-L) │ (top-R) │
            ├─────────┼─────────┤
            │  img2   │  img3   │
            │ (bot-L) │ (bot-R) │
            └─────────┴─────────┘
        """
        S = self.input_size
        # Random centre, biased toward the middle to avoid very thin slices
        cx = int(random.uniform(S * 0.35, S * 0.65))
        cy = int(random.uniform(S * 0.35, S * 0.65))

        # Pick 3 other random indices (may repeat but that's fine)
        indices = [idx] + random.choices(range(len(self.image_files)), k=3)

        # Canvas (RGB uint8 — we normalise at the end)
        canvas = Image.new("RGB", (S, S), (114, 114, 114))

        all_boxes: List[torch.Tensor] = []
        all_labels: List[torch.Tensor] = []

        # Placement specs: (x_start, y_start, x_end, y_end) on canvas,
        # and which corner of the source image to crop from.
        placements = [
            # top-left  → take bottom-right corner of source
            (0,  0,  cx, cy,  "br"),
            # top-right → take bottom-left corner of source
            (cx, 0,  S,  cy,  "bl"),
            # bot-left  → take top-right corner of source
            (0,  cy, cx, S,   "tr"),
            # bot-right → take top-left corner of source
            (cx, cy, S,  S,   "tl"),
        ]

        for i, (x1, y1, x2, y2, corner) in enumerate(placements):
            pw, ph = x2 - x1, y2 - y1          # patch size on canvas
            if pw <= 0 or ph <= 0:
                continue

            img_i, anns_i = self._load_raw(indices[i])
            iw, ih = img_i.size

            # Scale source to the full target size, then crop the relevant corner
            img_scaled = img_i.resize((S, S), Image.BILINEAR)

            if corner == "br":
                crop = (S - pw, S - ph, S, S)
            elif corner == "bl":
                crop = (0, S - ph, pw, S)
            elif corner == "tr":
                crop = (S - pw, 0, S, ph)
            else:   # tl
                crop = (0, 0, pw, ph)

            patch = img_scaled.crop(crop)
            canvas.paste(patch, (x1, y1))

            # Map bounding boxes into canvas space
            # Source boxes are in original pixel coords (iw × ih).
            # After resize to S×S: scale by (S/iw, S/ih).
            # After crop from corner: offset by (-crop_x, -crop_y).
            # After paste at (x1, y1): offset by (x1, y1).
            if anns_i:
                bx = torch.tensor([a["bbox"] for a in anns_i], dtype=torch.float32)
                # scale to S×S
                sx, sy = S / iw, S / ih
                bx[:, [0, 2]] *= sx
                bx[:, [1, 3]] *= sy
                # shift by crop offset
                crop_x, crop_y = crop[0], crop[1]
                bx[:, [0, 2]] -= crop_x
                bx[:, [1, 3]] -= crop_y
                # shift by paste position
                bx[:, [0, 2]] += x1
                bx[:, [1, 3]] += y1
                # clip to this patch on the canvas
                bx[:, [0, 2]] = bx[:, [0, 2]].clamp(x1, x2)
                bx[:, [1, 3]] = bx[:, [1, 3]].clamp(y1, y2)
                # filter degenerate boxes
                valid = (bx[:, 2] > bx[:, 0]) & (bx[:, 3] > bx[:, 1])
                bx = bx[valid]
                lbl_i = torch.tensor(
                    [self.class_id_to_idx[a["class_id"]] + 1 for a in anns_i],
                    dtype=torch.int64,
                )
                lbl_i = lbl_i[valid]
                if bx.numel() > 0:
                    all_boxes.append(bx)
                    all_labels.append(lbl_i)

        # Color jitter on the full canvas (single call, consistent look)
        if self.color_jitter is not None:
            canvas = self._color_jitter_transform()(canvas)

        # Random horizontal flip of the whole mosaic
        if random.random() < self.flip_prob:
            canvas = TF.hflip(canvas)
            for bx in all_boxes:
                bx[:, 0], bx[:, 2] = S - bx[:, 2], S - bx[:, 0]

        # Normalise canvas → tensor
        img_t = self.transforms(canvas) if self.transforms else TF.to_tensor(canvas)

        if all_boxes:
            boxes  = torch.cat(all_boxes, dim=0)
            labels = torch.cat(all_labels, dim=0)
            # Final validity check after flip
            valid = (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
            boxes  = boxes[valid]
            labels = labels[valid]
        else:
            boxes  = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,),   dtype=torch.int64)

        return img_t, {"boxes": boxes, "labels": labels}

    # ---------------------------------------------------------------------- #
    # Standard single-image path
    # ---------------------------------------------------------------------- #

    def _load_single(self, idx: int) -> Tuple[torch.Tensor, Dict]:
        img, anns = self._load_raw(idx)
        orig_w, orig_h = img.size

        boxes, labels = self._ann_to_tensors(anns, orig_w, orig_h)

        if self.augment:
            # Random horizontal flip
            if random.random() < self.flip_prob:
                img = TF.hflip(img)
                if boxes.numel() > 0:
                    flipped = boxes.clone()
                    flipped[:, 0] = orig_w - boxes[:, 2]
                    flipped[:, 2] = orig_w - boxes[:, 0]
                    boxes = flipped
            # Color jitter
            if self.color_jitter is not None:
                img = self._color_jitter_transform()(img)

        if self.transforms:
            img = self.transforms(img)

        if isinstance(img, torch.Tensor):
            _, th, tw = img.shape
        else:
            tw, th = img.size

        if boxes.numel() > 0:
            sx = tw / orig_w
            sy = th / orig_h
            boxes[:, 0] *= sx
            boxes[:, 1] *= sy
            boxes[:, 2] *= sx
            boxes[:, 3] *= sy
            boxes[:, [0, 2]] = boxes[:, [0, 2]].clamp(0, tw)
            boxes[:, [1, 3]] = boxes[:, [1, 3]].clamp(0, th)
            valid = (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
            boxes  = boxes[valid]
            labels = labels[valid]

        return img, {"boxes": boxes, "labels": labels}

    # ---------------------------------------------------------------------- #

    def __len__(self) -> int:
        return len(self.image_files)

    def __getitem__(
        self, idx: int
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        if self.mosaic and random.random() < self.mosaic_prob:
            return self._load_mosaic(idx)
        return self._load_single(idx)

    # ---------------------------------------------------------------------- #

    def get_class_names(
        self, yaml_path: Optional[Path] = None
    ) -> List[str]:
        """
        Return a list of human-readable class names.

        Loads from *yaml_path* if available; otherwise generates
        ``class_<id>`` placeholders.
        """
        if yaml_path and yaml_path.exists():
            names = load_yaml_classes(yaml_path)
            if names:
                logger.info(f"Class names from YAML: {names}")
                return [
                    names[cid] if cid < len(names) else f"class_{cid}"
                    for cid in self.class_ids
                ]
        logger.warning("No YAML found – using auto-generated class names.")
        return [f"class_{cid}" for cid in self.class_ids]


# ---------------------------------------------------------------------------
# Transforms
# ---------------------------------------------------------------------------

def get_transforms(input_size: int = 512, training: bool = True) -> T.Compose:
    """
    Build the preprocessing pipeline for the given split.

    Both training and validation images are resized and normalised with
    standard statistics before being fed into the model.

    Parameters
    ----------
    input_size : int   Target square resolution.
    training   : bool  (reserved for future data augmentation).
    """
    return T.Compose([
        T.Resize((input_size, input_size)),
        T.ToTensor(),
        T.Normalize(mean=NORM_MEAN, std=NORM_STD),
    ])


# ---------------------------------------------------------------------------
# Lightweight class-info reader (no image loading)
# ---------------------------------------------------------------------------

def read_dataset_info(
    labels_dir: Path,
    yaml_path: Optional[Path] = None,
) -> Tuple[int, List[str]]:
    """
    Return ``(num_classes, class_names)`` by scanning label files only —
    no images are loaded.  Used by ``BILT.train()`` to avoid creating a
    full dataset object just to discover class metadata.

    Parameters
    ----------
    labels_dir : Path  Directory containing ``.txt`` label files.
    yaml_path  : Path  Optional ``data.yaml`` for human-readable names.

    Returns
    -------
    (num_classes, class_names)
    """
    labels_dir = Path(labels_dir)
    class_ids: set = set()
    for lbl in labels_dir.glob("*.txt"):
        try:
            with open(lbl) as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        class_ids.add(int(parts[0]))
        except Exception:
            pass

    sorted_ids = sorted(class_ids)
    num_classes = len(sorted_ids)

    if yaml_path and yaml_path.exists():
        names = load_yaml_classes(yaml_path)
        if names:
            class_names = [
                names[cid] if cid < len(names) else f"class_{cid}"
                for cid in sorted_ids
            ]
            return num_classes, class_names

    return num_classes, [f"class_{cid}" for cid in sorted_ids]


# ---------------------------------------------------------------------------
# Collation
# ---------------------------------------------------------------------------

def collate_fn(batch):
    """Stack images into a batch tensor; keep targets as a list of dicts."""
    images, targets = zip(*batch)
    return torch.stack(images, dim=0), list(targets)


# ---------------------------------------------------------------------------
# DataLoader factory
# ---------------------------------------------------------------------------

def create_dataloader(
    images_dir: Path,
    labels_dir: Path,
    batch_size: int = 4,
    num_workers: int = 0,
    shuffle: bool = True,
    input_size: int = 512,
    training: bool = True,
    pin_memory: bool = False,
    augment: Optional[bool] = None,
    flip_prob: float = 0.5,
    color_jitter: Optional[Tuple[float, float, float, float]] = (0.4, 0.4, 0.4, 0.1),
    cache_images: bool = False,
    mosaic: bool = False,
    mosaic_prob: float = 0.5,
) -> Tuple[DataLoader, int]:
    """
    Create a DataLoader for the given images and labels directories.

    Parameters
    ----------
    cache_images : bool   Pre-load all images into RAM (recommended for
                          small datasets, eliminates disk I/O after epoch 1).
    mosaic       : bool   Apply mosaic 4-image augmentation (training only).
    mosaic_prob  : float  Probability of mosaic per sample.

    Returns
    -------
    (dataloader, num_classes)
    """
    dataset = ObjectDetectionDataset(
        images_dir=images_dir,
        labels_dir=labels_dir,
        transforms=get_transforms(input_size, training),
        input_size=input_size,
        training=training,
        augment=augment,
        flip_prob=flip_prob,
        color_jitter=color_jitter,
        cache_images=cache_images,
        mosaic=mosaic,
        mosaic_prob=mosaic_prob,
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=pin_memory,
        persistent_workers=(num_workers > 0),
    )

    return loader, dataset.num_classes
