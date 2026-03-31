# BILT (Because I Like Twice) - A PyTorch-based object detection library
# Copyright (C) 2026 Rikiza89
# Licensed under the Apache License, Version 2.0

"""
Basic unit tests for BILT.

Run with:  pytest tests/test_basic.py -v
"""

import shutil
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from bilt import BILT
from bilt.utils import parse_bilt_label, validate_dataset_structure
from bilt.variants import get_variant_config, is_variant_name, VARIANT_CONFIGS


# ---------------------------------------------------------------------------
# BILT API tests
# ---------------------------------------------------------------------------

class TestBILT:
    """Test the main BILT class."""

    def test_init_no_weights(self):
        model = BILT()
        assert model.model is None
        assert model.class_names is None

    def test_init_variant_name(self):
        """Passing a variant name stores it but does NOT build the model."""
        for name in ["spark", "flash", "core", "pro", "max"]:
            model = BILT(name)
            assert model.model is None
            assert model.variant == name

    def test_init_variant_aliases(self):
        for alias, canonical in [("n", "spark"), ("s", "flash"), ("m", "core"),
                                  ("l", "pro"), ("x", "max")]:
            model = BILT(alias)
            assert model.variant == canonical

    def test_device_cpu(self):
        model = BILT(device="cpu")
        assert model.device == torch.device("cpu")

    def test_repr_unloaded(self):
        model = BILT("core")
        r = repr(model)
        assert "core" in r
        assert "unloaded" in r

    def test_variants_static(self):
        """BILT.variants() should not raise."""
        BILT.variants()


# ---------------------------------------------------------------------------
# Variant configuration tests
# ---------------------------------------------------------------------------

class TestVariants:
    """Test variant configuration helpers."""

    def test_all_variants_present(self):
        for name in ["spark", "flash", "core", "pro", "max"]:
            cfg = get_variant_config(name)
            assert "backbone" in cfg
            assert "input_size" in cfg
            assert "fpn_channels" in cfg

    def test_aliases_resolve(self):
        assert get_variant_config("n")["backbone"] == get_variant_config("spark")["backbone"]
        assert get_variant_config("x")["backbone"] == get_variant_config("max")["backbone"]

    def test_unknown_variant_raises(self):
        with pytest.raises(ValueError):
            get_variant_config("unknown_variant_xyz")

    def test_is_variant_name(self):
        assert is_variant_name("spark")
        assert is_variant_name("core")
        assert is_variant_name("n")
        assert not is_variant_name("weights/best.pth")
        assert not is_variant_name("somefile.pth")


# ---------------------------------------------------------------------------
# Utility tests
# ---------------------------------------------------------------------------

class TestUtils:
    """Test utility functions."""

    def test_parse_label_basic(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("0 0.5 0.5 0.4 0.6\n")
            f.write("1 0.3 0.3 0.2 0.2\n")
            lbl = Path(f.name)
        try:
            anns = parse_bilt_label(lbl, 640, 480)
            assert len(anns) == 2
            assert anns[0]["class_id"] == 0
            assert anns[1]["class_id"] == 1
            assert len(anns[0]["bbox"]) == 4
        finally:
            lbl.unlink()

    def test_parse_label_missing_file(self):
        anns = parse_bilt_label(Path("nonexistent.txt"), 640, 480)
        assert anns == []

    def test_validate_dataset_valid(self):
        with tempfile.TemporaryDirectory() as tmp:
            dp = Path(tmp)
            for split in ["train", "val"]:
                (dp / split / "images").mkdir(parents=True)
                (dp / split / "labels").mkdir(parents=True)
            # Add at least one image
            Image.new("RGB", (64, 64)).save(dp / "train" / "images" / "a.jpg")
            ok, msg = validate_dataset_structure(dp)
            assert ok

    def test_validate_dataset_invalid(self):
        with tempfile.TemporaryDirectory() as tmp:
            ok, msg = validate_dataset_structure(Path(tmp))
            assert not ok


# ---------------------------------------------------------------------------
# Dataset tests
# ---------------------------------------------------------------------------

class TestDataset:
    """Test ObjectDetectionDataset."""

    def setup_method(self):
        self.tmp = tempfile.mkdtemp()
        dp = Path(self.tmp)
        img_dir = dp / "images"
        lbl_dir = dp / "labels"
        img_dir.mkdir(parents=True)
        lbl_dir.mkdir(parents=True)

        Image.new("RGB", (320, 240), "blue").save(img_dir / "t.jpg")
        (lbl_dir / "t.txt").write_text("0 0.5 0.5 0.3 0.4\n")

        self.img_dir = img_dir
        self.lbl_dir = lbl_dir

    def teardown_method(self):
        shutil.rmtree(self.tmp)

    def test_dataset_loads(self):
        from bilt.dataset import ObjectDetectionDataset, get_transforms

        ds = ObjectDetectionDataset(
            images_dir=self.img_dir,
            labels_dir=self.lbl_dir,
            transforms=get_transforms(320, training=True),
            input_size=320,
        )
        assert len(ds) == 1
        img, target = ds[0]
        assert isinstance(img, torch.Tensor)
        assert img.shape[0] == 3          # RGB channels
        assert "boxes" in target
        assert "labels" in target

    def test_labels_are_one_indexed(self):
        """Labels stored as 1-indexed (0 = background reserved for anchors)."""
        from bilt.dataset import ObjectDetectionDataset, get_transforms

        ds = ObjectDetectionDataset(
            images_dir=self.img_dir,
            labels_dir=self.lbl_dir,
            transforms=get_transforms(320),
            input_size=320,
        )
        _, target = ds[0]
        if target["labels"].numel() > 0:
            assert (target["labels"] >= 1).all()


# ---------------------------------------------------------------------------
# Architecture smoke tests (no training, no weights)
# ---------------------------------------------------------------------------

class TestArchitecture:
    """Instantiate each variant and do a forward pass."""

    @pytest.mark.parametrize("variant", ["spark", "core"])
    def test_detector_forward_inference(self, variant):
        from bilt.core import BILTDetector

        cfg = get_variant_config(variant)
        det = BILTDetector(variant, num_classes=3)
        det.eval()

        h = w = cfg["input_size"]
        x = torch.zeros(1, 3, h, w)
        with torch.no_grad():
            out = det(x)

        assert isinstance(out, list)
        assert len(out) == 1
        assert "boxes" in out[0]
        assert "scores" in out[0]
        assert "labels" in out[0]

    @pytest.mark.parametrize("variant", ["spark", "core"])
    def test_detector_forward_training(self, variant):
        from bilt.core import BILTDetector

        cfg = get_variant_config(variant)
        det = BILTDetector(variant, num_classes=3)
        det.train()

        h = w = cfg["input_size"]
        x = torch.zeros(2, 3, h, w)
        targets = [
            {
                "boxes":  torch.tensor([[10.0, 10.0, 50.0, 50.0]]),
                "labels": torch.tensor([1]),
            }
        ] * 2
        loss = det(x, targets)

        assert "total" in loss
        assert loss["total"].item() >= 0.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
