"""Tests for predict mode fallback behavior without requiring GPU models."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.services.inference_service import InferenceService
from pipeline import InferenceBundle


def _fake_bundle() -> InferenceBundle:
    transform = MagicMock(side_effect=lambda image: "tensor")
    return InferenceBundle(
        deployment_dir=Path("deployment"),
        manifest={
            "deploy_ready": True,
            "schema_version": 3,
            "thresholds": {
                "detector_confidence": 0.25,
                "classifier_confidence": 0.5,
            },
            "segmenter": {"image_size": 640, "candidate": "yolo11n_seg"},
            "classifier": {"candidate": "convnext_tiny"},
            "preprocessing": {
                "crop_padding_ratio": 0.05,
                "mask_background": False,
                "mask_fill_rgb": [0, 0, 0],
            },
        },
        detector=MagicMock(),
        classifier=MagicMock(),
        device="cpu",
        transform=transform,
        disease_classes=["miner", "rust", "phoma"],
        healthy_label="healthy",
        classifier_image_size=224,
    )


class ModeFallbackTests(unittest.TestCase):
    def setUp(self) -> None:
        self.service = InferenceService(history_root=ROOT / "runs" / "history-test")
        self.bundle = _fake_bundle()
        self.image = Image.new("RGB", (320, 240), color=(40, 120, 40))

    def tearDown(self) -> None:
        import shutil

        history = ROOT / "runs" / "history-test"
        if history.exists():
            shutil.rmtree(history, ignore_errors=True)

    @patch("app.services.classification_service.classify_leaf_crops")
    @patch("app.services.segmentation_service.segment_leaves")
    def test_auto_fallback_keeps_segmentation_time(
        self,
        segment_leaves: MagicMock,
        classify_leaf_crops: MagicMock,
    ) -> None:
        segment_leaves.return_value = {
            "elapsed_ms": 26.4,
            "instances": [],
            "boxes": [],
            "confidences": [],
            "masks": [],
            "width": 320,
            "height": 240,
        }
        classify_leaf_crops.return_value = [
            {
                "predicted_labels": ["rust"],
                "primary_label": "rust",
                "classification_confidence": 0.9,
                "accepted": True,
                "probabilities": {
                    "healthy": 0.1,
                    "miner": 0.1,
                    "rust": 0.9,
                    "phoma": 0.05,
                },
            }
        ]
        payload = self.service._run_pipeline(
            image=self.image, bundle=self.bundle, mode="auto"
        )
        self.assertEqual(payload["mode"], "single_leaf")
        self.assertTrue(payload["fallback_to_single_leaf"])
        self.assertEqual(payload["timing"]["segmentation_ms"], 26.4)
        self.assertEqual(len(payload["leaves"]), 1)

    @patch("app.services.classification_service.classify_leaf_crops")
    @patch("app.services.segmentation_service.segment_leaves")
    def test_whole_image_empty_does_not_fallback(
        self,
        segment_leaves: MagicMock,
        classify_leaf_crops: MagicMock,
    ) -> None:
        segment_leaves.return_value = {
            "elapsed_ms": 18.0,
            "instances": [],
            "boxes": [],
            "confidences": [],
            "masks": [],
            "width": 320,
            "height": 240,
        }
        classify_leaf_crops.return_value = []
        payload = self.service._run_pipeline(
            image=self.image, bundle=self.bundle, mode="whole_image"
        )
        self.assertEqual(payload["mode"], "whole_image")
        self.assertFalse(payload["fallback_to_single_leaf"])
        self.assertEqual(payload["leaves"], [])
        self.assertEqual(payload["timing"]["segmentation_ms"], 18.0)
        classify_leaf_crops.assert_called()


if __name__ == "__main__":
    unittest.main()
