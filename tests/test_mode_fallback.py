"""Tests for predict mode fallback behavior without requiring GPU models."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from PIL import Image, ImageChops

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.services import visualization_service
from app.services.inference_service import InferenceService
from app.services.statistics_service import build_summary
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


def _fake_classification(label: str = "rust", confidence: float = 0.9) -> dict:
    return {
        "predicted_labels": [label] if label != "healthy" else ["healthy"],
        "primary_label": label,
        "classification_confidence": confidence,
        "accepted": True,
        "probabilities": {
            "healthy": 0.9 if label == "healthy" else 0.1,
            "miner": 0.9 if label == "miner" else 0.1,
            "rust": 0.9 if label == "rust" else 0.1,
            "phoma": 0.9 if label == "phoma" else 0.05,
        },
    }


def _make_instances(count: int, width: int = 320, height: int = 240) -> list[dict]:
    instances = []
    for index in range(count):
        x1 = 10 + index * 40
        y1 = 20
        x2 = x1 + 30
        y2 = y1 + 40
        mask = Image.new("L", (width, height), 0)
        patch = Image.new("L", (x2 - x1, y2 - y1), 255)
        mask.paste(patch, (x1, y1))
        instances.append(
            {
                "leaf_id": index,
                "bbox_xyxy": [x1, y1, x2, y2],
                "segmentation_confidence": 0.8,
                "detector_confidence": 0.8,
                "mask": mask,
                "mask_area": (x2 - x1) * (y2 - y1),
            }
        )
    return instances


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
    def test_single_leaf_skips_segmenter(
        self,
        segment_leaves: MagicMock,
        classify_leaf_crops: MagicMock,
    ) -> None:
        classify_leaf_crops.return_value = [_fake_classification("rust")]
        payload = self.service._run_pipeline(
            image=self.image, bundle=self.bundle, requested_mode="single_leaf"
        )
        segment_leaves.assert_not_called()
        self.assertEqual(payload["requested_mode"], "single_leaf")
        self.assertEqual(payload["resolved_mode"], "single_leaf")
        self.assertFalse(payload["fallback_to_single_leaf"])
        self.assertEqual(payload["timing"]["segmentation_ms"], 0.0)
        self.assertEqual(len(payload["leaves"]), 1)
        self.assertEqual(payload["valid_instances"], 1)
        self.assertEqual(payload["raw_instances"], 0)

    @patch("app.services.classification_service.classify_leaf_crops")
    @patch("app.services.segmentation_service.segment_leaves")
    def test_auto_with_five_valid_instances(
        self,
        segment_leaves: MagicMock,
        classify_leaf_crops: MagicMock,
    ) -> None:
        instances = _make_instances(5)
        segment_leaves.return_value = {
            "elapsed_ms": 25.4,
            "raw_instances": 5,
            "valid_instances": 5,
            "instances": instances,
            "boxes": [item["bbox_xyxy"] for item in instances],
            "confidences": [0.8] * 5,
            "masks": [item["mask"] for item in instances],
            "width": 320,
            "height": 240,
        }
        classify_leaf_crops.return_value = [
            _fake_classification("rust") for _ in range(5)
        ]
        payload = self.service._run_pipeline(
            image=self.image, bundle=self.bundle, requested_mode="auto"
        )
        segment_leaves.assert_called_once()
        self.assertEqual(payload["requested_mode"], "auto")
        self.assertEqual(payload["resolved_mode"], "whole_image")
        self.assertFalse(payload["fallback_to_single_leaf"])
        self.assertEqual(len(payload["leaves"]), 5)
        self.assertEqual(payload["valid_instances"], 5)
        self.assertEqual(payload["raw_instances"], 5)
        summary = build_summary(
            payload["leaves"],
            "healthy",
            disease_classes=["miner", "rust", "phoma"],
        )
        self.assertEqual(summary["total_leaves"], 5)
        self.assertEqual(summary["total_leaves"], len(payload["leaves"]))

    @patch("app.services.classification_service.classify_leaf_crops")
    @patch("app.services.segmentation_service.segment_leaves")
    def test_auto_fallback_keeps_requested_mode(
        self,
        segment_leaves: MagicMock,
        classify_leaf_crops: MagicMock,
    ) -> None:
        segment_leaves.return_value = {
            "elapsed_ms": 26.4,
            "raw_instances": 0,
            "valid_instances": 0,
            "instances": [],
            "boxes": [],
            "confidences": [],
            "masks": [],
            "width": 320,
            "height": 240,
        }
        classify_leaf_crops.return_value = [_fake_classification("rust")]
        payload = self.service._run_pipeline(
            image=self.image, bundle=self.bundle, requested_mode="auto"
        )
        segment_leaves.assert_called_once()
        self.assertEqual(payload["requested_mode"], "auto")
        self.assertEqual(payload["resolved_mode"], "single_leaf")
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
            "raw_instances": 2,
            "valid_instances": 0,
            "instances": [],
            "boxes": [],
            "confidences": [],
            "masks": [],
            "width": 320,
            "height": 240,
        }
        classify_leaf_crops.return_value = []
        payload = self.service._run_pipeline(
            image=self.image, bundle=self.bundle, requested_mode="whole_image"
        )
        self.assertEqual(payload["requested_mode"], "whole_image")
        self.assertEqual(payload["resolved_mode"], "whole_image")
        self.assertFalse(payload["fallback_to_single_leaf"])
        self.assertEqual(payload["leaves"], [])
        self.assertEqual(payload["timing"]["segmentation_ms"], 18.0)
        self.assertEqual(payload["raw_instances"], 2)
        self.assertEqual(payload["valid_instances"], 0)
        classify_leaf_crops.assert_called()


class VisualizationConsistencyTests(unittest.TestCase):
    def test_single_leaf_visualizations_do_not_draw_full_image_box(self) -> None:
        image = Image.new("RGB", (120, 80), color=(10, 20, 30))
        leaf = {
            "leaf_id": 0,
            "bbox_xyxy": [0, 0, 120, 80],
            "predicted_labels": ["rust"],
            "primary_label": "rust",
            "classification_confidence": 0.9,
        }
        mask = Image.new("L", (120, 80), 255)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            overlay = visualization_service.save_prediction_overlay(
                image,
                [leaf],
                [mask],
                "healthy",
                ["miner", "rust", "phoma"],
                root / "overlay.jpg",
                draw_annotations=False,
            )
            boxes = visualization_service.save_boxes_overlay(
                image,
                [leaf],
                root / "boxes.jpg",
                draw_annotations=False,
            )
            mask_path = visualization_service.save_mask_overlay(
                image,
                [leaf],
                [mask],
                "healthy",
                root / "mask.jpg",
                draw_annotations=False,
            )
            for path in (overlay, boxes, mask_path):
                rendered = Image.open(path).convert("RGB")
                diff = ImageChops.difference(rendered, image.convert("RGB"))
                self.assertFalse(diff.getbbox(), f"{path.name} should equal original")

    def test_boxes_overlay_draws_one_box_per_leaf(self) -> None:
        image = Image.new("RGB", (200, 120), color=(8, 8, 8))
        leaves = []
        for index, box in enumerate([[10, 10, 50, 50], [70, 20, 120, 80], [130, 30, 180, 90]]):
            leaves.append(
                {
                    "leaf_id": index,
                    "bbox_xyxy": box,
                    "predicted_labels": ["rust"],
                    "primary_label": "rust",
                    "classification_confidence": 0.8,
                }
            )
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "boxes.jpg"
            visualization_service.save_boxes_overlay(image, leaves, path)
            rendered = Image.open(path).convert("RGB")
            # Annotated image must differ from plain original.
            self.assertTrue(ImageChops.difference(rendered, image).getbbox())


class ConsistencyGuardTests(unittest.TestCase):
    def test_summary_matches_leaf_count(self) -> None:
        leaves = [
            {
                "predicted_labels": ["healthy"],
                "classification_confidence": 0.9,
            },
            {
                "predicted_labels": ["rust"],
                "classification_confidence": 0.8,
            },
        ]
        summary = build_summary(
            leaves, "healthy", disease_classes=["miner", "rust", "phoma"]
        )
        InferenceService._assert_prediction_consistency(summary, leaves)
        self.assertEqual(summary["total_leaves"], len(leaves))
        self.assertEqual(
            summary["healthy_leaves"] + summary["diseased_leaves"],
            summary["total_leaves"],
        )

    def test_inconsistent_summary_raises(self) -> None:
        with self.assertRaises(RuntimeError):
            InferenceService._assert_prediction_consistency(
                {
                    "total_leaves": 2,
                    "healthy_leaves": 1,
                    "diseased_leaves": 1,
                },
                [],
            )


if __name__ == "__main__":
    unittest.main()
