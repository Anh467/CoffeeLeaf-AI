"""Unit tests for per-leaf prediction statistics and geometry helpers."""

from __future__ import annotations

import unittest
from pathlib import Path
import sys

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.services.inference_service import resolve_predict_mode
from app.services.segmentation_service import (
    bbox_from_mask,
    build_leaf_instances,
    clamp_bbox,
)
from app.services.statistics_service import build_summary
from app.schemas.response import LeafPrediction, PredictResponse, SummaryInfo


class SummaryStatisticsTests(unittest.TestCase):
    def test_multilabel_leaf_counts(self) -> None:
        leaves = [
            {
                "predicted_labels": ["healthy"],
                "classification_confidence": 0.96,
            },
            {
                "predicted_labels": ["rust"],
                "classification_confidence": 0.89,
            },
            {
                "predicted_labels": ["miner", "rust"],
                "classification_confidence": 0.84,
            },
        ]
        summary = build_summary(
            leaves,
            "healthy",
            disease_classes=["miner", "rust", "phoma"],
            mode="auto",
            fallback_to_single_leaf=False,
        )
        self.assertEqual(summary["total_leaves"], 3)
        self.assertEqual(summary["healthy_leaves"], 1)
        self.assertEqual(summary["diseased_leaves"], 2)
        self.assertEqual(summary["disease_counts"]["miner"], 1)
        self.assertEqual(summary["disease_counts"]["rust"], 2)
        self.assertEqual(summary["disease_counts"]["phoma"], 0)
        self.assertEqual(summary["multi_disease_leaves"], 1)
        self.assertEqual(
            summary["healthy_leaves"] + summary["diseased_leaves"],
            summary["total_leaves"],
        )
        self.assertGreater(
            sum(summary["disease_counts"].values()),
            summary["diseased_leaves"],
        )

    def test_empty_whole_image_summary(self) -> None:
        summary = build_summary(
            [],
            "healthy",
            disease_classes=["miner", "rust", "phoma"],
            mode="whole_image",
        )
        self.assertEqual(summary["total_leaves"], 0)
        self.assertEqual(summary["healthy_leaves"], 0)
        self.assertEqual(summary["diseased_leaves"], 0)
        self.assertEqual(summary["disease_counts"]["miner"], 0)


class GeometryTests(unittest.TestCase):
    def test_clamp_bbox_to_image(self) -> None:
        self.assertEqual(clamp_bbox(-10, -5, 50, 80, 40, 60), [0, 0, 40, 60])
        self.assertEqual(clamp_bbox(30, 20, 10, 5, 100, 100), [10, 5, 30, 20])

    def test_bbox_from_mask_and_stable_leaf_ids(self) -> None:
        width, height = 200, 160
        left = Image.new("L", (width, height), 0)
        right = Image.new("L", (width, height), 0)
        ImageDraw.Draw(left).rectangle([20, 40, 60, 100], fill=255)
        ImageDraw.Draw(right).rectangle([120, 30, 170, 90], fill=255)

        # Intentionally pass right mask first to verify left-to-right sorting.
        instances = build_leaf_instances(
            boxes=[],
            confidences=[],
            masks=[right, left],
            width=width,
            height=height,
            min_crop_size=10,
        )
        self.assertEqual(len(instances), 2)
        self.assertEqual(instances[0]["leaf_id"], 0)
        self.assertEqual(instances[1]["leaf_id"], 1)
        # Left leaf (x=20) should come before right leaf (x=120).
        self.assertEqual(instances[0]["bbox_xyxy"], bbox_from_mask(left))
        self.assertEqual(instances[1]["bbox_xyxy"], bbox_from_mask(right))

    def test_boxes_without_masks(self) -> None:
        instances = build_leaf_instances(
            boxes=[[10, 10, 80, 90], [100, 20, 160, 100]],
            confidences=[0.9, 0.8],
            masks=[],
            width=200,
            height=160,
            min_crop_size=10,
        )
        self.assertEqual(len(instances), 2)
        self.assertTrue(all(item["mask_area"] > 0 for item in instances))


class ModeResolutionTests(unittest.TestCase):
    def test_default_is_auto(self) -> None:
        self.assertEqual(resolve_predict_mode(None), "auto")
        self.assertEqual(resolve_predict_mode(None, allow_single_leaf_fallback=True), "auto")

    def test_legacy_flag_and_aliases(self) -> None:
        self.assertEqual(
            resolve_predict_mode(None, allow_single_leaf_fallback=False),
            "whole_image",
        )
        self.assertEqual(resolve_predict_mode("tree"), "whole_image")
        self.assertEqual(resolve_predict_mode("single_leaf"), "single_leaf")


class SchemaCompatibilityTests(unittest.TestCase):
    def test_leaf_bbox_accepts_list_and_object(self) -> None:
        from_list = LeafPrediction(
            leaf_id=1,
            bbox=[1, 2, 3, 4],
            prediction="rust",
            labels=["rust"],
            confidence=0.9,
            detector_confidence=0.8,
            accepted=True,
            probabilities={"rust": 0.9},
        )
        self.assertEqual(from_list.bbox.x1, 1)
        self.assertEqual(from_list.bbox.y2, 4)

        summary = SummaryInfo(
            total_leaves=0,
            healthy_leaves=0,
            diseased_leaves=0,
            average_confidence=0.0,
            disease_counts={"miner": 0, "rust": 0, "phoma": 0},
        )
        self.assertEqual(summary.disease_counts["phoma"], 0)


if __name__ == "__main__":
    unittest.main()
