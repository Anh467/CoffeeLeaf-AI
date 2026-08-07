from __future__ import annotations

import numpy as np
from PIL import Image

from app.services.segmentation_service import (
    build_leaf_instances,
    deduplicate_leaf_instances,
    mask_overlap_metrics,
)


def _mask(width: int, height: int, x1: int, y1: int, x2: int, y2: int) -> Image.Image:
    array = np.zeros((height, width), dtype=np.uint8)
    array[y1:y2, x1:x2] = 255
    return Image.fromarray(array, mode="L")


def test_mask_overlap_metrics_detect_contained_partial_duplicate() -> None:
    large = _mask(100, 100, 10, 10, 80, 80)
    partial = _mask(100, 100, 30, 30, 70, 70)

    iou, containment = mask_overlap_metrics(large, partial)

    assert 0.30 < iou < 0.35
    assert containment == 1.0


def test_deduplicate_keeps_higher_confidence_instance() -> None:
    large = _mask(100, 100, 10, 10, 80, 80)
    partial = _mask(100, 100, 30, 30, 70, 70)
    instances = [
        {
            "bbox_xyxy": [10, 10, 80, 80],
            "detector_confidence": 0.91,
            "segmentation_confidence": 0.91,
            "mask": large,
            "mask_area": 4900,
            "leaf_id": 0,
        },
        {
            "bbox_xyxy": [30, 30, 70, 70],
            "detector_confidence": 0.47,
            "segmentation_confidence": 0.47,
            "mask": partial,
            "mask_area": 1600,
            "leaf_id": 1,
        },
    ]

    kept = deduplicate_leaf_instances(instances)

    assert len(kept) == 1
    assert kept[0]["detector_confidence"] == 0.91
    assert kept[0]["leaf_id"] == 0


def test_deduplicate_does_not_remove_distinct_leaves() -> None:
    first = _mask(120, 100, 5, 10, 50, 80)
    second = _mask(120, 100, 55, 15, 110, 85)
    instances = [
        {
            "bbox_xyxy": [5, 10, 50, 80],
            "detector_confidence": 0.88,
            "segmentation_confidence": 0.88,
            "mask": first,
            "mask_area": 3150,
            "leaf_id": 0,
        },
        {
            "bbox_xyxy": [55, 15, 110, 85],
            "detector_confidence": 0.86,
            "segmentation_confidence": 0.86,
            "mask": second,
            "mask_area": 3850,
            "leaf_id": 1,
        },
    ]

    kept = deduplicate_leaf_instances(instances)

    assert len(kept) == 2
    assert [item["leaf_id"] for item in kept] == [0, 1]


def test_build_leaf_instances_deduplicates_before_counting() -> None:
    masks = [
        _mask(100, 100, 10, 10, 80, 80),
        _mask(100, 100, 30, 30, 70, 70),
        _mask(100, 100, 82, 10, 98, 50),
    ]
    boxes = [
        [10, 10, 80, 80],
        [30, 30, 70, 70],
        [82, 10, 98, 50],
    ]

    instances = build_leaf_instances(
        boxes=boxes,
        confidences=[0.93, 0.52, 0.89],
        masks=masks,
        width=100,
        height=100,
        min_crop_size=10,
    )

    assert len(instances) == 2
    assert [item["leaf_id"] for item in instances] == [0, 1]
