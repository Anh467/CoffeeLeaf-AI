"""Classification helpers wrapping the warm multi-label classifier."""

from __future__ import annotations

import logging
import time
from typing import Any

from PIL import Image

from pipeline import InferenceBundle, classify_leaf_crops, crop_leaf_instance

LOGGER = logging.getLogger(__name__)


def prepare_leaf_crops(
    image: Image.Image,
    instances: list[dict[str, Any]],
    bundle: InferenceBundle,
) -> dict[str, Any]:
    """Crop / mask-composite leaf instances and build tensors for classification."""
    started = time.perf_counter()
    leaves: list[dict[str, Any]] = []
    crops: list[Image.Image] = []
    tensors: list[Any] = []
    masks: list[Image.Image] = []
    for instance in instances:
        mask = instance["mask"]
        box = instance["bbox_xyxy"]
        crop, bbox = crop_leaf_instance(image, mask, box, bundle.manifest)
        crops.append(crop)
        tensors.append(bundle.transform(crop))
        masks.append(mask)
        leaves.append(
            {
                "leaf_id": int(instance["leaf_id"]),
                "bbox_xyxy": bbox,
                "detector_confidence": float(instance.get("detector_confidence", 0.0)),
                "segmentation_confidence": float(
                    instance.get("segmentation_confidence", instance.get("detector_confidence", 0.0))
                ),
                "mask_area": int(instance.get("mask_area", 0)),
            }
        )
    return {
        "elapsed_ms": (time.perf_counter() - started) * 1000.0,
        "leaves": leaves,
        "crops": crops,
        "tensors": tensors,
        "masks": masks,
    }


def prepare_single_leaf_crop(
    image: Image.Image,
    bundle: InferenceBundle,
) -> dict[str, Any]:
    """Treat the whole RGB image as one leaf (close-up / explicit single-leaf mode)."""
    started = time.perf_counter()
    width, height = image.size
    full_mask = Image.new("L", (width, height), 255)
    crop = image.convert("RGB")
    return {
        "elapsed_ms": (time.perf_counter() - started) * 1000.0,
        "leaves": [
            {
                "leaf_id": 0,
                "bbox_xyxy": [0, 0, width, height],
                "detector_confidence": 1.0,
                "segmentation_confidence": 1.0,
                "mask_area": width * height,
            }
        ],
        "crops": [crop],
        "tensors": [bundle.transform(crop)],
        "masks": [full_mask],
    }


def classify_crops(
    tensors: list[Any],
    leaves: list[dict[str, Any]],
    bundle: InferenceBundle,
) -> dict[str, Any]:
    """Run multi-label disease classification on prepared leaf crops."""
    started = time.perf_counter()
    decoded = classify_leaf_crops(bundle, tensors)
    for leaf, prediction in zip(leaves, decoded):
        leaf.update(prediction)
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    LOGGER.debug("Classified %s leaves in %.1f ms", len(leaves), elapsed_ms)
    return {"elapsed_ms": elapsed_ms, "leaves": leaves}
