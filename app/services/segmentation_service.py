"""Segmentation helpers wrapping the warm YOLO detector."""

from __future__ import annotations

import logging
import time
from typing import Any

import numpy as np
from PIL import Image

from pipeline import InferenceBundle

LOGGER = logging.getLogger(__name__)


def clamp_bbox(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    width: int,
    height: int,
) -> list[int]:
    """Clamp xyxy box to image bounds and ensure x1<=x2, y1<=y2."""
    left = int(round(max(0, min(float(x1), float(x2)))))
    right = int(round(min(width, max(float(x1), float(x2)))))
    top = int(round(max(0, min(float(y1), float(y2)))))
    bottom = int(round(min(height, max(float(y1), float(y2)))))
    if right < left:
        left, right = right, left
    if bottom < top:
        top, bottom = bottom, top
    return [left, top, right, bottom]


def bbox_from_mask(mask: Image.Image | np.ndarray) -> list[int] | None:
    """Compute tight xyxy bbox from a binary mask, or None if empty."""
    array = np.asarray(mask)
    if array.ndim == 3:
        array = array[..., 0]
    ys, xs = np.where(array >= 128)
    if len(xs) == 0 or len(ys) == 0:
        return None
    return [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]


def mask_from_bbox(box: list[int], width: int, height: int) -> Image.Image:
    """Create a rectangular binary mask covering ``box``."""
    mask = Image.new("L", (width, height), 0)
    x1, y1, x2, y2 = box
    if x2 > x1 and y2 > y1:
        patch = Image.new("L", (x2 - x1, y2 - y1), 255)
        mask.paste(patch, (x1, y1))
    return mask


def mask_area(mask: Image.Image | np.ndarray) -> int:
    array = np.asarray(mask)
    if array.ndim == 3:
        array = array[..., 0]
    return int(np.count_nonzero(array >= 128))


def mask_overlap_metrics(
    mask_a: Image.Image | np.ndarray,
    mask_b: Image.Image | np.ndarray,
) -> tuple[float, float]:
    """Return (IoU, containment) for two binary instance masks.

    ``containment`` is intersection / smaller-mask-area. It catches duplicate
    predictions where one mask covers only part of the same physical leaf,
    while IoU catches near-identical duplicate masks.
    """
    a = np.asarray(mask_a)
    b = np.asarray(mask_b)
    if a.ndim == 3:
        a = a[..., 0]
    if b.ndim == 3:
        b = b[..., 0]
    a = a >= 128
    b = b >= 128

    area_a = int(np.count_nonzero(a))
    area_b = int(np.count_nonzero(b))
    if area_a == 0 or area_b == 0:
        return 0.0, 0.0

    intersection = int(np.count_nonzero(np.logical_and(a, b)))
    if intersection == 0:
        return 0.0, 0.0

    union = area_a + area_b - intersection
    iou = intersection / union if union > 0 else 0.0
    containment = intersection / min(area_a, area_b)
    return float(iou), float(containment)


def deduplicate_leaf_instances(
    instances: list[dict[str, Any]],
    *,
    iou_threshold: float = 0.60,
    containment_threshold: float = 0.90,
) -> list[dict[str, Any]]:
    """Remove duplicate segmenter predictions for the same physical leaf.

    YOLO NMS handles most duplicate boxes, but instance segmentation can still
    return a second partial mask for the same leaf. We keep the higher-confidence
    prediction when masks are near-identical (high IoU) or one mask is almost
    completely contained inside the other (high containment).
    """
    if len(instances) <= 1:
        return instances

    ranked = sorted(
        instances,
        key=lambda item: (
            float(item.get("detector_confidence", 0.0)),
            int(item.get("mask_area", 0)),
        ),
        reverse=True,
    )
    kept: list[dict[str, Any]] = []

    for candidate in ranked:
        duplicate = False
        for existing in kept:
            iou, containment = mask_overlap_metrics(candidate["mask"], existing["mask"])
            if iou >= iou_threshold or containment >= containment_threshold:
                duplicate = True
                LOGGER.debug(
                    "Dropping duplicate leaf instance conf=%.3f against conf=%.3f "
                    "(mask_iou=%.3f containment=%.3f)",
                    float(candidate.get("detector_confidence", 0.0)),
                    float(existing.get("detector_confidence", 0.0)),
                    iou,
                    containment,
                )
                break
        if not duplicate:
            kept.append(candidate)

    kept.sort(key=lambda item: tuple(item["bbox_xyxy"]))
    for leaf_id, item in enumerate(kept):
        item["leaf_id"] = leaf_id
    return kept


def min_crop_size_from_manifest(manifest: dict[str, Any]) -> int:
    preprocessing = manifest.get("preprocessing") or {}
    if "min_crop_size" in preprocessing:
        return int(preprocessing["min_crop_size"])
    protocol = (manifest.get("dataset") or {}).get("evaluation_protocol") or {}
    return int(protocol.get("min_crop_size", 32))


def build_leaf_instances(
    *,
    boxes: list[list[float]],
    confidences: list[float],
    masks: list[Image.Image],
    width: int,
    height: int,
    min_crop_size: int = 32,
) -> list[dict[str, Any]]:
    """Pair YOLO boxes/masks into stable leaf instances in original image space.

    - Prefer YOLO xyxy boxes; fall back to bbox-from-mask when needed.
    - Prefer mask instances; synthesize a box mask when only boxes exist.
    - Drop empty / too-small instances.
    - Remove duplicate masks for the same physical leaf.
    - Sort left-to-right, top-to-bottom for stable leaf_id assignment.
    """
    count = max(len(boxes), len(masks))
    if count == 0:
        return []

    instances: list[dict[str, Any]] = []
    for index in range(count):
        mask = masks[index] if index < len(masks) else None
        raw_box = boxes[index] if index < len(boxes) else None
        confidence = float(confidences[index]) if index < len(confidences) else 0.0

        if mask is None and raw_box is None:
            continue

        if raw_box is not None:
            box = clamp_bbox(raw_box[0], raw_box[1], raw_box[2], raw_box[3], width, height)
        else:
            derived = bbox_from_mask(mask)
            if derived is None:
                continue
            box = clamp_bbox(derived[0], derived[1], derived[2], derived[3], width, height)

        if mask is None:
            mask = mask_from_bbox(box, width, height)
        elif mask.size != (width, height):
            mask = mask.resize((width, height), Image.Resampling.NEAREST)

        area = mask_area(mask)
        box_w = box[2] - box[0]
        box_h = box[3] - box[1]
        if area <= 0 or box_w < min_crop_size or box_h < min_crop_size:
            continue

        instances.append(
            {
                "bbox_xyxy": box,
                "segmentation_confidence": confidence,
                "detector_confidence": confidence,
                "mask": mask,
                "mask_area": area,
            }
        )

    instances = deduplicate_leaf_instances(instances)
    instances.sort(key=lambda item: (item["bbox_xyxy"][0], item["bbox_xyxy"][1], item["bbox_xyxy"][2], item["bbox_xyxy"][3]))
    for leaf_id, item in enumerate(instances):
        item["leaf_id"] = leaf_id
    return instances


def segment_leaves(
    image: Image.Image,
    bundle: InferenceBundle,
) -> dict[str, Any]:
    """Run leaf instance segmentation and return masks/boxes in image space."""
    rgb = image.convert("RGB")
    width, height = rgb.size
    started = time.perf_counter()
    detection = bundle.detector.predict(
        source=np.asarray(rgb),
        conf=float(bundle.manifest["thresholds"]["detector_confidence"]),
        imgsz=int(bundle.manifest["segmenter"]["image_size"]),
        retina_masks=True,
        verbose=False,
    )
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    if len(detection) != 1:
        raise ValueError("Segmentation currently accepts exactly one image")
    result = detection[0]

    boxes = [] if result.boxes is None else result.boxes.xyxy.cpu().tolist()
    confidences = [] if result.boxes is None else result.boxes.conf.cpu().tolist()
    masks: list[Image.Image] = []
    if result.masks is not None and result.masks.data is not None:
        for mask_array in result.masks.data.cpu().numpy():
            binary = (mask_array >= 0.5).astype(np.uint8) * 255
            mask = Image.fromarray(binary, mode="L")
            if mask.size != (width, height):
                mask = mask.resize((width, height), Image.Resampling.NEAREST)
            masks.append(mask)

    min_size = min_crop_size_from_manifest(bundle.manifest)
    raw_instances = max(len(boxes), len(masks))
    instances = build_leaf_instances(
        boxes=boxes,
        confidences=confidences,
        masks=masks,
        width=width,
        height=height,
        min_crop_size=min_size,
    )
    LOGGER.debug(
        "Segmented raw=%s valid=%s in %.1f ms",
        raw_instances,
        len(instances),
        elapsed_ms,
    )
    return {
        "elapsed_ms": elapsed_ms,
        "raw_instances": raw_instances,
        "valid_instances": len(instances),
        "instances": instances,
        "boxes": [item["bbox_xyxy"] for item in instances],
        "confidences": [item["segmentation_confidence"] for item in instances],
        "masks": [item["mask"] for item in instances],
        "width": width,
        "height": height,
    }
