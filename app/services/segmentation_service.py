"""Segmentation helpers wrapping the warm YOLO detector."""

from __future__ import annotations

import logging
import time
from typing import Any

import numpy as np
from PIL import Image

from pipeline import InferenceBundle

LOGGER = logging.getLogger(__name__)


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
        verbose=False,
    )
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    if len(detection) != 1:
        raise ValueError("Segmentation currently accepts exactly one image")
    result = detection[0]

    boxes = [] if result.boxes is None else result.boxes.xyxy.cpu().tolist()
    confidences = [] if result.boxes is None else result.boxes.conf.cpu().tolist()
    masks: list[Image.Image] = []
    if result.masks is not None:
        for mask_tensor in result.masks.data.cpu():
            array = (mask_tensor.numpy() >= 0.5).astype(np.uint8) * 255
            mask = Image.fromarray(array, mode="L").resize(
                (width, height), Image.Resampling.NEAREST
            )
            masks.append(mask)

    LOGGER.debug(
        "Segmented %s leaves in %.1f ms",
        min(len(masks), len(boxes)),
        elapsed_ms,
    )
    return {
        "elapsed_ms": elapsed_ms,
        "boxes": boxes,
        "confidences": confidences,
        "masks": masks,
        "width": width,
        "height": height,
    }
