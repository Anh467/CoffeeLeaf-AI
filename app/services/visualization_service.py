"""Visualization helpers for dashboard overlays and leaf artifacts."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

from PIL import Image, ImageDraw

from pipeline import render_annotated_prediction

LABEL_COLORS = {
    "healthy": (64, 180, 75),
    "miner": (255, 179, 0),
    "rust": (225, 87, 89),
    "phoma": (116, 78, 170),
}


def _label_color(name: str) -> tuple[int, int, int]:
    return LABEL_COLORS.get(name, (80, 80, 80))


def save_mask_overlay(
    image: Image.Image,
    leaves: Sequence[dict],
    masks: Sequence[Image.Image],
    healthy_label: str,
    output_path: Path,
) -> Path:
    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    for leaf, mask in zip(leaves, masks):
        color = _label_color(str(leaf.get("primary_label", healthy_label))) + (90,)
        color_layer = Image.new("RGBA", image.size, color)
        overlay.alpha_composite(
            Image.composite(color_layer, Image.new("RGBA", image.size), mask)
        )
    composed = Image.alpha_composite(image.convert("RGBA"), overlay).convert("RGB")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    composed.save(output_path, quality=92)
    return output_path


def save_boxes_overlay(
    image: Image.Image,
    leaves: Sequence[dict],
    output_path: Path,
) -> Path:
    canvas = image.convert("RGB").copy()
    drawer = ImageDraw.Draw(canvas)
    for leaf in leaves:
        color = _label_color(str(leaf.get("primary_label", "unknown")))
        box = leaf["bbox_xyxy"]
        drawer.rectangle(box, outline=color, width=3)
        label = f"#{int(leaf['leaf_id']) + 1}"
        drawer.text((box[0] + 4, max(0, box[1] - 14)), label, fill=color)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path, quality=92)
    return output_path


def save_prediction_overlay(
    image: Image.Image,
    leaves: Sequence[dict],
    masks: Sequence[Image.Image],
    healthy_label: str,
    disease_classes: Sequence[str],
    output_path: Path,
) -> Path:
    annotated = render_annotated_prediction(
        image, leaves, masks, healthy_label, disease_classes
    )
    # Prefer Leaf #N labels for the dashboard overlay.
    drawer = ImageDraw.Draw(annotated)
    for leaf in leaves:
        box = leaf["bbox_xyxy"]
        color = _label_color(str(leaf.get("primary_label", "unknown")))
        drawer.text((box[0] + 4, min(image.size[1] - 14, box[1] + 4)), f"#{int(leaf['leaf_id']) + 1}", fill=color)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    annotated.save(output_path, quality=92)
    return output_path


def save_thumbnail(image: Image.Image, output_path: Path, max_side: int = 320) -> Path:
    canvas = image.convert("RGB").copy()
    canvas.thumbnail((max_side, max_side))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path, quality=85)
    return output_path


def write_leaf_artifacts(
    run_dir: Path,
    leaves: Sequence[dict],
    masks: Sequence[Image.Image],
    crops: Sequence[Image.Image],
) -> list[dict]:
    mask_dir = run_dir / "masks"
    crop_dir = run_dir / "crops"
    mask_dir.mkdir(parents=True, exist_ok=True)
    crop_dir.mkdir(parents=True, exist_ok=True)
    enriched: list[dict] = []
    for leaf, mask, crop in zip(leaves, masks, crops):
        leaf_id = int(leaf["leaf_id"]) + 1
        mask_path = mask_dir / f"leaf_{leaf_id}.png"
        crop_path = crop_dir / f"leaf_{leaf_id}.jpg"
        mask.save(mask_path)
        crop.convert("RGB").save(crop_path, quality=92)
        item = dict(leaf)
        item["mask_path"] = mask_path
        item["crop_path"] = crop_path
        enriched.append(item)
    return enriched
