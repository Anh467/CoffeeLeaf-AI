"""Visualization helpers for dashboard overlays and leaf artifacts."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

from PIL import Image, ImageDraw, ImageFont

from app.services.statistics_service import display_label_for_leaf

# Shared disease/status colors used by overlays, leaf list, and probability UI.
LABEL_COLORS = {
    "healthy": (64, 180, 75),
    "miner": (255, 179, 0),
    "rust": (225, 87, 89),
    "phoma": (116, 78, 170),
    "multi-disease": (30, 136, 229),
    "unknown": (80, 80, 80),
}


def _label_color(name: str) -> tuple[int, int, int]:
    return LABEL_COLORS.get(name, LABEL_COLORS["unknown"])


def leaf_status_color(leaf: dict, healthy_label: str = "healthy") -> tuple[int, int, int]:
    labels = [
        name
        for name in leaf.get("predicted_labels", leaf.get("labels", []))
        if name != healthy_label
    ]
    if not labels:
        return _label_color(healthy_label)
    if len(labels) >= 2:
        return _label_color("multi-disease")
    return _label_color(str(labels[0]))


def leaf_display_label(leaf: dict, healthy_label: str = "healthy") -> str:
    return display_label_for_leaf(leaf, healthy_label)


def _load_font(size: int = 14):
    try:
        return ImageFont.truetype("arial.ttf", size)
    except OSError:
        try:
            return ImageFont.truetype("DejaVuSans.ttf", size)
        except OSError:
            return ImageFont.load_default()


def _draw_label_box(
    drawer: ImageDraw.ImageDraw,
    box: Sequence[int],
    text: str,
    color: tuple[int, int, int],
    image_size: tuple[int, int],
    font: ImageFont.ImageFont,
) -> None:
    width, height = image_size
    x1, y1, x2, y2 = [int(value) for value in box]
    drawer.rectangle([x1, y1, x2, y2], outline=color, width=3)

    text_bbox = drawer.textbbox((0, 0), text, font=font)
    text_w = text_bbox[2] - text_bbox[0]
    text_h = text_bbox[3] - text_bbox[1]
    pad = 3
    label_x = min(max(0, x1), max(0, width - text_w - pad * 2))
    label_y = y1 - text_h - pad * 2
    if label_y < 0:
        label_y = min(y1 + pad, max(0, height - text_h - pad * 2))
    background = [
        label_x,
        label_y,
        label_x + text_w + pad * 2,
        label_y + text_h + pad * 2,
    ]
    drawer.rectangle(background, fill=color)
    drawer.text((label_x + pad, label_y + pad), text, fill=(255, 255, 255), font=font)


def save_plain_image(image: Image.Image, output_path: Path, quality: int = 92) -> Path:
    """Save the RGB image without adding model overlays."""
    canvas = image.convert("RGB")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path, quality=quality)
    return output_path


def save_mask_overlay(
    image: Image.Image,
    leaves: Sequence[dict],
    masks: Sequence[Image.Image],
    healthy_label: str,
    output_path: Path,
    *,
    draw_annotations: bool = True,
) -> Path:
    if not draw_annotations or not leaves or not masks:
        return save_plain_image(image, output_path)
    base = image.convert("RGBA")
    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    for leaf, mask in zip(leaves, masks):
        color = leaf_status_color(leaf, healthy_label) + (100,)
        color_layer = Image.new("RGBA", image.size, color)
        overlay.alpha_composite(
            Image.composite(color_layer, Image.new("RGBA", image.size), mask)
        )
    composed = Image.alpha_composite(base, overlay).convert("RGB")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    composed.save(output_path, quality=92)
    return output_path


def save_boxes_overlay(
    image: Image.Image,
    leaves: Sequence[dict],
    output_path: Path,
    healthy_label: str = "healthy",
    *,
    draw_annotations: bool = True,
) -> Path:
    if not draw_annotations or not leaves:
        return save_plain_image(image, output_path)
    canvas = image.convert("RGB").copy()
    drawer = ImageDraw.Draw(canvas)
    font = _load_font(15)
    for leaf in leaves:
        color = leaf_status_color(leaf, healthy_label)
        box = leaf["bbox_xyxy"]
        leaf_no = int(leaf["leaf_id"]) + 1
        label = leaf_display_label(leaf, healthy_label)
        confidence = float(leaf.get("classification_confidence", 0.0))
        text = f"Leaf #{leaf_no} | {label} | {confidence * 100:.1f}%"
        _draw_label_box(drawer, box, text, color, canvas.size, font)
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
    *,
    draw_annotations: bool = True,
) -> Path:
    del disease_classes  # color mapping is status-based via LABEL_COLORS
    if not draw_annotations or not leaves:
        return save_plain_image(image, output_path)
    base = image.convert("RGBA")
    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    for leaf, mask in zip(leaves, masks):
        color = leaf_status_color(leaf, healthy_label) + (90,)
        color_layer = Image.new("RGBA", image.size, color)
        overlay.alpha_composite(
            Image.composite(color_layer, Image.new("RGBA", image.size), mask)
        )
    annotated = Image.alpha_composite(base, overlay).convert("RGB")
    drawer = ImageDraw.Draw(annotated)
    font = _load_font(15)
    for leaf in leaves:
        color = leaf_status_color(leaf, healthy_label)
        box = leaf["bbox_xyxy"]
        leaf_no = int(leaf["leaf_id"]) + 1
        label = leaf_display_label(leaf, healthy_label)
        confidence = float(leaf.get("classification_confidence", 0.0))
        text = f"Leaf #{leaf_no} | {label} | {confidence * 100:.1f}%"
        _draw_label_box(drawer, box, text, color, annotated.size, font)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    annotated.save(output_path, quality=92)
    return output_path


def save_highlighted_overlay(
    image: Image.Image,
    leaves: Sequence[dict],
    selected_leaf_id: int,
    output_path: Path,
    healthy_label: str = "healthy",
) -> Path:
    """Boxes view with one leaf emphasized (used optionally by clients)."""
    canvas = image.convert("RGB").copy()
    drawer = ImageDraw.Draw(canvas)
    font = _load_font(15)
    for leaf in leaves:
        leaf_no = int(leaf["leaf_id"]) + 1
        is_selected = leaf_no == selected_leaf_id
        color = leaf_status_color(leaf, healthy_label)
        box = leaf["bbox_xyxy"]
        width = 5 if is_selected else 2
        drawer.rectangle(box, outline=color, width=width)
        if is_selected:
            label = leaf_display_label(leaf, healthy_label)
            confidence = float(leaf.get("classification_confidence", 0.0))
            text = f"Leaf #{leaf_no} | {label} | {confidence * 100:.1f}%"
            _draw_label_box(drawer, box, text, color, canvas.size, font)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path, quality=92)
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
