"""Reproducible two-stage coffee-leaf disease pipeline.

Preparation intentionally consumes two independent datasets. BRACOT supplies
VIA polygons for class-agnostic leaf instance segmentation, while the Kaggle
Coffee Leaf Diseases dataset supplies leaf images, semantic masks and CSV
multi-label targets. DVC executes the two training notebooks; this module owns
data preparation, model selection, registry publishing, inference and the
shared classifier contract.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib
import json
import math
import os
import random
import shutil
import subprocess
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
CLASSIFIER_ARCHITECTURES = {
    "convnext_tiny",
    "densenet121",
    "efficientnet_v2_s",
    "regnet_y_3_2gf",
}
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


@dataclass(frozen=True)
class Polygon:
    class_id: int
    points: tuple[tuple[float, float], ...]


@dataclass(frozen=True)
class SegmentationRecord:
    image: Path
    polygons: tuple[Polygon, ...]
    split: str
    group_id: str


@dataclass(frozen=True)
class ClassificationRecord:
    image: Path
    mask: Path
    targets: tuple[int, ...]
    split: str
    group_id: str


def project_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def load_config(path: str | Path = "params.yaml") -> dict[str, Any]:
    config_path = project_path(path)
    with config_path.open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, dict):
        raise TypeError(f"Invalid YAML object in {config_path}")
    return config


def require_project_output(path: Path) -> Path:
    resolved = path.resolve()
    if resolved == ROOT or ROOT not in resolved.parents:
        raise ValueError(f"Generated output must be inside the repository: {path}")
    return resolved


def reset_dir(path: Path) -> Path:
    resolved = require_project_output(path)
    if resolved.exists():
        shutil.rmtree(resolved)
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")
    temporary.replace(path)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"Cannot write an empty CSV: {path}")
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def record_asset_id(image: Path, raw_dir: Path) -> str:
    """Keep output names readable while preventing cross-folder collisions."""
    relative = image.relative_to(raw_dir).as_posix()
    suffix = hashlib.sha256(relative.encode()).hexdigest()[:12]
    return f"{image.stem}__{suffix}"


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


def inside(path: Path, parent: Path) -> Path:
    resolved = path.resolve()
    parent_resolved = parent.resolve()
    if resolved != parent_resolved and parent_resolved not in resolved.parents:
        raise ValueError(f"Path escapes raw dataset directory: {path}")
    return resolved


def polygon_area(points: Sequence[tuple[float, float]]) -> float:
    total = 0.0
    for index, (x1, y1) in enumerate(points):
        x2, y2 = points[(index + 1) % len(points)]
        total += x1 * y2 - x2 * y1
    return abs(total) / 2.0


def normalize_token(value: str) -> str:
    return "".join(character for character in value.casefold() if character.isalnum())


def resolve_split_ratios(
    settings: dict[str, Any], splits: Sequence[str]
) -> dict[str, float]:
    configured = settings.get("split_ratios", {})
    if not isinstance(configured, dict) or set(configured) != set(splits):
        raise ValueError(
            "split_ratios must contain exactly these keys: " + ", ".join(splits)
        )
    ratios = {split: float(configured[split]) for split in splits}
    if any(value <= 0.0 for value in ratios.values()) or not math.isclose(
        sum(ratios.values()), 1.0, abs_tol=1e-6
    ):
        raise ValueError("split_ratios values must be positive and sum to 1.0")
    return ratios


def allocated_split_counts(
    total: int, splits: Sequence[str], ratios: dict[str, float]
) -> dict[str, int]:
    if total < len(splits):
        raise ValueError(
            f"Need at least {len(splits)} items to create non-empty splits; got {total}"
        )
    counts = {split: 1 for split in splits}
    remaining = total - len(splits)
    quotas = {split: remaining * ratios[split] for split in splits}
    for split in splits:
        counts[split] += math.floor(quotas[split])
    leftover = total - sum(counts.values())
    ranked = sorted(
        splits,
        key=lambda split: (quotas[split] - math.floor(quotas[split]), split),
        reverse=True,
    )
    for split in ranked[:leftover]:
        counts[split] += 1
    return counts


def deterministic_split(
    identities: Sequence[str],
    splits: Sequence[str],
    ratios: dict[str, float],
    seed: int,
    stratum: str,
) -> dict[str, str]:
    ordered = sorted(identities)
    stratum_seed = int(hashlib.sha256(stratum.encode()).hexdigest()[:8], 16)
    random.Random(seed + stratum_seed).shuffle(ordered)
    counts = allocated_split_counts(len(ordered), splits, ratios)
    assigned: dict[str, str] = {}
    cursor = 0
    for split in splits:
        for identity in ordered[cursor : cursor + counts[split]]:
            assigned[identity] = split
        cursor += counts[split]
    return assigned


def via_entries(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return [
            item
            for item in payload
            if isinstance(item, dict) and "filename" in item and "regions" in item
        ]
    if not isinstance(payload, dict):
        return []
    metadata = payload.get("_via_img_metadata")
    if isinstance(metadata, dict):
        return [item for item in metadata.values() if isinstance(item, dict)]
    return [
        item
        for item in payload.values()
        if isinstance(item, dict) and "filename" in item and "regions" in item
    ]


def load_via_documents(
    raw_dir: Path, settings: dict[str, Any]
) -> list[tuple[Path, list[dict[str, Any]]]]:
    configured = settings.get("annotations_file", "")
    if configured:
        candidates = [inside(raw_dir / str(configured), raw_dir)]
    else:
        candidates = sorted(raw_dir.rglob("*.json"))
    documents: list[tuple[Path, list[dict[str, Any]]]] = []
    for path in candidates:
        try:
            payload = json.loads(path.read_text(encoding="utf-8-sig"))
        except (OSError, json.JSONDecodeError) as error:
            if configured:
                raise ValueError(f"Cannot read VIA annotation {path}: {error}") from error
            continue
        entries = via_entries(payload)
        if entries:
            documents.append((path, entries))
    if not documents:
        location = str(configured) if configured else "a VIA JSON file"
        raise ValueError(
            f"No BRACOT VIA annotations found under {raw_dir}; expected {location}"
        )
    return documents


def resolve_via_image(
    filename: str,
    annotation_path: Path,
    raw_dir: Path,
    images_by_name: dict[str, list[Path]],
) -> Path:
    normalized = filename.replace("\\", "/").strip()
    relative = Path(normalized)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"Unsafe VIA image path {filename!r} in {annotation_path}")
    for base in (annotation_path.parent, raw_dir):
        candidate = inside(base / relative, raw_dir)
        if candidate.is_file():
            return candidate
    matches = images_by_name.get(relative.name, [])
    if len(matches) == 1:
        return matches[0]
    if not matches:
        raise FileNotFoundError(
            f"VIA annotation {annotation_path} references missing image {filename!r}"
        )
    raise ValueError(
        f"VIA filename {filename!r} is ambiguous; use paths relative to {raw_dir}"
    )


def polygons_from_via_entry(
    entry: dict[str, Any], image: Path, annotation_path: Path
) -> tuple[Polygon, ...]:
    regions = entry.get("regions", [])
    if isinstance(regions, dict):
        region_values = list(regions.values())
    elif isinstance(regions, list):
        region_values = regions
    else:
        raise ValueError(f"Invalid VIA regions for {image} in {annotation_path}")
    with Image.open(image) as source:
        width, height = source.size
    if width <= 0 or height <= 0:
        raise ValueError(f"Invalid image dimensions: {image}")
    polygons: list[Polygon] = []
    for region_number, region in enumerate(region_values, start=1):
        if not isinstance(region, dict):
            continue
        shape = region.get("shape_attributes", {})
        if not isinstance(shape, dict):
            continue
        xs = shape.get("all_points_x")
        ys = shape.get("all_points_y")
        if not isinstance(xs, list) or not isinstance(ys, list):
            continue
        if len(xs) != len(ys) or len(xs) < 3:
            raise ValueError(
                f"{annotation_path}: region {region_number} for {image.name} "
                "must contain at least three paired polygon points"
            )
        try:
            points = tuple(
                (
                    min(1.0, max(0.0, float(x) / width)),
                    min(1.0, max(0.0, float(y) / height)),
                )
                for x, y in zip(xs, ys)
            )
        except (TypeError, ValueError) as error:
            raise ValueError(
                f"{annotation_path}: non-numeric polygon for {image.name}"
            ) from error
        if any(not math.isfinite(value) for point in points for value in point):
            raise ValueError(f"{annotation_path}: non-finite polygon for {image.name}")
        if polygon_area(points) <= 1e-8:
            raise ValueError(f"{annotation_path}: zero-area polygon for {image.name}")
        polygons.append(Polygon(0, points))
    if not polygons:
        raise ValueError(f"VIA annotation contains no leaf polygons for {image}")
    return tuple(polygons)


def load_segmentation_manifest(
    raw_dir: Path, settings: dict[str, Any], splits: Sequence[str]
) -> tuple[dict[str, tuple[str, str]], bool]:
    configured = str(settings.get("manifest_file", "")).strip()
    manifest = (
        inside(raw_dir / configured, raw_dir)
        if configured
        else raw_dir / "manifest.csv"
    )
    if not manifest.exists():
        if configured:
            raise FileNotFoundError(manifest)
        return {}, False
    assignments: dict[str, tuple[str, str]] = {}
    with manifest.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        required = {"image", "split", "group_id"}
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(
                f"{manifest} is missing columns: {sorted(missing)}"
            )
        for row_number, row in enumerate(reader, start=2):
            identity = row["image"].strip().replace("\\", "/")
            split = row["split"].strip().lower()
            group_id = row["group_id"].strip()
            if not identity or split not in splits or not group_id:
                raise ValueError(
                    f"{manifest}:{row_number}: image, valid split and group_id are required"
                )
            if identity in assignments:
                raise ValueError(f"{manifest}:{row_number}: duplicate image {identity!r}")
            assignments[identity] = (split, group_id)
    return assignments, True


def load_segmentation_records(
    raw_dir: Path,
    settings: dict[str, Any],
    splits: Sequence[str],
    seed: int,
) -> tuple[list[SegmentationRecord], bool]:
    if not raw_dir.is_dir():
        raise FileNotFoundError(
            f"BRACOT segmentation directory does not exist: {raw_dir}"
        )
    annotation_format = str(settings.get("annotation_format", "via")).lower()
    if annotation_format not in {"via", "auto"}:
        raise ValueError("data.segmentation.annotation_format must be 'via' or 'auto'")
    image_paths = sorted(
        path.resolve()
        for path in raw_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )
    images_by_name: dict[str, list[Path]] = {}
    for image in image_paths:
        images_by_name.setdefault(image.name, []).append(image)

    annotated: dict[Path, tuple[Polygon, ...]] = {}
    for annotation_path, entries in load_via_documents(raw_dir, settings):
        for entry in entries:
            filename = str(entry.get("filename", "")).strip()
            if not filename:
                raise ValueError(f"Missing VIA filename in {annotation_path}")
            image = resolve_via_image(
                filename, annotation_path, raw_dir, images_by_name
            )
            polygons = polygons_from_via_entry(
                entry, image, annotation_path
            )
            if image in annotated and annotated[image] != polygons:
                raise ValueError(f"Image has conflicting VIA annotations: {image}")
            annotated[image] = polygons
    if not annotated:
        raise ValueError(f"No annotated BRACOT images found under {raw_dir}")

    manifest, used_manifest = load_segmentation_manifest(raw_dir, settings, splits)
    identities = {
        image: image.relative_to(raw_dir).as_posix() for image in annotated
    }
    if used_manifest:
        missing = sorted(set(identities.values()).difference(manifest))
        extra = sorted(set(manifest).difference(identities.values()))
        if missing or extra:
            raise ValueError(
                "Segmentation manifest does not match VIA images; "
                f"missing={missing[:5]}, extra={extra[:5]}"
            )
        assignments = {identity: manifest[identity][0] for identity in manifest}
        groups = {identity: manifest[identity][1] for identity in manifest}
    else:
        ratios = resolve_split_ratios(settings, splits)
        assignments = deterministic_split(
            list(identities.values()), splits, ratios, seed, "segmentation"
        )
        groups = {identity: identity for identity in identities.values()}

    records = [
        SegmentationRecord(
            image=image,
            polygons=annotated[image],
            split=assignments[identity],
            group_id=groups[identity],
        )
        for image, identity in identities.items()
    ]
    group_splits: dict[str, set[str]] = {}
    for record in records:
        group_splits.setdefault(record.group_id, set()).add(record.split)
    leaking = {
        group: values for group, values in group_splits.items() if len(values) > 1
    }
    if leaking:
        raise ValueError(
            f"Segmentation group leakage across splits: {dict(list(leaking.items())[:5])}"
        )
    missing_splits = [
        split for split in splits if not any(r.split == split for r in records)
    ]
    if missing_splits:
        raise ValueError(f"Segmentation data is missing splits: {missing_splits}")
    return records, used_manifest


def find_unique_dataset_file(raw_dir: Path, configured: Any, label: str) -> Path:
    relative = Path(str(configured).strip())
    if not relative.name or relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"Invalid {label} path: {configured!r}")
    direct = inside(raw_dir / relative, raw_dir)
    if direct.is_file():
        return direct
    matches = sorted(
        path.resolve()
        for path in raw_dir.rglob(relative.name)
        if path.is_file()
    )
    if len(matches) == 1:
        return matches[0]
    if not matches:
        raise FileNotFoundError(
            f"Cannot find {label} {relative.as_posix()!r} under {raw_dir}"
        )
    raise ValueError(
        f"Multiple {label} files named {relative.name!r} found under {raw_dir}: "
        f"{[str(path) for path in matches[:5]]}"
    )


def index_assets_by_stem(directory: Path, kind: str) -> dict[str, Path]:
    if not directory.is_dir():
        raise FileNotFoundError(f"Missing Kaggle {kind} directory: {directory}")
    assets: dict[str, Path] = {}
    for path in sorted(directory.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        key = normalize_token(path.stem)
        if not key:
            raise ValueError(f"Invalid empty Kaggle sample id for {path}")
        if key in assets:
            raise ValueError(
                f"Duplicate Kaggle {kind} stem {path.stem!r}: "
                f"{assets[key]} and {path}"
            )
        assets[key] = path.resolve()
    if not assets:
        raise ValueError(f"No Kaggle {kind} files found under {directory}")
    return assets


def parse_binary_target(value: Any, path: Path, row_number: int, name: str) -> int:
    try:
        parsed = float(str(value).strip())
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"{path}:{row_number}: target {name!r} must be 0 or 1"
        ) from error
    if parsed not in {0.0, 1.0}:
        raise ValueError(f"{path}:{row_number}: target {name!r} must be 0 or 1")
    return int(parsed)


def load_mask_background_rgb(path: Path, disease_classes: Sequence[str]) -> tuple[int, int, int]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        headers = reader.fieldnames or []
        by_token = {normalize_token(header): header for header in headers}
        required_columns = {"channels", "background", "leaf", *disease_classes}
        missing = [name for name in required_columns if normalize_token(name) not in by_token]
        if missing:
            raise ValueError(f"{path} is missing mask-color columns: {sorted(missing)}")
        rows = list(reader)
    channels: dict[str, int] = {}
    channel_column = by_token["channels"]
    background_column = by_token["background"]
    for row_number, row in enumerate(rows, start=2):
        channel = normalize_token(row.get(channel_column, ""))
        if channel not in {"red", "green", "blue"}:
            continue
        try:
            value = int(str(row.get(background_column, "")).strip())
        except (TypeError, ValueError) as error:
            raise ValueError(
                f"{path}:{row_number}: invalid background color value"
            ) from error
        if not 0 <= value <= 255:
            raise ValueError(f"{path}:{row_number}: background color must be 0..255")
        channels[channel] = value
    if set(channels) != {"red", "green", "blue"}:
        raise ValueError(f"{path} must define red, green and blue mask channels")
    return channels["red"], channels["green"], channels["blue"]


def load_kaggle_label_rows(
    path: Path,
    dataset_root: Path,
    source_split: str,
    settings: dict[str, Any],
    disease_classes: Sequence[str],
) -> list[tuple[Path, Path, tuple[int, ...], str]]:
    image_dir = dataset_root / source_split / str(settings.get("images_dir", "images"))
    mask_dir = dataset_root / source_split / str(settings.get("masks_dir", "masks"))
    images = index_assets_by_stem(image_dir, f"{source_split} images")
    masks = index_assets_by_stem(mask_dir, f"{source_split} masks")
    rows: list[tuple[Path, Path, tuple[int, ...], str]] = []
    seen: set[str] = set()
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        headers = reader.fieldnames or []
        by_token = {normalize_token(header): header for header in headers}
        id_token = normalize_token(str(settings.get("id_column", "id")))
        missing = [
            name
            for name in (id_token, *(normalize_token(name) for name in disease_classes))
            if name not in by_token
        ]
        if missing:
            raise ValueError(f"{path} is missing label columns: {sorted(missing)}")
        for row_number, row in enumerate(reader, start=2):
            raw_id = str(row.get(by_token[id_token], "")).strip()
            sample_key = normalize_token(Path(raw_id).stem)
            if not sample_key:
                raise ValueError(f"{path}:{row_number}: sample id is empty")
            if sample_key in seen:
                raise ValueError(f"{path}:{row_number}: duplicate sample id {raw_id!r}")
            seen.add(sample_key)
            image = images.get(sample_key)
            mask = masks.get(sample_key)
            if image is None or mask is None:
                raise FileNotFoundError(
                    f"{path}:{row_number}: sample {raw_id!r} has no matching "
                    f"image or mask in {source_split}/"
                )
            targets = tuple(
                parse_binary_target(
                    row.get(by_token[normalize_token(name)]), path, row_number, name
                )
                for name in disease_classes
            )
            rows.append((image, mask, targets, f"{source_split}:{sample_key}"))
    unused_images = sorted(set(images).difference(seen))
    unused_masks = sorted(set(masks).difference(seen))
    if unused_images or unused_masks:
        raise ValueError(
            f"{path} does not cover every {source_split} asset; "
            f"images={unused_images[:5]}, masks={unused_masks[:5]}"
        )
    return rows


def load_classification_records(
    raw_dir: Path,
    settings: dict[str, Any],
    disease_classes: Sequence[str],
    splits: Sequence[str],
    seed: int,
) -> tuple[list[ClassificationRecord], tuple[int, int, int]]:
    if not raw_dir.is_dir():
        raise FileNotFoundError(
            f"Kaggle classification directory does not exist: {raw_dir}"
        )
    if list(splits) != ["train", "val", "test"]:
        raise ValueError("Kaggle loader requires splits [train, val, test]")
    train_csv = find_unique_dataset_file(
        raw_dir, settings.get("train_labels_file", "train_classes.csv"), "train labels"
    )
    test_csv = find_unique_dataset_file(
        raw_dir, settings.get("test_labels_file", "test_classes.csv"), "test labels"
    )
    colors_csv = find_unique_dataset_file(
        raw_dir, settings.get("mask_colors_file", "mask_colors.csv"), "mask colors"
    )
    if train_csv.parent != test_csv.parent or train_csv.parent != colors_csv.parent:
        raise ValueError(
            "Kaggle train_classes.csv, test_classes.csv and mask_colors.csv "
            "must share one dataset directory"
        )
    dataset_root = train_csv.parent
    background_rgb = load_mask_background_rgb(colors_csv, disease_classes)
    train_rows = load_kaggle_label_rows(
        train_csv, dataset_root, "train", settings, disease_classes
    )
    test_rows = load_kaggle_label_rows(
        test_csv, dataset_root, "test", settings, disease_classes
    )

    validation_fraction = float(settings.get("validation_fraction", 0.15))
    if not 0.0 < validation_fraction < 0.5:
        raise ValueError("data.classification.validation_fraction must be between 0 and 0.5")
    assignments: dict[str, str] = {}
    by_signature: dict[str, list[str]] = {}
    for _, _, targets, identity in train_rows:
        signature = "".join(str(value) for value in targets)
        by_signature.setdefault(signature, []).append(identity)
    for signature, identities in by_signature.items():
        if len(identities) == 1:
            assignments[identities[0]] = "train"
            continue
        assignments.update(
            deterministic_split(
                identities,
                ("train", "val"),
                {"train": 1.0 - validation_fraction, "val": validation_fraction},
                seed,
                f"classification:{signature}",
            )
        )

    records = [
        ClassificationRecord(image, mask, targets, assignments[identity], identity)
        for image, mask, targets, identity in train_rows
    ]
    records.extend(
        ClassificationRecord(image, mask, targets, "test", identity)
        for image, mask, targets, identity in test_rows
    )
    for record in records:
        with Image.open(record.image) as image, Image.open(record.mask) as mask:
            if image.size != mask.size:
                raise ValueError(
                    f"Kaggle image/mask dimensions differ for {record.group_id}: "
                    f"{image.size} != {mask.size}"
                )
    missing_splits = [
        split for split in splits if not any(record.split == split for record in records)
    ]
    missing_labels = [
        f"{split}/{name}"
        for split in splits
        for index, name in enumerate(disease_classes)
        if not any(record.split == split and record.targets[index] for record in records)
    ]
    missing_healthy = [
        split
        for split in splits
        if not any(record.split == split and not any(record.targets) for record in records)
    ]
    if missing_splits or missing_labels or missing_healthy:
        raise ValueError(
            "Kaggle labels do not cover the required evaluation contract; "
            f"missing_splits={missing_splits}, missing_labels={missing_labels}, "
            f"missing_healthy={missing_healthy}"
        )
    return records, background_rgb


def segmentation_fingerprint(
    records: Sequence[SegmentationRecord], raw_dir: Path
) -> str:
    digest = hashlib.sha256()
    for record in sorted(
        records,
        key=lambda item: (
            item.split,
            item.group_id,
            item.image.relative_to(raw_dir).as_posix(),
        ),
    ):
        payload = {
            "group_id": record.group_id,
            "image": record.image.relative_to(raw_dir).as_posix(),
            "polygons": [polygon.points for polygon in record.polygons],
            "split": record.split,
        }
        digest.update(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        )
        digest.update(bytes.fromhex(sha256_file(record.image)))
    return digest.hexdigest()


def classification_fingerprint(
    records: Sequence[ClassificationRecord], raw_dir: Path
) -> str:
    digest = hashlib.sha256()
    for record in sorted(
        records,
        key=lambda item: (
            item.split,
            item.targets,
            item.image.relative_to(raw_dir).as_posix(),
        ),
    ):
        payload = {
            "group_id": record.group_id,
            "image": record.image.relative_to(raw_dir).as_posix(),
            "mask": record.mask.relative_to(raw_dir).as_posix(),
            "split": record.split,
            "targets": record.targets,
        }
        digest.update(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        )
        digest.update(bytes.fromhex(sha256_file(record.image)))
        digest.update(bytes.fromhex(sha256_file(record.mask)))
    return digest.hexdigest()


def combined_fingerprint(*values: str) -> str:
    return hashlib.sha256("\n".join(values).encode()).hexdigest()


def prepare(config: dict[str, Any]) -> None:
    data = config["data"]
    disease_classes = list(data["disease_classes"])
    healthy_label = str(data["healthy_label"])
    output_classes = [healthy_label, *disease_classes]
    splits = list(data["splits"])
    seed = int(config["seed"])
    segmentation_settings = data["segmentation"]
    classification_settings = data["classification"]
    segmentation_raw = project_path(segmentation_settings["raw_dir"])
    classification_raw = project_path(classification_settings["raw_dir"])

    segmentation_records, used_segmentation_manifest = load_segmentation_records(
        segmentation_raw, segmentation_settings, splits, seed
    )
    classification_records, mask_background_rgb = load_classification_records(
        classification_raw, classification_settings, disease_classes, splits, seed
    )
    seed_everything(seed)

    segmentation_full_fingerprint = segmentation_fingerprint(
        segmentation_records, segmentation_raw
    )
    classification_full_fingerprint = classification_fingerprint(
        classification_records, classification_raw
    )
    full_fingerprint = combined_fingerprint(
        segmentation_full_fingerprint, classification_full_fingerprint
    )
    segmentation_validation_fingerprint = segmentation_fingerprint(
        [record for record in segmentation_records if record.split == "val"],
        segmentation_raw,
    )
    classification_validation_fingerprint = classification_fingerprint(
        [record for record in classification_records if record.split == "val"],
        classification_raw,
    )
    evaluation_data_fingerprint = combined_fingerprint(
        segmentation_validation_fingerprint, classification_validation_fingerprint
    )
    evaluation_protocol = {
        "classification_mode": "multilabel",
        "disease_classes": disease_classes,
        "healthy_label": healthy_label,
        "crop_padding_ratio": float(data["crop_padding_ratio"]),
        "mask_background": bool(data["mask_background"]),
        "mask_background_rgb": list(mask_background_rgb),
        "mask_fill_rgb": [int(value) for value in data["mask_fill_rgb"]],
        "min_crop_size": int(data["min_crop_size"]),
    }
    evaluation_fingerprint = hashlib.sha256(
        (
            evaluation_data_fingerprint
            + json.dumps(evaluation_protocol, sort_keys=True, separators=(",", ":"))
        ).encode()
    ).hexdigest()

    output_dir = reset_dir(project_path(data["processed_dir"]))
    segment_root = output_dir / "segmentation"
    classifier_root = output_dir / "classification"
    label_counts: Counter[tuple[str, str]] = Counter()
    healthy_counts: Counter[str] = Counter()
    multilabel_counts: Counter[str] = Counter()
    segmentation_image_counts: Counter[str] = Counter()
    segmentation_instance_counts: Counter[str] = Counter()

    for record in segmentation_records:
        asset_id = record_asset_id(record.image, segmentation_raw)
        image_target = (
            segment_root
            / "images"
            / record.split
            / f"{asset_id}{record.image.suffix.lower()}"
        )
        label_target = segment_root / "labels" / record.split / f"{asset_id}.txt"
        image_target.parent.mkdir(parents=True, exist_ok=True)
        label_target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(record.image, image_target)
        label_target.write_text(
            "\n".join(
                "0 "
                + " ".join(
                    f"{coordinate:.8f}"
                    for point in polygon.points
                    for coordinate in point
                )
                for polygon in record.polygons
            )
            + "\n",
            encoding="utf-8",
        )
        segmentation_image_counts[record.split] += 1
        segmentation_instance_counts[record.split] += len(record.polygons)

    classification_rows: list[dict[str, Any]] = []
    for record in classification_records:
        asset_id = record_asset_id(record.image, classification_raw)
        image_target = (
            classifier_root
            / "images"
            / record.split
            / f"{asset_id}{record.image.suffix.lower()}"
        )
        mask_target = (
            classifier_root
            / "masks"
            / record.split
            / f"{asset_id}{record.mask.suffix.lower()}"
        )
        image_target.parent.mkdir(parents=True, exist_ok=True)
        mask_target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(record.image, image_target)
        shutil.copy2(record.mask, mask_target)
        is_healthy = not any(record.targets)
        healthy_counts[record.split] += int(is_healthy)
        multilabel_counts[record.split] += int(sum(record.targets) > 1)
        row: dict[str, Any] = {
            "image": image_target.relative_to(classifier_root).as_posix(),
            "mask": mask_target.relative_to(classifier_root).as_posix(),
            "split": record.split,
            "group_id": record.group_id,
            healthy_label: int(is_healthy),
        }
        for name, target in zip(disease_classes, record.targets):
            row[name] = target
            label_counts[(record.split, name)] += target
        classification_rows.append(row)

    write_csv(classifier_root / "labels.csv", classification_rows)
    write_json(
        classifier_root / "dataset.json",
        {
            "schema_version": 1,
            "classification_mode": "multilabel",
            "disease_classes": disease_classes,
            "healthy_label": healthy_label,
            "mask_background_rgb": list(mask_background_rgb),
            "mask_fill_rgb": [int(value) for value in data["mask_fill_rgb"]],
        },
    )

    dataset_yaml = {
        "path": ".",
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": {0: "leaf"},
    }
    with (segment_root / "dataset.yaml").open("w", encoding="utf-8") as stream:
        yaml.safe_dump(dataset_yaml, stream, sort_keys=False, allow_unicode=True)

    missing_splits = [
        split for split in splits if segmentation_image_counts[split] == 0
    ]
    missing_labels = [
        f"{split}/{class_name}"
        for split in splits
        for class_name in disease_classes
        if label_counts[(split, class_name)] == 0
    ]
    missing_healthy = [split for split in splits if healthy_counts[split] == 0]
    if missing_splits or missing_labels or missing_healthy:
        raise ValueError(
            f"Incomplete prepared dataset; missing splits={missing_splits}, "
            f"missing disease labels={missing_labels}, missing healthy={missing_healthy}"
        )

    metrics: dict[str, Any] = {
        "segmentation_images_total": sum(segmentation_image_counts.values()),
        "segmentation_instances_total": sum(segmentation_instance_counts.values()),
        "classification_images_total": len(classification_records),
        "used_segmentation_manifest": int(used_segmentation_manifest),
    }
    for split in splits:
        metrics[f"segmentation_images_{split}"] = segmentation_image_counts[split]
        metrics[f"segmentation_instances_{split}"] = segmentation_instance_counts[split]
        metrics[f"classification_images_{split}"] = sum(
            record.split == split for record in classification_records
        )
        metrics[f"classification_images_{split}_{healthy_label}"] = healthy_counts[split]
        metrics[f"classification_images_{split}_multi_disease"] = multilabel_counts[split]
        for name in disease_classes:
            metrics[f"classification_positive_{split}_{name}"] = label_counts[(split, name)]
    write_json(
        output_dir / "dataset_manifest.json",
        {
            "schema_version": 3,
            "classification_mode": "multilabel",
            "classes": output_classes,
            "disease_classes": disease_classes,
            "healthy_label": healthy_label,
            "dataset_fingerprint": full_fingerprint,
            "evaluation_data_fingerprint": evaluation_data_fingerprint,
            "evaluation_fingerprint": evaluation_fingerprint,
            "evaluation_protocol": evaluation_protocol,
            "segmentation": {
                "source_name": segmentation_settings.get("source_name", "BRACOT"),
                "source_url": segmentation_settings.get("source_url", ""),
                "fingerprint": segmentation_full_fingerprint,
                "records": len(segmentation_records),
                "instances": sum(len(record.polygons) for record in segmentation_records),
                "splits": {
                    split: segmentation_image_counts[split] for split in splits
                },
                "used_group_manifest": used_segmentation_manifest,
            },
            "classification": {
                "source_name": classification_settings.get(
                    "source_name", "Coffee Leaf Disease"
                ),
                "source_url": classification_settings.get("source_url", ""),
                "fingerprint": classification_full_fingerprint,
                "records": len(classification_records),
                "splits": {
                    split: sum(record.split == split for record in classification_records)
                    for split in splits
                },
                "positive_counts": {
                    split: {
                        healthy_label: healthy_counts[split],
                        **{
                            name: label_counts[(split, name)]
                            for name in disease_classes
                        },
                        "multi_disease": multilabel_counts[split],
                    }
                    for split in splits
                },
                "mask_background_rgb": list(mask_background_rgb),
            },
        },
    )
    write_json(ROOT / "metrics" / "data.json", metrics)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


def build_classifier(
    name: str,
    number_of_classes: int,
    dropout: float,
    unfreeze_blocks: int,
    pretrained: bool,
) -> tuple[Any, Any]:
    from torch import nn
    from torchvision.models import (
        ConvNeXt_Tiny_Weights,
        DenseNet121_Weights,
        EfficientNet_V2_S_Weights,
        RegNet_Y_3_2GF_Weights,
        convnext_tiny,
        densenet121,
        efficientnet_v2_s,
        regnet_y_3_2gf,
    )

    if name == "efficientnet_v2_s":
        weights = EfficientNet_V2_S_Weights.DEFAULT if pretrained else None
        model = efficientnet_v2_s(weights=weights)
        for parameter in model.features.parameters():
            parameter.requires_grad = False
        blocks = list(model.features.children())
        if unfreeze_blocks > 0:
            for block in blocks[-unfreeze_blocks:]:
                for parameter in block.parameters():
                    parameter.requires_grad = True
        input_features = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(dropout), nn.Linear(input_features, number_of_classes)
        )
        head = model.classifier
    elif name == "regnet_y_3_2gf":
        weights = RegNet_Y_3_2GF_Weights.DEFAULT if pretrained else None
        model = regnet_y_3_2gf(weights=weights)
        for parameter in model.parameters():
            parameter.requires_grad = False
        blocks = list(model.trunk_output.children())
        if unfreeze_blocks > 0:
            for block in blocks[-unfreeze_blocks:]:
                for parameter in block.parameters():
                    parameter.requires_grad = True
        input_features = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(dropout), nn.Linear(input_features, number_of_classes)
        )
        head = model.fc
    elif name == "convnext_tiny":
        weights = ConvNeXt_Tiny_Weights.DEFAULT if pretrained else None
        model = convnext_tiny(weights=weights)
        for parameter in model.features.parameters():
            parameter.requires_grad = False
        blocks = list(model.features.children())
        if unfreeze_blocks > 0:
            for block in blocks[-unfreeze_blocks:]:
                for parameter in block.parameters():
                    parameter.requires_grad = True
        input_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Sequential(
            nn.Dropout(dropout), nn.Linear(input_features, number_of_classes)
        )
        head = model.classifier
    elif name == "densenet121":
        weights = DenseNet121_Weights.DEFAULT if pretrained else None
        model = densenet121(weights=weights)
        for parameter in model.features.parameters():
            parameter.requires_grad = False
        blocks = list(model.features.children())
        if unfreeze_blocks > 0:
            for block in blocks[-unfreeze_blocks:]:
                for parameter in block.parameters():
                    parameter.requires_grad = True
        input_features = model.classifier.in_features
        model.classifier = nn.Sequential(
            nn.Dropout(dropout), nn.Linear(input_features, number_of_classes)
        )
        head = model.classifier
    else:
        raise ValueError(f"Unsupported classifier candidate: {name}")

    for parameter in head.parameters():
        parameter.requires_grad = True
    return model, head


def lower_is_better(value: float, values: Sequence[float]) -> float:
    minimum, maximum = min(values), max(values)
    if math.isclose(minimum, maximum):
        return 1.0
    return 1.0 - (value - minimum) / (maximum - minimum)


def deployment_score(
    segmenter_metrics: dict[str, Any],
    classifier_metrics: dict[str, Any],
    weights: dict[str, Any],
) -> float:
    """Return a stable validation score that is comparable between runs."""
    values = {
        "segmenter_map50_95": float(segmenter_metrics["map50_95"]),
        "segmenter_recall": float(segmenter_metrics["recall"]),
        "classifier_macro_f1": float(classifier_metrics["macro_f1"]),
        "classifier_min_class_recall": float(classifier_metrics["min_class_recall"]),
    }
    return sum(float(weights[name]) * value for name, value in values.items())


def score_candidates(
    candidates: dict[str, dict[str, Any]],
    kind: str,
    selection: dict[str, Any],
) -> tuple[str, dict[str, dict[str, Any]]]:
    if not candidates:
        raise ValueError(f"No {kind} candidates to select")
    settings = selection[kind]
    latency_values = [float(item["latency_ms"]) for item in candidates.values()]
    size_values = [float(item["size_mb"]) for item in candidates.values()]
    scores: dict[str, dict[str, Any]] = {}
    for name, metrics in candidates.items():
        latency_score = lower_is_better(float(metrics["latency_ms"]), latency_values)
        size_score = lower_is_better(float(metrics["size_mb"]), size_values)
        if kind == "segmentation":
            passed = float(metrics["map50_95"]) >= float(
                settings["min_map50_95"]
            ) and float(metrics["recall"]) >= float(settings["min_recall"])
            weights = settings["weights"]
            score = (
                float(weights["map50_95"]) * float(metrics["map50_95"])
                + float(weights["recall"]) * float(metrics["recall"])
                + float(weights["latency"]) * latency_score
                + float(weights["size"]) * size_score
            )
        else:
            passed = float(metrics["macro_f1"]) >= float(
                settings["min_macro_f1"]
            ) and float(metrics["min_class_recall"]) >= float(
                settings["min_class_recall"]
            )
            weights = settings["weights"]
            score = (
                float(weights["macro_f1"]) * float(metrics["macro_f1"])
                + float(weights["min_class_recall"])
                * float(metrics["min_class_recall"])
                + float(weights["latency"]) * latency_score
                + float(weights["size"]) * size_score
            )
        scores[name] = {
            "score": score,
            "quality_gate_passed": passed,
            "latency_score": latency_score,
            "size_score": size_score,
        }
    eligible = [name for name, value in scores.items() if value["quality_gate_passed"]]
    pool = eligible or list(scores)
    winner = max(pool, key=lambda name: (scores[name]["score"], name))
    return winner, scores


def export_onnx_models(
    deployment_dir: Path, manifest: dict[str, Any], opset: int
) -> dict[str, str]:
    try:
        import torch
        from ultralytics import YOLO
    except ImportError as error:
        raise RuntimeError(
            "ONNX export requires the full training environment"
        ) from error

    segmenter = YOLO(str(deployment_dir / "leaf_segmenter.pt"))
    exported_segmenter = Path(
        segmenter.export(
            format="onnx",
            imgsz=int(manifest["segmenter"]["image_size"]),
            opset=opset,
            simplify=True,
            dynamic=True,
        )
    )
    segmenter_target = deployment_dir / "leaf_segmenter.onnx"
    if exported_segmenter.resolve() != segmenter_target.resolve():
        shutil.move(exported_segmenter, segmenter_target)

    checkpoint = torch.load(
        deployment_dir / "leaf_classifier.pt", map_location="cpu", weights_only=False
    )
    classifier, _ = build_classifier(
        checkpoint["model_name"],
        len(checkpoint["disease_classes"]),
        float(checkpoint["dropout"]),
        int(checkpoint["unfreeze_blocks"]),
        False,
    )
    classifier.load_state_dict(checkpoint["state_dict"])
    classifier.eval()
    classifier_target = deployment_dir / "leaf_classifier.onnx"
    dummy = torch.randn(
        1, 3, int(checkpoint["image_size"]), int(checkpoint["image_size"])
    )
    torch.onnx.export(
        classifier,
        dummy,
        classifier_target,
        input_names=["image"],
        output_names=["logits"],
        dynamic_axes={"image": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=opset,
    )
    return {
        "segmenter_onnx": sha256_file(segmenter_target),
        "classifier_onnx": sha256_file(classifier_target),
    }


def select_models(config: dict[str, Any]) -> None:
    segmenter_report = json.loads(
        (ROOT / "metrics" / "segmenters.json").read_text(encoding="utf-8")
    )
    classifier_report = json.loads(
        (ROOT / "metrics" / "classifiers.json").read_text(encoding="utf-8")
    )
    expected_disease_classes = list(config["data"]["disease_classes"])
    expected_healthy_label = str(config["data"]["healthy_label"])
    if classifier_report.get("classification_mode") != "multilabel":
        raise ValueError("Classifier metrics are not from a multi-label run")
    if list(classifier_report.get("disease_classes", [])) != expected_disease_classes:
        raise ValueError("Classifier metric class order differs from params.yaml")
    if str(classifier_report.get("healthy_label")) != expected_healthy_label:
        raise ValueError("Classifier metric healthy label differs from params.yaml")
    if not math.isclose(
        float(classifier_report.get("decision_threshold", -1.0)),
        float(config["deployment"]["classifier_confidence"]),
        abs_tol=1e-9,
    ):
        raise ValueError("Classifier metric threshold differs from params.yaml")
    segmenter_candidates = segmenter_report["candidates"]
    classifier_candidates = classifier_report["candidates"]
    dataset_manifest = json.loads(
        (
            project_path(config["data"]["processed_dir"]) / "dataset_manifest.json"
        ).read_text(encoding="utf-8")
    )
    segmenter_name, segmenter_scores = score_candidates(
        segmenter_candidates, "segmentation", config["selection"]
    )
    classifier_name, classifier_scores = score_candidates(
        classifier_candidates, "classification", config["selection"]
    )
    deployment_dir = reset_dir(ROOT / "deployment")
    segmenter_source = ROOT / "models" / "segmenters" / f"{segmenter_name}.pt"
    classifier_source = ROOT / "models" / "classifiers" / f"{classifier_name}.pt"
    segmenter_target = deployment_dir / "leaf_segmenter.pt"
    classifier_target = deployment_dir / "leaf_classifier.pt"
    shutil.copy2(segmenter_source, segmenter_target)
    shutil.copy2(classifier_source, classifier_target)

    segmenter_passed = bool(segmenter_scores[segmenter_name]["quality_gate_passed"])
    classifier_passed = bool(classifier_scores[classifier_name]["quality_gate_passed"])
    stable_score = deployment_score(
        segmenter_candidates[segmenter_name],
        classifier_candidates[classifier_name],
        config["selection"]["deployment_weights"],
    )
    disease_classes = list(config["data"]["disease_classes"])
    healthy_label = str(config["data"]["healthy_label"])
    manifest: dict[str, Any] = {
        "schema_version": 3,
        "deploy_ready": segmenter_passed and classifier_passed,
        "deployment_score": stable_score,
        "dataset": dataset_manifest,
        "classification_mode": "multilabel",
        "classes": [healthy_label, *disease_classes],
        "disease_classes": disease_classes,
        "healthy_label": healthy_label,
        "preprocessing": {
            "mask_background": bool(config["data"]["mask_background"]),
            "mask_fill_rgb": list(config["data"]["mask_fill_rgb"]),
            "crop_padding_ratio": float(config["data"]["crop_padding_ratio"]),
            "imagenet_mean": list(IMAGENET_MEAN),
            "imagenet_std": list(IMAGENET_STD),
        },
        "thresholds": {
            "detector_confidence": float(config["deployment"]["detector_confidence"]),
            "classifier_confidence": float(
                config["deployment"]["classifier_confidence"]
            ),
        },
        "segmenter": {
            "candidate": segmenter_name,
            "score": segmenter_scores[segmenter_name]["score"],
            "quality_gate_passed": segmenter_passed,
            "image_size": int(segmenter_candidates[segmenter_name]["image_size"]),
            "sha256": sha256_file(segmenter_target),
            "metrics": segmenter_candidates[segmenter_name],
        },
        "classifier": {
            "candidate": classifier_name,
            "architecture": str(
                config["classification"]["candidates"][classifier_name].get(
                    "architecture", classifier_name
                )
            ),
            "score": classifier_scores[classifier_name]["score"],
            "quality_gate_passed": classifier_passed,
            "image_size": int(classifier_candidates[classifier_name]["image_size"]),
            "sha256": sha256_file(classifier_target),
            "metrics": classifier_candidates[classifier_name],
        },
    }
    if bool(config["deployment"].get("export_onnx", False)):
        manifest["onnx"] = export_onnx_models(
            deployment_dir, manifest, int(config["deployment"]["onnx_opset"])
        )
    manifest["bundle_sha256"] = hashlib.sha256(
        json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    write_json(deployment_dir / "manifest.json", manifest)

    leaderboard: list[dict[str, Any]] = []
    for name, values in segmenter_scores.items():
        leaderboard.append(
            {
                "candidate": f"segmenter/{name}",
                "type": "segmentation",
                "score": values["score"],
                "quality_gate_passed": int(values["quality_gate_passed"]),
                "selected": int(name == segmenter_name),
            }
        )
    for name, values in classifier_scores.items():
        leaderboard.append(
            {
                "candidate": f"classifier/{name}",
                "type": "classification",
                "score": values["score"],
                "quality_gate_passed": int(values["quality_gate_passed"]),
                "selected": int(name == classifier_name),
            }
        )
    write_csv(ROOT / "metrics" / "leaderboard.csv", leaderboard)
    write_json(
        ROOT / "metrics" / "selection.json",
        {
            "deploy_ready": int(manifest["deploy_ready"]),
            "deployment_score": stable_score,
            "segmenter_score": segmenter_scores[segmenter_name]["score"],
            "segmenter_quality_gate_passed": int(segmenter_passed),
            "classifier_score": classifier_scores[classifier_name]["score"],
            "classifier_quality_gate_passed": int(classifier_passed),
        },
    )
    print(
        f"Selected {segmenter_name} + {classifier_name}; "
        f"deploy_ready={manifest['deploy_ready']}; score={stable_score:.6f}"
    )


def dagshub_coordinates(config: dict[str, Any]) -> tuple[str, str]:
    settings = config["dagshub"]
    owner = os.getenv("DAGSHUB_REPO_OWNER", str(settings["repo_owner"])).strip()
    repo = os.getenv("DAGSHUB_REPO_NAME", str(settings["repo_name"])).strip()
    if not owner or not repo:
        raise ValueError("DagsHub repo owner and name must not be empty")
    return owner, repo


def current_git_branch() -> str:
    for variable in ("GITHUB_REF_NAME", "DAGSHUB_GIT_BRANCH"):
        value = os.getenv(variable, "").strip()
        if value:
            return value
    result = subprocess.run(
        ["git", "branch", "--show-current"],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def git_ref_matches(required_branch: str) -> bool:
    if not required_branch:
        return True
    if current_git_branch() == required_branch:
        return True
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    ).stdout.strip()
    for reference in (
        f"refs/heads/{required_branch}",
        f"refs/remotes/origin/{required_branch}",
    ):
        revision = subprocess.run(
            ["git", "rev-parse", "--verify", reference],
            cwd=ROOT,
            check=False,
            capture_output=True,
            text=True,
        ).stdout.strip()
        if head and revision and head == revision:
            return True
    return False


def initialize_dagshub(config: dict[str, Any], configure_dvc: bool) -> tuple[str, str]:
    if not os.getenv("DAGSHUB_USER_TOKEN"):
        raise RuntimeError(
            "Set DAGSHUB_USER_TOKEN for non-interactive DagsHub authentication"
        )
    try:
        import dagshub
    except ImportError as error:
        raise RuntimeError("Install requirements.txt to use DagsHub") from error
    owner, repo = dagshub_coordinates(config)
    dagshub.init(
        repo_owner=owner,
        repo_name=repo,
        root=str(ROOT),
        mlflow=True,
        dvc=configure_dvc,
        patch_mlflow=False,
    )
    return owner, repo


def setup_dagshub(config: dict[str, Any]) -> None:
    owner, repo = initialize_dagshub(config, configure_dvc=True)
    print(
        f"Configured DVC and MLflow for https://dagshub.com/{owner}/{repo}. "
        "Credentials remain local."
    )


def find_bundle_run(client: Any, experiment_id: str, bundle_sha256: str) -> Any | None:
    runs = client.search_runs(
        [experiment_id],
        filter_string=f"tags.bundle_sha256 = '{bundle_sha256}'",
        order_by=["attributes.start_time DESC"],
        max_results=20,
    )
    return next((run for run in runs if run.info.status == "FINISHED"), None)


def log_bundle_run(
    mlflow: Any,
    client: Any,
    experiment_id: str,
    manifest: dict[str, Any],
    owner: str,
    repo: str,
) -> str:
    bundle_sha256 = str(manifest["bundle_sha256"])
    existing = find_bundle_run(client, experiment_id, bundle_sha256)
    if existing is not None:
        return str(existing.info.run_id)

    segmenter = manifest["segmenter"]
    classifier = manifest["classifier"]
    git_branch = current_git_branch()
    with mlflow.start_run(
        experiment_id=experiment_id,
        run_name=f"bundle-{bundle_sha256[:12]}",
    ) as active_run:
        mlflow.set_tags(
            {
                "bundle_sha256": bundle_sha256,
                "dataset_fingerprint": manifest["dataset"]["dataset_fingerprint"],
                "evaluation_fingerprint": manifest["dataset"]["evaluation_fingerprint"],
                "deploy_ready": str(bool(manifest["deploy_ready"])).lower(),
                "dagshub_repo": f"{owner}/{repo}",
                "pipeline": "leaf-segmentation-then-classification",
                "git_branch": git_branch or "detached",
            }
        )
        mlflow.log_params(
            {
                "segmenter": segmenter["candidate"],
                "segmenter_image_size": segmenter["image_size"],
                "classifier": classifier["candidate"],
                "classifier_architecture": classifier.get(
                    "architecture", classifier["candidate"]
                ),
                "classifier_image_size": classifier["image_size"],
                "classes": ",".join(manifest["classes"]),
                "schema_version": manifest["schema_version"],
            }
        )
        mlflow.log_metrics(
            {
                "deployment_score": float(manifest["deployment_score"]),
                "segmenter_map50_95": float(segmenter["metrics"]["map50_95"]),
                "segmenter_recall": float(segmenter["metrics"]["recall"]),
                "segmenter_latency_ms": float(segmenter["metrics"]["latency_ms"]),
                "classifier_macro_f1": float(classifier["metrics"]["macro_f1"]),
                "classifier_min_class_recall": float(
                    classifier["metrics"]["min_class_recall"]
                ),
                "classifier_latency_ms": float(classifier["metrics"]["latency_ms"]),
                "bundle_size_mb": (
                    float(segmenter["metrics"]["size_mb"])
                    + float(classifier["metrics"]["size_mb"])
                ),
            }
        )
        mlflow.log_artifacts(str(ROOT / "deployment"), artifact_path="bundle")
        for artifact in (
            ROOT / "params.yaml",
            ROOT / "dvc.yaml",
            ROOT / "metrics" / "data.json",
            ROOT / "metrics" / "classification_eda.json",
            ROOT / "metrics" / "segmenters.json",
            ROOT / "metrics" / "classifiers.json",
            ROOT / "metrics" / "selection.json",
        ):
            mlflow.log_artifact(str(artifact), artifact_path="reproducibility")
        for artifact in (
            ROOT / "metrics" / "classifiers.csv",
            ROOT / "metrics" / "classification_eda.png",
            ROOT / "metrics" / "classifier_training_curves.png",
            ROOT / "metrics" / "classifier_comparison.png",
            ROOT / "metrics" / "classifier_confusion_matrices.png",
        ):
            if artifact.is_file():
                mlflow.log_artifact(str(artifact), artifact_path="analysis")
        candidate_dir = ROOT / "models" / "classifiers"
        for pattern in ("*_history.csv", "*_report.json"):
            for artifact in sorted(candidate_dir.glob(pattern)):
                mlflow.log_artifact(str(artifact), artifact_path="analysis/candidates")
        return str(active_run.info.run_id)


def find_model_version(
    client: Any, model_name: str, bundle_sha256: str, run_id: str
) -> Any | None:
    versions = client.search_model_versions(f"name = '{model_name}'")
    return next(
        (
            version
            for version in versions
            if (version.tags or {}).get("bundle_sha256") == bundle_sha256
            or str(version.run_id) == run_id
        ),
        None,
    )


def production_version(client: Any, model_name: str) -> Any | None:
    from mlflow.exceptions import MlflowException

    try:
        versions = client.get_latest_versions(model_name, stages=["Production"])
    except MlflowException as error:
        if getattr(error, "error_code", "") == "RESOURCE_DOES_NOT_EXIST" or (
            "not found" in str(error).lower()
        ):
            return None
        raise
    return max(versions, key=lambda item: int(item.version), default=None)


def version_score(client: Any, version: Any) -> float | None:
    raw_score = (version.tags or {}).get("deployment_score")
    if raw_score is not None:
        return float(raw_score)
    if not version.run_id:
        return None
    metric = client.get_run(version.run_id).data.metrics.get("deployment_score")
    return None if metric is None else float(metric)


def set_registry_alias(client: Any, model_name: str, alias: str, version: Any) -> None:
    from mlflow.exceptions import MlflowException

    try:
        client.set_registered_model_alias(model_name, alias, str(version.version))
    except (AttributeError, NotImplementedError, MlflowException) as error:
        print(f"DagsHub registry alias {alias!r} is unavailable: {error}")


def publish(config: dict[str, Any]) -> None:
    settings = config["dagshub"]
    if not bool(settings.get("enabled", True)):
        write_json(
            ROOT / "metrics" / "publish.json",
            {"enabled": 0, "published": 0, "promoted": 0},
        )
        print("DagsHub publishing is disabled in params.yaml")
        return

    try:
        import mlflow
        from mlflow.tracking import MlflowClient
    except ImportError as error:
        raise RuntimeError("Install requirements.txt to publish with MLflow") from error

    owner, repo = initialize_dagshub(config, configure_dvc=False)
    manifest = json.loads(
        (ROOT / "deployment" / "manifest.json").read_text(encoding="utf-8")
    )
    experiment = mlflow.set_experiment(str(settings["experiment_name"]))
    client = MlflowClient()
    run_id = log_bundle_run(
        mlflow, client, str(experiment.experiment_id), manifest, owner, repo
    )
    candidate_score = float(manifest["deployment_score"])
    model_name = str(settings["registered_model_name"])
    champion = production_version(client, model_name)
    champion_score = None if champion is None else version_score(client, champion)
    champion_fingerprint = (
        None
        if champion is None
        else (champion.tags or {}).get("evaluation_fingerprint")
    )
    evaluation_fingerprint = str(manifest["dataset"]["evaluation_fingerprint"])
    git_branch = current_git_branch()
    required_branch = str(settings["promotion"].get("required_git_branch", "")).strip()
    branch_allowed = git_ref_matches(required_branch)
    same_evaluation = champion is None or (
        champion_fingerprint == evaluation_fingerprint
    )
    model_version = None
    promoted = False
    is_champion = False
    reason = "quality_gate_failed"

    if bool(manifest["deploy_ready"]):
        model_version = find_model_version(
            client, model_name, str(manifest["bundle_sha256"]), run_id
        )
        if model_version is None:
            model_version = mlflow.register_model(
                model_uri=f"runs:/{run_id}/bundle",
                name=model_name,
                await_registration_for=300,
            )
        version_tags = {
            "bundle_sha256": str(manifest["bundle_sha256"]),
            "dataset_fingerprint": str(manifest["dataset"]["dataset_fingerprint"]),
            "evaluation_fingerprint": evaluation_fingerprint,
            "deployment_score": f"{candidate_score:.12f}",
            "segmenter": str(manifest["segmenter"]["candidate"]),
            "classifier": str(manifest["classifier"]["candidate"]),
            "classifier_architecture": str(
                manifest["classifier"].get(
                    "architecture", manifest["classifier"]["candidate"]
                )
            ),
        }
        for key, value in version_tags.items():
            client.set_model_version_tag(
                model_name, str(model_version.version), key, value
            )
        set_registry_alias(client, model_name, "candidate", model_version)

        if champion is not None and str(champion.version) == str(model_version.version):
            reason = "already_champion"
            is_champion = True
        elif not branch_allowed:
            reason = "git_branch_not_allowed"
        elif champion is None:
            if bool(settings["promotion"]["allow_first_model"]):
                reason = "first_qualified_model"
                promoted = True
            else:
                reason = "first_model_requires_manual_approval"
        elif not same_evaluation and not bool(
            settings["promotion"].get("allow_evaluation_change", False)
        ):
            reason = "evaluation_set_changed"
        elif champion_score is None:
            reason = "champion_score_missing"
        elif candidate_score - champion_score >= float(
            settings["promotion"]["min_score_improvement"]
        ):
            reason = "score_improved"
            promoted = True
        else:
            reason = "score_not_improved"

        if promoted:
            client.transition_model_version_stage(
                name=model_name,
                version=str(model_version.version),
                stage="Production",
                archive_existing_versions=True,
            )
            set_registry_alias(client, model_name, "champion", model_version)
            is_champion = True

    client.set_tag(run_id, "last_check_status", reason)
    client.set_tag(run_id, "last_checked_git_branch", git_branch or "detached")
    client.set_tag(run_id, "registered_model", model_name)
    if promoted:
        client.set_tag(run_id, "promotion_status", reason)
        client.set_tag(run_id, "promotion_git_branch", git_branch or "detached")
    if model_version is not None:
        client.set_model_version_tag(
            model_name,
            str(model_version.version),
            "last_check_status",
            reason,
        )
        if promoted:
            client.set_model_version_tag(
                model_name, str(model_version.version), "promotion_status", reason
            )
            client.set_model_version_tag(
                model_name,
                str(model_version.version),
                "promotion_git_branch",
                git_branch or "detached",
            )

    score_improvement = (
        0.0 if champion_score is None else candidate_score - champion_score
    )
    result = {
        "enabled": 1,
        "published": 1,
        "deploy_ready": int(bool(manifest["deploy_ready"])),
        "promoted": int(promoted),
        "is_champion": int(is_champion),
        "same_evaluation": int(same_evaluation),
        "branch_allowed": int(branch_allowed),
        "model_version": 0 if model_version is None else int(model_version.version),
        "deployment_score": candidate_score,
        "champion_score_before": (-1.0 if champion_score is None else champion_score),
        "score_improvement": score_improvement,
    }
    write_json(ROOT / "metrics" / "publish.json", result)
    print(
        json.dumps(
            {
                **result,
                "promotion_status": reason,
                "git_branch": git_branch or "detached",
                "run_id": run_id,
                "registry": f"https://dagshub.com/{owner}/{repo}/models",
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def decode_multilabel_prediction(
    values: Sequence[float],
    disease_classes: Sequence[str],
    healthy_label: str,
    threshold: float,
) -> dict[str, Any]:
    if len(values) != len(disease_classes):
        raise ValueError("Classifier probability count does not match disease classes")
    disease_probabilities = {
        name: float(value) for name, value in zip(disease_classes, values)
    }
    selected = [
        name for name in disease_classes if disease_probabilities[name] >= threshold
    ]
    healthy_probability = 1.0 - max(disease_probabilities.values(), default=0.0)
    probabilities = {healthy_label: healthy_probability, **disease_probabilities}
    if selected:
        primary = max(selected, key=lambda name: disease_probabilities[name])
        confidence = disease_probabilities[primary]
        labels = selected
    else:
        primary = healthy_label
        confidence = healthy_probability
        labels = [healthy_label]
    return {
        "predicted_labels": labels,
        "primary_label": primary,
        "classification_confidence": confidence,
        "accepted": confidence >= threshold,
        "probabilities": probabilities,
    }


@dataclass
class InferenceBundle:
    """Loaded deployment bundle kept warm for repeated inference calls."""

    deployment_dir: Path
    manifest: dict[str, Any]
    detector: Any
    classifier: Any
    device: Any
    transform: Any
    disease_classes: list[str]
    healthy_label: str
    classifier_image_size: int

    @property
    def segmenter_name(self) -> str:
        return str(self.manifest["segmenter"]["candidate"])

    @property
    def classifier_name(self) -> str:
        return str(self.manifest["classifier"]["candidate"])


def load_inference_bundle(bundle_dir: Path | str = "deployment") -> InferenceBundle:
    """Load YOLO segmenter + multi-label classifier once from a deployment bundle."""
    try:
        import torch
        from torchvision import transforms
        from ultralytics import YOLO
    except ImportError as error:
        raise RuntimeError("Install requirements.txt before inference") from error

    deployment_dir = project_path(bundle_dir)
    manifest_path = deployment_dir / "manifest.json"
    segmenter_path = deployment_dir / "leaf_segmenter.pt"
    classifier_path = deployment_dir / "leaf_classifier.pt"
    for path in (manifest_path, segmenter_path, classifier_path):
        if not path.exists():
            raise FileNotFoundError(f"Missing deployment artifact: {path}")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if sha256_file(segmenter_path) != manifest["segmenter"]["sha256"]:
        raise RuntimeError("Segmenter checksum does not match deployment manifest")
    if sha256_file(classifier_path) != manifest["classifier"]["sha256"]:
        raise RuntimeError("Classifier checksum does not match deployment manifest")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(classifier_path, map_location=device, weights_only=False)
    if checkpoint.get("classification_mode") != "multilabel":
        raise RuntimeError("Classifier checkpoint is not a multi-label disease model")
    disease_classes = list(checkpoint["disease_classes"])
    healthy_label = str(checkpoint["healthy_label"])
    classifier, _ = build_classifier(
        checkpoint["model_name"],
        len(disease_classes),
        float(checkpoint["dropout"]),
        int(checkpoint["unfreeze_blocks"]),
        False,
    )
    classifier.load_state_dict(checkpoint["state_dict"])
    classifier.to(device).eval()
    image_size = int(checkpoint["image_size"])
    transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
    detector = YOLO(str(segmenter_path))
    return InferenceBundle(
        deployment_dir=deployment_dir,
        manifest=manifest,
        detector=detector,
        classifier=classifier,
        device=device,
        transform=transform,
        disease_classes=disease_classes,
        healthy_label=healthy_label,
        classifier_image_size=image_size,
    )


def crop_leaf_instance(
    image: Image.Image,
    mask: Image.Image,
    box: Sequence[float],
    manifest: dict[str, Any],
) -> tuple[Image.Image, list[int]]:
    width, height = image.size
    x1, y1, x2, y2 = [round(value) for value in box]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(width, x2), min(height, y2)
    padding = float(manifest["preprocessing"]["crop_padding_ratio"])
    pad_x, pad_y = round((x2 - x1) * padding), round((y2 - y1) * padding)
    crop_box = (
        max(0, x1 - pad_x),
        max(0, y1 - pad_y),
        min(width, x2 + pad_x),
        min(height, y2 + pad_y),
    )
    crop = image.crop(crop_box)
    if manifest["preprocessing"]["mask_background"]:
        crop_mask = mask.crop(crop_box)
        background = Image.new(
            "RGB", crop.size, tuple(manifest["preprocessing"]["mask_fill_rgb"])
        )
        crop = Image.composite(crop, background, crop_mask)
    return crop, [x1, y1, x2, y2]


def classify_leaf_crops(
    bundle: InferenceBundle,
    crop_tensors: Sequence[Any],
) -> list[dict[str, Any]]:
    import torch

    if not crop_tensors:
        return []
    batch = torch.stack(list(crop_tensors)).to(bundle.device)
    with torch.inference_mode():
        probabilities = bundle.classifier(batch).sigmoid().cpu()
    threshold = float(bundle.manifest["thresholds"]["classifier_confidence"])
    decoded: list[dict[str, Any]] = []
    for probability in probabilities:
        decoded.append(
            decode_multilabel_prediction(
                probability.tolist(),
                bundle.disease_classes,
                bundle.healthy_label,
                threshold,
            )
        )
    return decoded


def build_prediction_summary(
    leaves: Sequence[dict[str, Any]],
    healthy_label: str,
) -> dict[str, Any]:
    accepted = [leaf for leaf in leaves if leaf.get("accepted")]
    counts: Counter[str] = Counter()
    for leaf in accepted:
        counts.update(leaf["predicted_labels"])
    diseased = sum(
        any(name != healthy_label for name in leaf["predicted_labels"])
        for leaf in accepted
    )
    return {
        "detected_leaves": len(leaves),
        "accepted_classifications": len(accepted),
        "class_counts": dict(counts),
        "diseased_leaf_fraction": diseased / len(accepted) if accepted else None,
    }


def render_annotated_prediction(
    image: Image.Image,
    leaves: Sequence[dict[str, Any]],
    masks: Sequence[Image.Image],
    healthy_label: str,
    disease_classes: Sequence[str],
) -> Image.Image:
    """Draw instance masks, boxes and multi-label text onto a copy of the image."""
    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    overlay_draw = ImageDraw.Draw(overlay)
    colors = [
        (64, 180, 75, 75),
        (255, 179, 0, 75),
        (225, 87, 89, 75),
        (116, 78, 170, 75),
    ]
    output_classes = [healthy_label, *disease_classes]
    for leaf, mask in zip(leaves, masks):
        class_name = leaf.get("primary_label", "unknown")
        class_index = (
            output_classes.index(class_name) if class_name in output_classes else 0
        )
        color = colors[class_index % len(colors)]
        color_layer = Image.new("RGBA", image.size, color)
        overlay.alpha_composite(
            Image.composite(color_layer, Image.new("RGBA", image.size), mask)
        )
        box = leaf["bbox_xyxy"]
        label_names = "+".join(leaf.get("predicted_labels", [class_name]))
        label = f"{label_names} {leaf.get('classification_confidence', 0.0):.2f}"
        overlay_draw.rectangle(box, outline=color[:3] + (255,), width=3)
        overlay_draw.text(
            (box[0] + 3, max(0, box[1] - 13)), label, fill=color[:3] + (255,)
        )
    return Image.alpha_composite(image.convert("RGBA"), overlay).convert("RGB")


def run_inference(
    image: Image.Image,
    bundle: InferenceBundle,
    *,
    allow_single_leaf_fallback: bool = True,
) -> dict[str, Any]:
    """Run segmentation + multi-label classification on an in-memory RGB image.

    When the segmenter finds no leaves and ``allow_single_leaf_fallback`` is true,
    the whole image is treated as a single leaf crop (close-up photos).
    """
    import time

    rgb = image.convert("RGB")
    width, height = rgb.size
    timing: dict[str, float] = {
        "preprocess_ms": 0.0,
        "segmentation_ms": 0.0,
        "classification_ms": 0.0,
    }

    started = time.perf_counter()
    detection = bundle.detector.predict(
        source=np.asarray(rgb),
        conf=float(bundle.manifest["thresholds"]["detector_confidence"]),
        imgsz=int(bundle.manifest["segmenter"]["image_size"]),
        verbose=False,
    )
    timing["segmentation_ms"] = (time.perf_counter() - started) * 1000.0
    if len(detection) != 1:
        raise ValueError("Inference currently accepts exactly one image")
    result = detection[0]

    leaves: list[dict[str, Any]] = []
    crop_images: list[Image.Image] = []
    crop_tensors: list[Any] = []
    masks: list[Image.Image] = []
    boxes = [] if result.boxes is None else result.boxes.xyxy.cpu().tolist()
    detector_confidences = (
        [] if result.boxes is None else result.boxes.conf.cpu().tolist()
    )
    if result.masks is not None:
        for mask_tensor in result.masks.data.cpu():
            array = (mask_tensor.numpy() >= 0.5).astype(np.uint8) * 255
            mask = Image.fromarray(array, mode="L").resize(
                (width, height), Image.Resampling.NEAREST
            )
            masks.append(mask)

    mode = "tree"
    if masks and boxes:
        preprocess_started = time.perf_counter()
        for index, (mask, box) in enumerate(zip(masks, boxes)):
            crop, bbox = crop_leaf_instance(rgb, mask, box, bundle.manifest)
            crop_images.append(crop)
            crop_tensors.append(bundle.transform(crop))
            leaves.append(
                {
                    "leaf_id": index,
                    "bbox_xyxy": bbox,
                    "detector_confidence": float(detector_confidences[index]),
                }
            )
        timing["preprocess_ms"] = (time.perf_counter() - preprocess_started) * 1000.0
    elif allow_single_leaf_fallback:
        mode = "single_leaf"
        preprocess_started = time.perf_counter()
        full_mask = Image.new("L", (width, height), 255)
        masks = [full_mask]
        crop_images = [rgb]
        crop_tensors = [bundle.transform(rgb)]
        leaves = [
            {
                "leaf_id": 0,
                "bbox_xyxy": [0, 0, width, height],
                "detector_confidence": 1.0,
            }
        ]
        timing["preprocess_ms"] = (time.perf_counter() - preprocess_started) * 1000.0
        timing["segmentation_ms"] = 0.0

    classify_started = time.perf_counter()
    decoded = classify_leaf_crops(bundle, crop_tensors)
    timing["classification_ms"] = (time.perf_counter() - classify_started) * 1000.0
    for leaf, prediction in zip(leaves, decoded):
        leaf.update(prediction)

    summary = build_prediction_summary(leaves, bundle.healthy_label)
    return {
        "mode": mode,
        "image_size": {"width": width, "height": height},
        "deploy_ready": bool(bundle.manifest["deploy_ready"]),
        "timing": timing,
        "summary": summary,
        "leaves": leaves,
        "masks": masks,
        "crops": crop_images,
        "image": rgb,
        "disease_classes": list(bundle.disease_classes),
        "healthy_label": bundle.healthy_label,
        "segmenter": bundle.segmenter_name,
        "classifier": bundle.classifier_name,
        "manifest": bundle.manifest,
    }


def predict(source: Path, output_dir: Path, bundle_dir: Path) -> None:
    """CLI-compatible two-stage inference that writes result.json + annotated.jpg."""
    bundle = load_inference_bundle(bundle_dir)
    with Image.open(source) as opened:
        image = opened.convert("RGB")
    inference = run_inference(image, bundle, allow_single_leaf_fallback=True)
    payload = {
        "source": str(source),
        "deploy_ready": inference["deploy_ready"],
        "mode": inference["mode"],
        "summary": inference["summary"],
        "leaves": [
            {
                key: value
                for key, value in leaf.items()
                if key
                in {
                    "leaf_id",
                    "bbox_xyxy",
                    "detector_confidence",
                    "predicted_labels",
                    "primary_label",
                    "classification_confidence",
                    "accepted",
                    "probabilities",
                }
            }
            for leaf in inference["leaves"]
        ],
    }
    output_dir = reset_dir(output_dir)
    write_json(output_dir / "result.json", payload)
    annotated = render_annotated_prediction(
        inference["image"],
        inference["leaves"],
        inference["masks"],
        inference["healthy_label"],
        inference["disease_classes"],
    )
    annotated.save(output_dir / "annotated.jpg", quality=95)
    print(json.dumps(payload["summary"], ensure_ascii=False, indent=2))


def doctor(config: dict[str, Any], check_data: bool) -> None:
    errors: list[str] = []
    data_settings = config.get("data", {})
    disease_classes = data_settings.get("disease_classes", [])
    healthy_label = str(data_settings.get("healthy_label", "")).strip()
    splits = data_settings.get("splits", [])
    if len(disease_classes) < 1 or len(disease_classes) != len(set(disease_classes)):
        errors.append("data.disease_classes must contain unique disease names")
    if not healthy_label or healthy_label in disease_classes:
        errors.append("data.healthy_label must be set and not overlap disease_classes")
    if list(splits) != ["train", "val", "test"]:
        errors.append("data.splits must be [train, val, test]")
    for source in ("segmentation", "classification"):
        source_settings = data_settings.get(source, {})
        if not str(source_settings.get("raw_dir", "")).strip():
            errors.append(f"data.{source}.raw_dir must not be empty")
    try:
        resolve_split_ratios(data_settings.get("segmentation", {}), splits)
    except (TypeError, ValueError) as error:
        errors.append(f"data.segmentation.{error}")
    try:
        validation_fraction = float(
            data_settings.get("classification", {}).get("validation_fraction", 0.15)
        )
        if not 0.0 < validation_fraction < 0.5:
            raise ValueError("validation_fraction must be between 0 and 0.5")
    except (TypeError, ValueError) as error:
        errors.append(f"data.classification.{error}")
    try:
        fill_rgb = [int(value) for value in data_settings.get("mask_fill_rgb", [])]
        if len(fill_rgb) != 3 or any(value < 0 or value > 255 for value in fill_rgb):
            raise ValueError("data.mask_fill_rgb must contain three values from 0 to 255")
    except (TypeError, ValueError) as error:
        errors.append(str(error))
    classification_settings = config.get("classification", {})
    augmentation = classification_settings.get("augmentation", {})
    try:
        crop_scale = [float(value) for value in augmentation.get("crop_scale", [])]
        if (
            len(crop_scale) != 2
            or not 0.0 < crop_scale[0] <= crop_scale[1] <= 1.0
        ):
            raise ValueError(
                "classification.augmentation.crop_scale must be two ordered "
                "values in (0, 1]"
            )
        crop_ratio = [float(value) for value in augmentation.get("crop_ratio", [])]
        if len(crop_ratio) != 2 or not 0.0 < crop_ratio[0] <= crop_ratio[1]:
            raise ValueError(
                "classification.augmentation.crop_ratio must be two positive "
                "ordered values"
            )
        for name in (
            "gaussian_blur_probability",
            "random_erasing_probability",
        ):
            probability = float(augmentation.get(name, -1.0))
            if not 0.0 <= probability <= 1.0:
                raise ValueError(
                    f"classification.augmentation.{name} must be between 0 and 1"
                )
        rotation = float(augmentation.get("rotation_degrees", -1.0))
        jitter = float(augmentation.get("color_jitter", -1.0))
        if not 0.0 <= rotation <= 180.0:
            raise ValueError(
                "classification.augmentation.rotation_degrees must be between 0 and 180"
            )
        if not 0.0 <= jitter <= 1.0:
            raise ValueError(
                "classification.augmentation.color_jitter must be between 0 and 1"
            )
    except (TypeError, ValueError) as error:
        errors.append(str(error))
    try:
        classifier_threshold = float(
            config.get("deployment", {}).get("classifier_confidence", -1.0)
        )
        if not 0.0 < classifier_threshold < 1.0:
            raise ValueError("deployment.classifier_confidence must be between 0 and 1")
    except (TypeError, ValueError) as error:
        errors.append(str(error))
    for kind in ("segmentation", "classification"):
        weights = config.get("selection", {}).get(kind, {}).get("weights", {})
        if not weights or not math.isclose(
            sum(float(value) for value in weights.values()), 1.0, abs_tol=1e-6
        ):
            errors.append(f"selection.{kind}.weights must sum to 1.0")
    deployment_weights = config.get("selection", {}).get("deployment_weights", {})
    expected_deployment_weights = {
        "segmenter_map50_95",
        "segmenter_recall",
        "classifier_macro_f1",
        "classifier_min_class_recall",
    }
    if set(deployment_weights) != expected_deployment_weights or not math.isclose(
        sum(float(value) for value in deployment_weights.values()),
        1.0,
        abs_tol=1e-6,
    ):
        errors.append(
            "selection.deployment_weights must contain the four documented metrics "
            "and sum to 1.0"
        )
    dagshub_settings = config.get("dagshub", {})
    for key in ("repo_owner", "repo_name", "experiment_name", "registered_model_name"):
        if not str(dagshub_settings.get(key, "")).strip():
            errors.append(f"dagshub.{key} must not be empty")
    if (
        float(dagshub_settings.get("promotion", {}).get("min_score_improvement", -1.0))
        < 0.0
    ):
        errors.append("dagshub.promotion.min_score_improvement must be non-negative")
    for candidate, settings in classification_settings.get("candidates", {}).items():
        try:
            architecture = str(settings.get("architecture", candidate))
            if architecture not in CLASSIFIER_ARCHITECTURES:
                raise ValueError(
                    f"uses unsupported architecture {architecture!r}"
                )
            if int(settings.get("image_size", 0)) <= 0:
                raise ValueError("image_size must be positive")
            if not 0.0 <= float(settings.get("dropout", -1.0)) < 1.0:
                raise ValueError("dropout must be in [0, 1)")
            if int(settings.get("unfreeze_blocks", -1)) < 0:
                raise ValueError("unfreeze_blocks must be non-negative")
            batch_size = int(
                settings.get("batch_size", classification_settings.get("batch_size", 0))
            )
            if batch_size <= 0:
                raise ValueError("batch_size must be positive")
        except (TypeError, ValueError) as error:
            errors.append(f"classification candidate {candidate!r} {error}")
    data_summary: dict[str, Any] = {}
    if check_data and not errors:
        try:
            segmentation_records, used_manifest = load_segmentation_records(
                project_path(data_settings["segmentation"]["raw_dir"]),
                data_settings["segmentation"],
                splits,
                int(config["seed"]),
            )
            classification_records, mask_background_rgb = load_classification_records(
                project_path(data_settings["classification"]["raw_dir"]),
                data_settings["classification"],
                disease_classes,
                splits,
                int(config["seed"]),
            )
            positive_counts = {
                name: sum(record.targets[index] for record in classification_records)
                for index, name in enumerate(disease_classes)
            }
            data_summary = {
                "segmentation_images": len(segmentation_records),
                "segmentation_instances": sum(
                    len(record.polygons) for record in segmentation_records
                ),
                "segmentation_manifest": used_manifest,
                "classification_images": len(classification_records),
                "classification_positive_labels": positive_counts,
                "classification_healthy": sum(
                    not any(record.targets) for record in classification_records
                ),
                "classification_multi_disease": sum(
                    sum(record.targets) > 1 for record in classification_records
                ),
                "classification_mask_background_rgb": list(mask_background_rgb),
            }
        except (KeyError, OSError, TypeError, ValueError) as error:
            errors.append(str(error))

    packages = {}
    for name in (
        "dagshub",
        "dvc",
        "papermill",
        "ipykernel",
        "jupyterlab",
        "mlflow",
        "matplotlib",
        "seaborn",
        "torch",
        "torchvision",
        "ultralytics",
        "sklearn",
    ):
        try:
            module = importlib.import_module(name)
            packages[name] = getattr(module, "__version__", "installed")
        except ImportError:
            packages[name] = "missing"
    print(
        json.dumps(
            {
                "configuration": "ok" if not errors else "invalid",
                "dagshub_token": (
                    "configured" if os.getenv("DAGSHUB_USER_TOKEN") else "missing"
                ),
                "data": data_summary if check_data else "not checked",
                "packages": packages,
            },
            indent=2,
        )
    )
    if errors:
        raise ValueError("; ".join(errors))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--params", default="params.yaml", help="Path to pipeline parameters"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser(
        "prepare", help="Validate annotations and build both processed datasets"
    )
    subparsers.add_parser(
        "select", help="Select candidates and create a deployment bundle"
    )
    subparsers.add_parser(
        "setup-dagshub", help="Configure the DagsHub MLflow and DVC integrations"
    )
    subparsers.add_parser(
        "publish", help="Log the bundle and promote it in the DagsHub Model Registry"
    )
    predict_parser = subparsers.add_parser(
        "predict", help="Run the selected two-stage pipeline"
    )
    predict_parser.add_argument("--source", required=True, type=Path)
    predict_parser.add_argument("--output", default=Path("runs/predict"), type=Path)
    predict_parser.add_argument(
        "--bundle", default=Path("deployment"), type=Path, help="Model bundle path"
    )
    doctor_parser = subparsers.add_parser(
        "doctor", help="Validate parameters and report dependencies"
    )
    doctor_parser.add_argument("--check-data", action="store_true")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    config = load_config(args.params)
    if args.command == "prepare":
        prepare(config)
    elif args.command == "select":
        select_models(config)
    elif args.command == "setup-dagshub":
        setup_dagshub(config)
    elif args.command == "publish":
        publish(config)
    elif args.command == "predict":
        predict(args.source, project_path(args.output), args.bundle)
    elif args.command == "doctor":
        doctor(config, args.check_data)
    else:
        parser.error(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
