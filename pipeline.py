"""Reproducible two-stage coffee-leaf disease pipeline.

The same YOLO-seg annotation is used twice: class ids describe disease states in
raw data, while ``prepare`` produces a class-agnostic leaf segmentation dataset
and masked leaf crops for classification.  Heavy ML dependencies are imported
inside training/inference commands so data preparation and configuration checks
remain lightweight.
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
import tempfile
import time
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
CLASSIFIER_ARCHITECTURES = {"efficientnet_v2_s", "regnet_y_3_2gf"}
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


@dataclass(frozen=True)
class Record:
    image: Path
    label: Path
    split: str
    group_id: str


@dataclass(frozen=True)
class Polygon:
    class_id: int
    points: tuple[tuple[float, float], ...]


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


def dataset_fingerprint(records: Sequence[Record], raw_dir: Path) -> str:
    """Hash dataset identity and bytes, independent of filesystem ordering."""
    digest = hashlib.sha256()
    ordered = sorted(
        records,
        key=lambda item: (
            item.split,
            item.group_id,
            item.image.relative_to(raw_dir).as_posix(),
        ),
    )
    for record in ordered:
        identity = {
            "group_id": record.group_id,
            "image": record.image.relative_to(raw_dir).as_posix(),
            "label": record.label.relative_to(raw_dir).as_posix(),
            "split": record.split,
        }
        digest.update(
            json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
        )
        digest.update(bytes.fromhex(sha256_file(record.image)))
        digest.update(bytes.fromhex(sha256_file(record.label)))
    return digest.hexdigest()


def record_asset_id(record: Record, raw_dir: Path) -> str:
    """Keep output names readable while preventing cross-folder collisions."""
    relative = record.image.relative_to(raw_dir).as_posix()
    suffix = hashlib.sha256(relative.encode()).hexdigest()[:12]
    return f"{record.image.stem}__{suffix}"


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


def load_records(raw_dir: Path, splits: Sequence[str]) -> tuple[list[Record], bool]:
    manifest = raw_dir / "manifest.csv"
    records: list[Record] = []
    used_manifest = manifest.exists()

    if used_manifest:
        with manifest.open("r", encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream)
            required = {"image", "label", "split", "group_id"}
            missing = required.difference(reader.fieldnames or [])
            if missing:
                raise ValueError(f"manifest.csv is missing columns: {sorted(missing)}")
            for row_number, row in enumerate(reader, start=2):
                split = row["split"].strip().lower()
                if split not in splits:
                    raise ValueError(
                        f"manifest.csv:{row_number}: invalid split {split!r}"
                    )
                group_id = row["group_id"].strip()
                if not group_id:
                    raise ValueError(f"manifest.csv:{row_number}: group_id is empty")
                image = inside(raw_dir / row["image"].strip(), raw_dir)
                label = inside(raw_dir / row["label"].strip(), raw_dir)
                records.append(Record(image, label, split, group_id))
    else:
        for split in splits:
            image_dir = raw_dir / "images" / split
            label_dir = raw_dir / "labels" / split
            if not image_dir.exists():
                continue
            for image in sorted(image_dir.iterdir()):
                if image.is_file() and image.suffix.lower() in IMAGE_EXTENSIONS:
                    label = label_dir / f"{image.stem}.txt"
                    records.append(
                        Record(image.resolve(), label.resolve(), split, image.stem)
                    )

    if not records:
        raise ValueError(
            "No annotated images found. Add data/raw/manifest.csv or use "
            "data/raw/images/{train,val,test} with matching labels directories."
        )

    group_splits: dict[str, set[str]] = {}
    for record in records:
        if not record.image.exists():
            raise FileNotFoundError(record.image)
        if not record.label.exists():
            raise FileNotFoundError(record.label)
        group_splits.setdefault(record.group_id, set()).add(record.split)
    leaking = {
        group: values for group, values in group_splits.items() if len(values) > 1
    }
    if leaking:
        preview = dict(list(leaking.items())[:5])
        raise ValueError(f"Group leakage across splits detected: {preview}")
    return records, used_manifest


def polygon_area(points: Sequence[tuple[float, float]]) -> float:
    total = 0.0
    for index, (x1, y1) in enumerate(points):
        x2, y2 = points[(index + 1) % len(points)]
        total += x1 * y2 - x2 * y1
    return abs(total) / 2.0


def read_polygons(path: Path, number_of_classes: int) -> list[Polygon]:
    polygons: list[Polygon] = []
    for line_number, raw_line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        line = raw_line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) < 7 or (len(parts) - 1) % 2:
            raise ValueError(
                f"{path}:{line_number}: YOLO polygon needs at least 3 points"
            )
        try:
            class_value = float(parts[0])
            class_id = int(class_value)
            coordinates = [float(item) for item in parts[1:]]
        except ValueError as error:
            raise ValueError(f"{path}:{line_number}: non-numeric annotation") from error
        if class_value != class_id or not 0 <= class_id < number_of_classes:
            raise ValueError(f"{path}:{line_number}: invalid class id {parts[0]!r}")
        if any(
            not math.isfinite(value) or not 0.0 <= value <= 1.0 for value in coordinates
        ):
            raise ValueError(
                f"{path}:{line_number}: coordinates must be normalized to [0, 1]"
            )
        points = tuple(zip(coordinates[0::2], coordinates[1::2]))
        if polygon_area(points) <= 1e-8:
            raise ValueError(f"{path}:{line_number}: polygon area is zero")
        polygons.append(Polygon(class_id, points))
    if not polygons:
        raise ValueError(f"Annotation contains no leaf polygons: {path}")
    return polygons


def pixel_polygon(polygon: Polygon, width: int, height: int) -> list[tuple[int, int]]:
    return [
        (
            min(width - 1, max(0, round(x * width))),
            min(height - 1, max(0, round(y * height))),
        )
        for x, y in polygon.points
    ]


def masked_crop(
    image: Image.Image,
    polygon: Polygon,
    padding_ratio: float,
    mask_background: bool,
    fill_rgb: tuple[int, int, int],
) -> Image.Image:
    width, height = image.size
    points = pixel_polygon(polygon, width, height)
    xs, ys = zip(*points)
    x1, x2, y1, y2 = min(xs), max(xs), min(ys), max(ys)
    pad_x = round((x2 - x1 + 1) * padding_ratio)
    pad_y = round((y2 - y1 + 1) * padding_ratio)
    box = (
        max(0, x1 - pad_x),
        max(0, y1 - pad_y),
        min(width, x2 + pad_x + 1),
        min(height, y2 + pad_y + 1),
    )
    crop = image.crop(box)
    if not mask_background:
        return crop
    mask = Image.new("L", image.size, 0)
    ImageDraw.Draw(mask).polygon(points, fill=255)
    crop_mask = mask.crop(box)
    background = Image.new("RGB", crop.size, fill_rgb)
    return Image.composite(crop, background, crop_mask)


def prepare(config: dict[str, Any]) -> None:
    data = config["data"]
    raw_dir = project_path(data["raw_dir"])
    output_dir = reset_dir(project_path(data["processed_dir"]))
    classes = list(data["classes"])
    splits = list(data["splits"])
    records, used_manifest = load_records(raw_dir, splits)
    if bool(data.get("require_group_manifest", False)) and not used_manifest:
        raise ValueError(
            "data.require_group_manifest=true but data/raw/manifest.csv is missing. "
            "Use a group-aware manifest so the same tree/session cannot leak across splits."
        )
    seed_everything(int(config["seed"]))
    full_fingerprint = dataset_fingerprint(records, raw_dir)
    evaluation_records = [record for record in records if record.split == "val"]
    evaluation_data_fingerprint = dataset_fingerprint(evaluation_records, raw_dir)
    evaluation_protocol = {
        "classes": classes,
        "crop_padding_ratio": float(data["crop_padding_ratio"]),
        "mask_background": bool(data["mask_background"]),
        "mask_fill_rgb": [int(value) for value in data["mask_fill_rgb"]],
        "min_crop_size": int(data["min_crop_size"]),
    }
    evaluation_fingerprint = hashlib.sha256(
        (
            evaluation_data_fingerprint
            + json.dumps(evaluation_protocol, sort_keys=True, separators=(",", ":"))
        ).encode()
    ).hexdigest()

    segment_root = output_dir / "segmentation"
    classifier_root = output_dir / "classification"
    class_counts: Counter[tuple[str, str]] = Counter()
    image_counts: Counter[str] = Counter()
    skipped_small = 0

    for record in records:
        polygons = read_polygons(record.label, len(classes))
        with Image.open(record.image) as source:
            image = source.convert("RGB")
        asset_id = record_asset_id(record, raw_dir)
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
                for polygon in polygons
            )
            + "\n",
            encoding="utf-8",
        )
        image_counts[record.split] += 1

        for leaf_number, polygon in enumerate(polygons):
            crop = masked_crop(
                image,
                polygon,
                float(data["crop_padding_ratio"]),
                bool(data["mask_background"]),
                tuple(int(value) for value in data["mask_fill_rgb"]),
            )
            if min(crop.size) < int(data["min_crop_size"]):
                skipped_small += 1
                continue
            class_name = classes[polygon.class_id]
            crop_target = (
                classifier_root
                / record.split
                / class_name
                / f"{asset_id}__leaf_{leaf_number:04d}.jpg"
            )
            crop_target.parent.mkdir(parents=True, exist_ok=True)
            crop.save(crop_target, format="JPEG", quality=95, subsampling=0)
            class_counts[(record.split, class_name)] += 1

    dataset_yaml = {
        "path": ".",
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": {0: "leaf"},
    }
    with (segment_root / "dataset.yaml").open("w", encoding="utf-8") as stream:
        yaml.safe_dump(dataset_yaml, stream, sort_keys=False, allow_unicode=True)

    missing_splits = [split for split in splits if image_counts[split] == 0]
    missing_classes = [
        f"{split}/{class_name}"
        for split in splits
        for class_name in classes
        if class_counts[(split, class_name)] == 0
    ]
    if missing_splits or missing_classes:
        raise ValueError(
            f"Incomplete prepared dataset; missing splits={missing_splits}, "
            f"missing class folders={missing_classes}"
        )

    metrics: dict[str, Any] = {
        "images_total": sum(image_counts.values()),
        "leaf_crops_total": sum(class_counts.values()),
        "skipped_small_crops": skipped_small,
        "used_group_manifest": int(used_manifest),
    }
    for split in splits:
        metrics[f"images_{split}"] = image_counts[split]
        metrics[f"leaf_crops_{split}"] = sum(
            class_counts[(split, name)] for name in classes
        )
        for name in classes:
            metrics[f"leaf_crops_{split}_{name}"] = class_counts[(split, name)]
    write_json(
        output_dir / "dataset_manifest.json",
        {
            "classes": classes,
            "dataset_fingerprint": full_fingerprint,
            "evaluation_data_fingerprint": evaluation_data_fingerprint,
            "evaluation_fingerprint": evaluation_fingerprint,
            "evaluation_protocol": evaluation_protocol,
            "records": len(records),
            "splits": {split: image_counts[split] for split in splits},
            "used_group_manifest": used_manifest,
        },
    )
    write_json(ROOT / "metrics" / "data.json", metrics)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


def nested_attr(obj: Any, path: str, default: float = 0.0) -> float:
    current = obj
    for name in path.split("."):
        if current is None:
            return default
        current = getattr(current, name, None)
    try:
        return float(current)
    except (TypeError, ValueError):
        return default


def runtime_yolo_yaml(segment_root: Path) -> Path:
    payload = {
        "path": str(segment_root.resolve()),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": {0: "leaf"},
    }
    with tempfile.NamedTemporaryFile(
        "w", suffix=".yaml", encoding="utf-8", delete=False
    ) as handle:
        yaml.safe_dump(payload, handle, sort_keys=False)
        path = Path(handle.name)
    return path


def resolve_device(value: Any) -> Any:
    if str(value).lower() == "auto":
        return None
    return value


def train_segmenters(config: dict[str, Any]) -> None:
    try:
        import torch
        from ultralytics import YOLO
    except ImportError as error:
        raise RuntimeError(
            "Install requirements.txt before training segmenters"
        ) from error

    seed = int(config["seed"])
    seed_everything(seed)
    settings = config["segmentation"]
    segment_root = project_path(config["data"]["processed_dir"]) / "segmentation"
    output_dir = reset_dir(ROOT / "models" / "segmenters")
    data_yaml = runtime_yolo_yaml(segment_root)
    metrics_by_candidate: dict[str, dict[str, Any]] = {}
    rows: list[dict[str, Any]] = []

    try:
        for candidate, pretrained_weights in settings["candidates"].items():
            print(f"\n=== Training segmenter: {candidate} ===")
            source = str(pretrained_weights)
            if not bool(settings.get("pretrained", True)) and source.endswith(".pt"):
                source = source[:-3] + ".yaml"
            model = YOLO(source)
            with tempfile.TemporaryDirectory(prefix=f"coffee-{candidate}-") as run_dir:
                kwargs: dict[str, Any] = {
                    "data": str(data_yaml),
                    "epochs": int(settings["epochs"]),
                    "imgsz": int(settings["image_size"]),
                    "batch": int(settings["batch_size"]),
                    "patience": int(settings["patience"]),
                    "workers": int(settings["workers"]),
                    "seed": seed,
                    "deterministic": bool(settings["deterministic"]),
                    "project": run_dir,
                    "name": candidate,
                    "exist_ok": True,
                    "verbose": True,
                }
                device = resolve_device(settings.get("device", "auto"))
                if device is not None:
                    kwargs["device"] = device
                model.train(**kwargs)
                best_path = Path(model.trainer.best)
                if not best_path.exists():
                    raise RuntimeError(
                        f"Ultralytics did not create best weights for {candidate}"
                    )
                target = output_dir / f"{candidate}.pt"
                shutil.copy2(best_path, target)

            best_model = YOLO(str(target))
            validation = best_model.val(
                data=str(data_yaml),
                split="val",
                imgsz=int(settings["image_size"]),
                batch=int(settings["batch_size"]),
                workers=int(settings["workers"]),
                verbose=False,
            )
            test_result = best_model.val(
                data=str(data_yaml),
                split="test",
                imgsz=int(settings["image_size"]),
                batch=int(settings["batch_size"]),
                workers=int(settings["workers"]),
                verbose=False,
            )
            speed = getattr(validation, "speed", {}) or {}
            values = {
                "map50_95": nested_attr(validation, "seg.map"),
                "map50": nested_attr(validation, "seg.map50"),
                "precision": nested_attr(validation, "seg.mp"),
                "recall": nested_attr(validation, "seg.mr"),
                "test_map50_95": nested_attr(test_result, "seg.map"),
                "test_map50": nested_attr(test_result, "seg.map50"),
                "test_precision": nested_attr(test_result, "seg.mp"),
                "test_recall": nested_attr(test_result, "seg.mr"),
                "latency_ms": float(speed.get("inference", 0.0)),
                "size_mb": target.stat().st_size / (1024 * 1024),
                "image_size": int(settings["image_size"]),
            }
            metrics_by_candidate[candidate] = values
            rows.append(
                {
                    "candidate": candidate,
                    **{
                        key: values[key]
                        for key in ("map50_95", "recall", "latency_ms", "size_mb")
                    },
                }
            )
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    finally:
        data_yaml.unlink(missing_ok=True)

    write_json(
        ROOT / "metrics" / "segmenters.json", {"candidates": metrics_by_candidate}
    )
    write_csv(ROOT / "metrics" / "segmenters.csv", rows)


def build_classifier(
    name: str,
    number_of_classes: int,
    dropout: float,
    unfreeze_blocks: int,
    pretrained: bool,
) -> tuple[Any, Any]:
    from torch import nn
    from torchvision.models import (
        EfficientNet_V2_S_Weights,
        RegNet_Y_3_2GF_Weights,
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
    else:
        raise ValueError(f"Unsupported classifier candidate: {name}")

    for parameter in head.parameters():
        parameter.requires_grad = True
    return model, head


class LeafCropDataset:
    """Small ImageFolder equivalent with a stable class mapping across splits."""

    def __init__(self, root: Path, classes: Sequence[str], transform: Any) -> None:
        self.transform = transform
        self.samples: list[tuple[Path, int]] = []
        for class_id, class_name in enumerate(classes):
            class_dir = root / class_name
            if not class_dir.exists():
                raise ValueError(f"Missing class directory: {class_dir}")
            images = [
                path
                for path in sorted(class_dir.iterdir())
                if path.suffix.lower() in IMAGE_EXTENSIONS
            ]
            if not images:
                raise ValueError(f"Class directory is empty: {class_dir}")
            self.samples.extend((path, class_id) for path in images)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[Any, int]:
        path, target = self.samples[index]
        with Image.open(path) as source:
            image = source.convert("RGB")
        return self.transform(image), target


def classifier_transforms(image_size: int) -> tuple[Any, Any]:
    from torchvision import transforms

    train_transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(0.5),
            transforms.RandomVerticalFlip(0.5),
            transforms.RandomRotation(20),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
    evaluation_transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
    return train_transform, evaluation_transform


def evaluate_classifier(
    model: Any, loader: Any, device: Any, criterion: Any = None
) -> dict[str, Any]:
    import torch
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support

    model.eval()
    targets: list[int] = []
    predictions: list[int] = []
    total_loss = 0.0
    with torch.inference_mode():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            logits = model(inputs)
            if criterion is not None:
                total_loss += float(criterion(logits, labels).item()) * labels.size(0)
            predictions.extend(logits.argmax(1).cpu().tolist())
            targets.extend(labels.cpu().tolist())
    labels_index = (
        list(range(len(loader.dataset.classes)))
        if hasattr(loader.dataset, "classes")
        else sorted(set(targets))
    )
    precision, recall, f1, support = precision_recall_fscore_support(
        targets, predictions, labels=labels_index, zero_division=0
    )
    return {
        "loss": total_loss / max(1, len(targets)),
        "accuracy": float(accuracy_score(targets, predictions)),
        "precision": precision.tolist(),
        "recall": recall.tolist(),
        "f1": f1.tolist(),
        "support": support.tolist(),
        "macro_f1": float(np.mean(f1)),
        "targets": targets,
        "predictions": predictions,
    }


def train_classifiers(config: dict[str, Any]) -> None:
    try:
        import torch
        from sklearn.metrics import confusion_matrix
        from torch import nn
        from torch.utils.data import DataLoader
    except ImportError as error:
        raise RuntimeError(
            "Install requirements.txt before training classifiers"
        ) from error

    seed = int(config["seed"])
    seed_everything(seed)
    settings = config["classification"]
    classes = list(config["data"]["classes"])
    data_root = project_path(config["data"]["processed_dir"]) / "classification"
    output_dir = reset_dir(ROOT / "models" / "classifiers")
    metrics_by_candidate: dict[str, dict[str, Any]] = {}
    comparison_rows: list[dict[str, Any]] = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for candidate, candidate_settings in settings["candidates"].items():
        print(f"\n=== Training classifier: {candidate} ===")
        architecture = str(candidate_settings.get("architecture", candidate))
        image_size = int(candidate_settings["image_size"])
        train_transform, eval_transform = classifier_transforms(image_size)
        train_dataset = LeafCropDataset(data_root / "train", classes, train_transform)
        val_dataset = LeafCropDataset(data_root / "val", classes, eval_transform)
        test_dataset = LeafCropDataset(data_root / "test", classes, eval_transform)
        # Used by evaluate_classifier without relying on ImageFolder internals.
        train_dataset.classes = classes
        val_dataset.classes = classes
        test_dataset.classes = classes
        generator = torch.Generator().manual_seed(seed)
        loader_args = {
            "batch_size": int(settings["batch_size"]),
            "num_workers": int(settings["workers"]),
            "pin_memory": device.type == "cuda",
        }
        train_loader = DataLoader(
            train_dataset, shuffle=True, generator=generator, **loader_args
        )
        val_loader = DataLoader(val_dataset, shuffle=False, **loader_args)
        test_loader = DataLoader(test_dataset, shuffle=False, **loader_args)

        model, head = build_classifier(
            architecture,
            len(classes),
            float(candidate_settings["dropout"]),
            int(candidate_settings["unfreeze_blocks"]),
            bool(settings["pretrained"]),
        )
        model.to(device)
        counts = Counter(target for _, target in train_dataset.samples)
        class_weights = torch.tensor(
            [
                len(train_dataset) / (len(classes) * counts[index])
                for index in range(len(classes))
            ],
            dtype=torch.float32,
            device=device,
        )
        criterion = nn.CrossEntropyLoss(
            weight=class_weights,
            label_smoothing=float(settings["label_smoothing"]),
        )
        head_ids = {id(parameter) for parameter in head.parameters()}
        backbone_parameters = [
            parameter
            for parameter in model.parameters()
            if parameter.requires_grad and id(parameter) not in head_ids
        ]
        optimizer = torch.optim.AdamW(
            [
                {"params": backbone_parameters, "lr": float(settings["backbone_lr"])},
                {"params": list(head.parameters()), "lr": float(settings["head_lr"])},
            ],
            weight_decay=float(settings["weight_decay"]),
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=int(settings["epochs"]), eta_min=1e-6
        )
        use_amp = device.type == "cuda"
        scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
        target_path = output_dir / f"{candidate}.pt"
        history: list[dict[str, Any]] = []
        best_macro_f1 = -1.0
        stale_epochs = 0

        for epoch in range(1, int(settings["epochs"]) + 1):
            model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            for inputs, labels in train_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(
                    device_type=device.type, dtype=torch.float16, enabled=use_amp
                ):
                    logits = model(inputs)
                    loss = criterion(logits, labels)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                train_loss += float(loss.item()) * labels.size(0)
                train_correct += int((logits.argmax(1) == labels).sum().item())
                train_total += labels.size(0)
            scheduler.step()
            validation = evaluate_classifier(model, val_loader, device, criterion)
            epoch_row = {
                "epoch": epoch,
                "train_loss": train_loss / max(1, train_total),
                "train_accuracy": train_correct / max(1, train_total),
                "val_loss": validation["loss"],
                "val_macro_f1": validation["macro_f1"],
            }
            history.append(epoch_row)
            print(
                f"epoch={epoch:03d} train_loss={epoch_row['train_loss']:.4f} "
                f"val_loss={epoch_row['val_loss']:.4f} val_f1={epoch_row['val_macro_f1']:.4f}"
            )
            if validation["macro_f1"] > best_macro_f1 + 1e-6:
                best_macro_f1 = validation["macro_f1"]
                stale_epochs = 0
                torch.save(
                    {
                        "model_name": architecture,
                        "candidate_name": candidate,
                        "state_dict": model.state_dict(),
                        "classes": classes,
                        "image_size": image_size,
                        "dropout": float(candidate_settings["dropout"]),
                        "unfreeze_blocks": int(candidate_settings["unfreeze_blocks"]),
                    },
                    target_path,
                )
            else:
                stale_epochs += 1
                if stale_epochs >= int(settings["patience"]):
                    break

        write_csv(output_dir / f"{candidate}_history.csv", history)
        checkpoint = torch.load(target_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["state_dict"])
        validation_result = evaluate_classifier(model, val_loader, device, criterion)
        test_result = evaluate_classifier(model, test_loader, device, criterion)
        matrix = confusion_matrix(
            test_result["targets"],
            test_result["predictions"],
            labels=list(range(len(classes))),
        ).tolist()

        sample, _ = test_dataset[0]
        sample = sample.unsqueeze(0).to(device)
        model.eval()
        with torch.inference_mode():
            for _ in range(5):
                model(sample)
            if device.type == "cuda":
                torch.cuda.synchronize()
            started = time.perf_counter()
            for _ in range(30):
                model(sample)
            if device.type == "cuda":
                torch.cuda.synchronize()
        latency_ms = (time.perf_counter() - started) * 1000 / 30

        validation_per_class = {
            name: {
                "precision": float(validation_result["precision"][index]),
                "recall": float(validation_result["recall"][index]),
                "f1": float(validation_result["f1"][index]),
                "support": int(validation_result["support"][index]),
            }
            for index, name in enumerate(classes)
        }
        test_per_class = {
            name: {
                "precision": float(test_result["precision"][index]),
                "recall": float(test_result["recall"][index]),
                "f1": float(test_result["f1"][index]),
                "support": int(test_result["support"][index]),
            }
            for index, name in enumerate(classes)
        }
        write_json(
            output_dir / f"{candidate}_report.json",
            {
                "validation_per_class": validation_per_class,
                "test_per_class": test_per_class,
                "test_confusion_matrix": matrix,
            },
        )
        values = {
            "accuracy": validation_result["accuracy"],
            "macro_f1": validation_result["macro_f1"],
            "min_class_recall": min(
                item["recall"] for item in validation_per_class.values()
            ),
            "test_accuracy": test_result["accuracy"],
            "test_macro_f1": test_result["macro_f1"],
            "test_min_class_recall": min(
                item["recall"] for item in test_per_class.values()
            ),
            "latency_ms": latency_ms,
            "size_mb": target_path.stat().st_size / (1024 * 1024),
            "image_size": image_size,
            "validation_per_class": validation_per_class,
            "test_per_class": test_per_class,
        }
        metrics_by_candidate[candidate] = values
        comparison_rows.append(
            {
                "candidate": candidate,
                **{
                    key: values[key]
                    for key in ("macro_f1", "min_class_recall", "latency_ms", "size_mb")
                },
            }
        )
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    write_json(
        ROOT / "metrics" / "classifiers.json", {"candidates": metrics_by_candidate}
    )
    write_csv(ROOT / "metrics" / "classifiers.csv", comparison_rows)


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
        len(checkpoint["classes"]),
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
    manifest: dict[str, Any] = {
        "schema_version": 2,
        "deploy_ready": segmenter_passed and classifier_passed,
        "deployment_score": stable_score,
        "dataset": dataset_manifest,
        "classes": list(config["data"]["classes"]),
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
            ROOT / "metrics" / "segmenters.json",
            ROOT / "metrics" / "classifiers.json",
            ROOT / "metrics" / "selection.json",
        ):
            mlflow.log_artifact(str(artifact), artifact_path="reproducibility")
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


def predict(source: Path, output_dir: Path, bundle_dir: Path) -> None:
    try:
        import torch
        from torchvision import transforms
        from ultralytics import YOLO
    except ImportError as error:
        raise RuntimeError("Install requirements.txt before inference") from error

    deployment_dir = project_path(bundle_dir)
    manifest = json.loads(
        (deployment_dir / "manifest.json").read_text(encoding="utf-8")
    )
    if (
        sha256_file(deployment_dir / "leaf_segmenter.pt")
        != manifest["segmenter"]["sha256"]
    ):
        raise RuntimeError("Segmenter checksum does not match deployment manifest")
    if (
        sha256_file(deployment_dir / "leaf_classifier.pt")
        != manifest["classifier"]["sha256"]
    ):
        raise RuntimeError("Classifier checksum does not match deployment manifest")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(
        deployment_dir / "leaf_classifier.pt", map_location=device, weights_only=False
    )
    classifier, _ = build_classifier(
        checkpoint["model_name"],
        len(checkpoint["classes"]),
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
    detector = YOLO(str(deployment_dir / "leaf_segmenter.pt"))
    results = detector.predict(
        source=str(source),
        conf=float(manifest["thresholds"]["detector_confidence"]),
        imgsz=int(manifest["segmenter"]["image_size"]),
        verbose=False,
    )
    if len(results) != 1:
        raise ValueError("predict currently accepts exactly one image")
    result = results[0]
    with Image.open(source) as opened:
        image = opened.convert("RGB")
    width, height = image.size
    leaves: list[dict[str, Any]] = []
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

    for index, (mask, box) in enumerate(zip(masks, boxes)):
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
        crop_tensors.append(transform(crop))
        leaves.append(
            {
                "leaf_id": index,
                "bbox_xyxy": [x1, y1, x2, y2],
                "detector_confidence": float(detector_confidences[index]),
            }
        )

    if crop_tensors:
        batch = torch.stack(crop_tensors).to(device)
        with torch.inference_mode():
            probabilities = classifier(batch).softmax(1).cpu()
        for leaf, probability in zip(leaves, probabilities):
            confidence, class_id = probability.max(0)
            confidence_value = float(confidence.item())
            leaf["predicted_class"] = checkpoint["classes"][int(class_id.item())]
            leaf["classification_confidence"] = confidence_value
            leaf["accepted"] = confidence_value >= float(
                manifest["thresholds"]["classifier_confidence"]
            )
            leaf["probabilities"] = {
                name: float(value)
                for name, value in zip(checkpoint["classes"], probability.tolist())
            }

    accepted = [leaf for leaf in leaves if leaf.get("accepted")]
    counts = Counter(leaf["predicted_class"] for leaf in accepted)
    diseased = sum(count for name, count in counts.items() if name != "healthy")
    payload = {
        "source": str(source),
        "deploy_ready": bool(manifest["deploy_ready"]),
        "summary": {
            "detected_leaves": len(leaves),
            "accepted_classifications": len(accepted),
            "class_counts": dict(counts),
            "diseased_leaf_fraction": diseased / len(accepted) if accepted else None,
        },
        "leaves": leaves,
    }
    output_dir = reset_dir(output_dir)
    write_json(output_dir / "result.json", payload)

    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    overlay_draw = ImageDraw.Draw(overlay)
    colors = [
        (64, 180, 75, 75),
        (255, 179, 0, 75),
        (225, 87, 89, 75),
        (116, 78, 170, 75),
    ]
    for index, (leaf, mask) in enumerate(zip(leaves, masks)):
        class_name = leaf.get("predicted_class", "unknown")
        class_index = (
            checkpoint["classes"].index(class_name)
            if class_name in checkpoint["classes"]
            else 0
        )
        color = colors[class_index % len(colors)]
        color_layer = Image.new("RGBA", image.size, color)
        overlay.alpha_composite(
            Image.composite(color_layer, Image.new("RGBA", image.size), mask)
        )
        box = leaf["bbox_xyxy"]
        label = f"{class_name} {leaf.get('classification_confidence', 0.0):.2f}"
        overlay_draw.rectangle(box, outline=color[:3] + (255,), width=3)
        overlay_draw.text(
            (box[0] + 3, max(0, box[1] - 13)), label, fill=color[:3] + (255,)
        )
    Image.alpha_composite(image.convert("RGBA"), overlay).convert("RGB").save(
        output_dir / "annotated.jpg", quality=95
    )
    print(json.dumps(payload["summary"], ensure_ascii=False, indent=2))


def doctor(config: dict[str, Any], check_data: bool) -> None:
    errors: list[str] = []
    classes = config.get("data", {}).get("classes", [])
    if len(classes) < 2 or len(classes) != len(set(classes)):
        errors.append("data.classes must contain at least two unique class names")
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
    for candidate, settings in (
        config.get("classification", {}).get("candidates", {}).items()
    ):
        architecture = str(settings.get("architecture", candidate))
        if architecture not in CLASSIFIER_ARCHITECTURES:
            errors.append(
                f"classification candidate {candidate!r} uses unsupported "
                f"architecture {architecture!r}"
            )
    if check_data:
        try:
            records, _ = load_records(
                project_path(config["data"]["raw_dir"]), config["data"]["splits"]
            )
            for record in records:
                read_polygons(record.label, len(classes))
        except (OSError, ValueError) as error:
            errors.append(str(error))

    packages = {}
    for name in (
        "dagshub",
        "dvc",
        "mlflow",
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
        "train-segmenters", help="Train and benchmark segmentation candidates"
    )
    subparsers.add_parser(
        "train-classifiers", help="Train and benchmark classification candidates"
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
    elif args.command == "train-segmenters":
        train_segmenters(config)
    elif args.command == "train-classifiers":
        train_classifiers(config)
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
