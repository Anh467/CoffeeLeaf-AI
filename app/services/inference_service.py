"""End-to-end inference orchestration for the web API."""

from __future__ import annotations

import json
import logging
import shutil
import threading
import time
import uuid
from pathlib import Path
from typing import Any

from PIL import Image

from app.database.sqlite import HistoryDatabase
from app.services import classification_service, segmentation_service, visualization_service
from app.services.statistics_service import build_summary
from pipeline import IMAGE_EXTENSIONS, ROOT, InferenceBundle, load_inference_bundle, write_json

LOGGER = logging.getLogger(__name__)

HISTORY_ROOT = ROOT / "runs" / "history"
STATIC_MEDIA_PREFIX = "/media/history"


class InferenceService:
    """Warm model bundle + thread-safe predict/history API."""

    def __init__(
        self,
        bundle_dir: Path | str = "deployment",
        history_root: Path | None = None,
        database: HistoryDatabase | None = None,
    ) -> None:
        self.bundle_dir = Path(bundle_dir)
        self.history_root = history_root or HISTORY_ROOT
        self.history_root.mkdir(parents=True, exist_ok=True)
        self.database = database or HistoryDatabase(self.history_root / "history.db")
        self._bundle: InferenceBundle | None = None
        self._lock = threading.RLock()
        self._load_error: str | None = None

    @property
    def models_loaded(self) -> bool:
        return self._bundle is not None

    @property
    def load_error(self) -> str | None:
        return self._load_error

    def startup(self) -> None:
        try:
            self._bundle = load_inference_bundle(self.bundle_dir)
            self._load_error = None
            LOGGER.info(
                "Loaded bundle segmenter=%s classifier=%s device=%s",
                self._bundle.segmenter_name,
                self._bundle.classifier_name,
                self._bundle.device,
            )
        except Exception as error:  # noqa: BLE001 - surface startup issues to API health
            self._bundle = None
            self._load_error = str(error)
            LOGGER.exception("Failed to load deployment bundle from %s", self.bundle_dir)

    def shutdown(self) -> None:
        self._bundle = None

    def require_bundle(self) -> InferenceBundle:
        if self._bundle is None:
            detail = self._load_error or "Deployment bundle is not loaded"
            raise RuntimeError(
                f"{detail}. Ensure `deployment/` exists (dvc pull deployment) "
                "and restart the server."
            )
        return self._bundle

    def health(self) -> dict[str, Any]:
        bundle = self._bundle
        return {
            "status": "ok" if bundle is not None else "degraded",
            "models_loaded": bundle is not None,
            "segmenter": None if bundle is None else bundle.segmenter_name,
            "classifier": None if bundle is None else bundle.classifier_name,
            "device": None if bundle is None else str(bundle.device),
            "error": self._load_error,
        }

    def predict_upload(
        self,
        *,
        image_bytes: bytes,
        filename: str,
        allow_single_leaf_fallback: bool = True,
    ) -> dict[str, Any]:
        extension = Path(filename).suffix.lower()
        if extension not in IMAGE_EXTENSIONS:
            raise ValueError(
                f"Unsupported image type '{extension}'. "
                f"Allowed: {', '.join(sorted(IMAGE_EXTENSIONS))}"
            )

        prediction_id = uuid.uuid4().hex
        run_dir = self.history_root / prediction_id
        run_dir.mkdir(parents=True, exist_ok=True)
        original_path = run_dir / f"original{extension}"
        original_path.write_bytes(image_bytes)
        file_size = original_path.stat().st_size

        with Image.open(original_path) as opened:
            image = opened.convert("RGB")

        total_started = time.perf_counter()
        with self._lock:
            bundle = self.require_bundle()
            payload = self._run_pipeline(
                image=image,
                bundle=bundle,
                allow_single_leaf_fallback=allow_single_leaf_fallback,
            )
        total_ms = (time.perf_counter() - total_started) * 1000.0

        enriched_leaves = visualization_service.write_leaf_artifacts(
            run_dir,
            payload["leaves"],
            payload["masks"],
            payload["crops"],
        )
        overlay_path = visualization_service.save_prediction_overlay(
            image,
            payload["leaves"],
            payload["masks"],
            payload["healthy_label"],
            payload["disease_classes"],
            run_dir / "overlay.jpg",
        )
        mask_path = visualization_service.save_mask_overlay(
            image,
            payload["leaves"],
            payload["masks"],
            payload["healthy_label"],
            run_dir / "mask_overlay.jpg",
        )
        boxes_path = visualization_service.save_boxes_overlay(
            image, payload["leaves"], run_dir / "boxes.jpg"
        )
        thumbnail_path = visualization_service.save_thumbnail(
            image, run_dir / "thumbnail.jpg"
        )

        summary = build_summary(payload["leaves"], payload["healthy_label"])
        processing = {
            "preprocess_time_ms": float(payload["timing"]["preprocess_ms"]),
            "segmentation_time_ms": float(payload["timing"]["segmentation_ms"]),
            "classification_time_ms": float(payload["timing"]["classification_ms"]),
            "total_time_ms": float(total_ms),
        }
        model_info = {
            "segmenter": payload["segmenter"],
            "classifier": payload["classifier"],
            "version": str(payload["manifest"].get("schema_version", "")),
            "deploy_ready": bool(payload["deploy_ready"]),
            "detector_confidence": float(
                payload["manifest"]["thresholds"]["detector_confidence"]
            ),
            "classifier_confidence": float(
                payload["manifest"]["thresholds"]["classifier_confidence"]
            ),
        }

        response = {
            "id": prediction_id,
            "image": {
                "width": image.size[0],
                "height": image.size[1],
                "file_size": file_size,
                "filename": Path(filename).name,
                "mode": payload["mode"],
            },
            "processing": processing,
            "model": model_info,
            "summary": summary,
            "leaves": [
                self._serialize_leaf(leaf, prediction_id) for leaf in enriched_leaves
            ],
            "visualizations": {
                "original": self._media_url(prediction_id, original_path.name),
                "overlay": self._media_url(prediction_id, overlay_path.name),
                "mask_overlay": self._media_url(prediction_id, mask_path.name),
                "boxes": self._media_url(prediction_id, boxes_path.name),
                "thumbnail": self._media_url(prediction_id, thumbnail_path.name),
            },
            "result_path": str(run_dir / "result.json"),
        }
        write_json(run_dir / "result.json", response)
        self.database.insert(
            prediction_id=prediction_id,
            filename=Path(filename).name,
            image_path=original_path,
            result_path=run_dir / "result.json",
            processing_time=total_ms,
            model=f"{model_info['segmenter']}+{model_info['classifier']}",
            summary=summary,
        )
        return response

    def list_history(self, limit: int = 50, offset: int = 0) -> dict[str, Any]:
        items = []
        for row in self.database.list_items(limit=limit, offset=offset):
            summary = row["summary"]
            thumbnail = self.history_root / row["id"] / "thumbnail.jpg"
            items.append(
                {
                    "id": row["id"],
                    "filename": row["filename"],
                    "created_at": row["created_at"],
                    "processing_time_ms": float(row["processing_time"]),
                    "model": row["model"],
                    "summary": summary,
                    "thumbnail": (
                        self._media_url(row["id"], "thumbnail.jpg")
                        if thumbnail.exists()
                        else None
                    ),
                    "total_leaves": int(summary.get("total_leaves", 0)),
                }
            )
        return {"items": items, "total": self.database.count()}

    def get_history(self, prediction_id: str) -> dict[str, Any]:
        row = self.database.get(prediction_id)
        if row is None:
            raise KeyError(f"Prediction not found: {prediction_id}")
        result_path = Path(row["result_path"])
        if not result_path.exists():
            raise FileNotFoundError(f"Result file missing for {prediction_id}")
        payload = json.loads(result_path.read_text(encoding="utf-8"))
        payload["created_at"] = row["created_at"]
        payload["filename"] = row["filename"]
        return payload

    def delete_history(self, prediction_id: str) -> bool:
        row = self.database.get(prediction_id)
        if row is None:
            return False
        deleted = self.database.delete(prediction_id)
        run_dir = self.history_root / prediction_id
        if run_dir.exists():
            shutil.rmtree(run_dir, ignore_errors=True)
        return deleted

    def _run_pipeline(
        self,
        *,
        image: Image.Image,
        bundle: InferenceBundle,
        allow_single_leaf_fallback: bool,
    ) -> dict[str, Any]:
        segmentation = segmentation_service.segment_leaves(image, bundle)
        masks = segmentation["masks"]
        boxes = segmentation["boxes"]
        confidences = segmentation["confidences"]

        if masks and boxes:
            mode = "tree"
            prepared = classification_service.prepare_leaf_crops(
                image, masks, boxes, confidences, bundle
            )
            used_masks = masks
        elif allow_single_leaf_fallback:
            mode = "single_leaf"
            prepared = classification_service.prepare_single_leaf_crop(image, bundle)
            used_masks = prepared["masks"]
            segmentation["elapsed_ms"] = 0.0
        else:
            mode = "tree"
            prepared = {
                "elapsed_ms": 0.0,
                "leaves": [],
                "crops": [],
                "tensors": [],
            }
            used_masks = []

        classified = classification_service.classify_crops(
            prepared["tensors"], prepared["leaves"], bundle
        )
        return {
            "mode": mode,
            "timing": {
                "preprocess_ms": float(prepared["elapsed_ms"]),
                "segmentation_ms": float(segmentation["elapsed_ms"]),
                "classification_ms": float(classified["elapsed_ms"]),
            },
            "leaves": classified["leaves"],
            "masks": used_masks,
            "crops": prepared["crops"],
            "healthy_label": bundle.healthy_label,
            "disease_classes": list(bundle.disease_classes),
            "segmenter": bundle.segmenter_name,
            "classifier": bundle.classifier_name,
            "deploy_ready": bool(bundle.manifest["deploy_ready"]),
            "manifest": bundle.manifest,
        }

    def _serialize_leaf(self, leaf: dict[str, Any], prediction_id: str) -> dict[str, Any]:
        labels = list(leaf.get("predicted_labels", []))
        prediction = "+".join(labels) if labels else "unknown"
        leaf_id = int(leaf["leaf_id"]) + 1
        relative_root = self.history_root / prediction_id
        mask_path = leaf.get("mask_path")
        crop_path = leaf.get("crop_path")
        return {
            "leaf_id": leaf_id,
            "bbox": [int(value) for value in leaf["bbox_xyxy"]],
            "mask": (
                self._media_url(
                    prediction_id,
                    Path(mask_path).relative_to(relative_root).as_posix(),
                )
                if mask_path is not None
                else None
            ),
            "crop": (
                self._media_url(
                    prediction_id,
                    Path(crop_path).relative_to(relative_root).as_posix(),
                )
                if crop_path is not None
                else None
            ),
            "prediction": prediction,
            "labels": labels,
            "confidence": float(leaf.get("classification_confidence", 0.0)),
            "detector_confidence": float(leaf.get("detector_confidence", 0.0)),
            "accepted": bool(leaf.get("accepted", False)),
            "probabilities": {
                str(key): float(value)
                for key, value in dict(leaf.get("probabilities", {})).items()
            },
        }

    @staticmethod
    def _media_url(prediction_id: str, relative: str) -> str:
        return f"{STATIC_MEDIA_PREFIX}/{prediction_id}/{relative.lstrip('/')}"
