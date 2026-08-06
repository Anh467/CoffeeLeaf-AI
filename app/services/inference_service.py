"""End-to-end inference orchestration for the web API."""

from __future__ import annotations

import json
import logging
import shutil
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Literal

from PIL import Image

from app.database.sqlite import HistoryDatabase
from app.services import classification_service, segmentation_service, visualization_service
from app.services.statistics_service import (
    build_summary,
    display_label_for_leaf,
    is_healthy_leaf,
)
from pipeline import IMAGE_EXTENSIONS, ROOT, InferenceBundle, load_inference_bundle, write_json

LOGGER = logging.getLogger(__name__)

HISTORY_ROOT = ROOT / "runs" / "history"
STATIC_MEDIA_PREFIX = "/media/history"
PredictMode = Literal["auto", "whole_image", "single_leaf"]
VALID_MODES = {"auto", "whole_image", "single_leaf"}


def resolve_predict_mode(
    mode: str | None = None,
    *,
    allow_single_leaf_fallback: bool | None = None,
) -> PredictMode:
    """Map request options to a canonical predict mode.

    Legacy clients that only send ``allow_single_leaf_fallback`` keep working:
    True -> auto, False -> whole_image.
    """
    if mode:
        normalized = str(mode).strip().lower()
        # Accept older "tree" alias used in previous responses.
        if normalized == "tree":
            return "whole_image"
        if normalized not in VALID_MODES:
            raise ValueError(
                f"Unsupported mode '{mode}'. Allowed: auto, whole_image, single_leaf"
            )
        return normalized  # type: ignore[return-value]
    if allow_single_leaf_fallback is False:
        return "whole_image"
    return "auto"


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
        mode: str | None = None,
        allow_single_leaf_fallback: bool | None = None,
    ) -> dict[str, Any]:
        extension = Path(filename).suffix.lower()
        if extension not in IMAGE_EXTENSIONS:
            raise ValueError(
                f"Unsupported image type '{extension}'. "
                f"Allowed: {', '.join(sorted(IMAGE_EXTENSIONS))}"
            )
        predict_mode = resolve_predict_mode(
            mode, allow_single_leaf_fallback=allow_single_leaf_fallback
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
                requested_mode=predict_mode,
            )
        total_ms = (time.perf_counter() - total_started) * 1000.0

        draw_annotations = payload["analysis_scope"] == "leaf_instances" and bool(
            payload["leaves"]
        )
        viz_started = time.perf_counter()
        enriched_leaves = visualization_service.write_leaf_artifacts(
            run_dir,
            payload["leaves"],
            payload["masks"],
            payload["crops"] if payload["leaves"] else [],
        )
        overlay_path = visualization_service.save_prediction_overlay(
            image,
            payload["leaves"],
            payload["masks"],
            payload["healthy_label"],
            payload["disease_classes"],
            run_dir / "overlay.jpg",
            draw_annotations=draw_annotations,
        )
        mask_path = visualization_service.save_mask_overlay(
            image,
            payload["leaves"],
            payload["masks"],
            payload["healthy_label"],
            run_dir / "mask_overlay.jpg",
            draw_annotations=draw_annotations,
        )
        boxes_path = visualization_service.save_boxes_overlay(
            image,
            payload["leaves"],
            run_dir / "boxes.jpg",
            healthy_label=payload["healthy_label"],
            draw_annotations=draw_annotations,
        )
        thumbnail_path = visualization_service.save_thumbnail(
            image, run_dir / "thumbnail.jpg"
        )
        visualization_ms = (time.perf_counter() - viz_started) * 1000.0

        response_leaves = [
            self._serialize_leaf(leaf, prediction_id, payload["healthy_label"])
            for leaf in enriched_leaves
        ]
        full_image_result = None
        if payload.get("full_image_leaf") is not None:
            # Optional crop preview for full-image analysis (not counted as a leaf).
            crop_url = None
            if payload["crops"]:
                crop_dir = run_dir / "crops"
                crop_dir.mkdir(parents=True, exist_ok=True)
                crop_path = crop_dir / "full_image.jpg"
                payload["crops"][0].convert("RGB").save(crop_path, quality=92)
                crop_url = self._media_url(prediction_id, "crops/full_image.jpg")
            full_image_result = self._serialize_full_image_result(
                payload["full_image_leaf"],
                payload["healthy_label"],
                crop_url=crop_url,
            )

        summary = build_summary(
            payload["leaves"],
            payload["healthy_label"],
            disease_classes=payload["disease_classes"],
            mode=payload["resolved_mode"],
            fallback_to_single_leaf=payload["fallback_to_single_leaf"],
        )
        if full_image_result is not None and not response_leaves:
            summary["average_confidence"] = float(
                full_image_result.get("classification_confidence", 0.0)
            )
            summary["detected_diseases"] = [
                name
                for name in full_image_result.get("labels", [])
                if name != payload["healthy_label"]
            ]
        self._assert_prediction_consistency(summary, response_leaves)

        message = self._build_message(payload, summary)

        processing = {
            "preprocess_time_ms": float(payload["timing"]["preprocess_ms"]),
            "segmentation_time_ms": float(payload["timing"]["segmentation_ms"]),
            "classification_time_ms": float(payload["timing"]["classification_ms"]),
            "visualization_time_ms": float(visualization_ms),
            "total_time_ms": float(total_ms),
            "mode": payload["resolved_mode"],
            "requested_mode": payload["requested_mode"],
            "resolved_mode": payload["resolved_mode"],
            "fallback_to_single_leaf": bool(payload["fallback_to_single_leaf"]),
            "analysis_scope": payload["analysis_scope"],
            "raw_instances": int(payload["raw_instances"]),
            "valid_instances": int(payload["valid_instances"]),
            "segmenter": payload["segmenter"],
            "classifier": payload["classifier"],
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
            "device": str(self._bundle.device) if self._bundle is not None else "unknown",
        }

        LOGGER.info(
            "Prediction requested_mode=%s resolved_mode=%s scope=%s raw_instances=%d "
            "valid_instances=%d leaves=%d fallback=%s segmentation_ms=%.1f",
            processing["requested_mode"],
            processing["resolved_mode"],
            processing["analysis_scope"],
            processing["raw_instances"],
            processing["valid_instances"],
            len(response_leaves),
            processing["fallback_to_single_leaf"],
            processing["segmentation_time_ms"],
        )

        response = {
            "id": prediction_id,
            "image": {
                "width": image.size[0],
                "height": image.size[1],
                "file_size": file_size,
                "filename": Path(filename).name,
                "mode": payload["resolved_mode"],
                "format": extension.lstrip(".").upper() or "UNKNOWN",
            },
            "processing": processing,
            "model": model_info,
            "summary": summary,
            "leaves": response_leaves,
            "full_image_result": full_image_result,
            "visualizations": {
                "original": self._media_url(prediction_id, original_path.name),
                "overlay": self._media_url(prediction_id, overlay_path.name),
                "mask_overlay": self._media_url(prediction_id, mask_path.name),
                "mask": self._media_url(prediction_id, mask_path.name),
                "boxes": self._media_url(prediction_id, boxes_path.name),
                "thumbnail": self._media_url(prediction_id, thumbnail_path.name),
            },
            "result_path": str(run_dir / "result.json"),
            "message": message,
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
        return self._normalize_legacy_payload(payload)

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
        requested_mode: PredictMode,
    ) -> dict[str, Any]:
        fallback = False
        segmentation_ms = 0.0
        raw_instances = 0
        valid_instances = 0
        analysis_scope = "leaf_instances"
        full_image_leaf: dict[str, Any] | None = None
        used_masks: list = []
        crops: list = []

        if requested_mode == "single_leaf":
            # Explicit full-image classification (API compatibility).
            prepared = classification_service.prepare_single_leaf_crop(image, bundle)
            classified = classification_service.classify_crops(
                prepared["tensors"], prepared["leaves"], bundle
            )
            full_image_leaf = classified["leaves"][0] if classified["leaves"] else None
            crops = prepared["crops"]
            resolved_mode: PredictMode = "single_leaf"
            analysis_scope = "full_image"
            leaves: list[dict[str, Any]] = []
            timing = {
                "preprocess_ms": float(prepared["elapsed_ms"]),
                "segmentation_ms": 0.0,
                "classification_ms": float(classified["elapsed_ms"]),
            }
        else:
            segmentation = segmentation_service.segment_leaves(image, bundle)
            segmentation_ms = float(segmentation["elapsed_ms"])
            raw_instances = int(segmentation.get("raw_instances", 0))
            instances = segmentation["instances"]
            valid_instances = len(instances)
            if instances:
                prepared = classification_service.prepare_leaf_crops(
                    image, instances, bundle
                )
                used_masks = prepared["masks"]
                crops = prepared["crops"]
                classified = classification_service.classify_crops(
                    prepared["tensors"], prepared["leaves"], bundle
                )
                leaves = classified["leaves"]
                resolved_mode = "whole_image"
                analysis_scope = "leaf_instances"
                timing = {
                    "preprocess_ms": float(prepared["elapsed_ms"]),
                    "segmentation_ms": segmentation_ms,
                    "classification_ms": float(classified["elapsed_ms"]),
                }
            elif requested_mode == "auto":
                # Auto fallback: classify whole image, but do NOT invent a detected leaf.
                fallback = True
                prepared = classification_service.prepare_single_leaf_crop(image, bundle)
                classified = classification_service.classify_crops(
                    prepared["tensors"], prepared["leaves"], bundle
                )
                full_image_leaf = (
                    classified["leaves"][0] if classified["leaves"] else None
                )
                crops = prepared["crops"]
                leaves = []
                valid_instances = 0
                resolved_mode = "single_leaf"
                analysis_scope = "full_image"
                timing = {
                    "preprocess_ms": float(prepared["elapsed_ms"]),
                    "segmentation_ms": segmentation_ms,
                    "classification_ms": float(classified["elapsed_ms"]),
                }
            else:
                # whole_image with zero detections: empty result, no crash.
                leaves = []
                resolved_mode = "whole_image"
                analysis_scope = "leaf_instances"
                timing = {
                    "preprocess_ms": 0.0,
                    "segmentation_ms": segmentation_ms,
                    "classification_ms": 0.0,
                }

        return {
            "requested_mode": requested_mode,
            "resolved_mode": resolved_mode,
            "mode": resolved_mode,
            "fallback_to_single_leaf": fallback,
            "analysis_scope": analysis_scope,
            "raw_instances": raw_instances,
            "valid_instances": valid_instances,
            "timing": timing,
            "leaves": leaves,
            "full_image_leaf": full_image_leaf,
            "masks": used_masks,
            "crops": crops,
            "healthy_label": bundle.healthy_label,
            "disease_classes": list(bundle.disease_classes),
            "segmenter": bundle.segmenter_name,
            "classifier": bundle.classifier_name,
            "deploy_ready": bool(bundle.manifest["deploy_ready"]),
            "manifest": bundle.manifest,
        }

    @staticmethod
    def _assert_prediction_consistency(
        summary: dict[str, Any],
        leaves: list[dict[str, Any]],
    ) -> None:
        total = int(summary.get("total_leaves", 0))
        healthy = int(summary.get("healthy_leaves", 0))
        diseased = int(summary.get("diseased_leaves", 0))
        if total != len(leaves):
            raise RuntimeError(
                f"Inconsistent prediction: total_leaves={total} but len(leaves)={len(leaves)}"
            )
        if healthy + diseased != total:
            raise RuntimeError(
                "Inconsistent prediction: "
                f"healthy_leaves({healthy}) + diseased_leaves({diseased}) != total_leaves({total})"
            )

    @staticmethod
    def _build_message(payload: dict[str, Any], summary: dict[str, Any]) -> str | None:
        if payload.get("analysis_scope") == "full_image":
            if payload.get("fallback_to_single_leaf"):
                return (
                    "Không phát hiện được từng lá riêng biệt. "
                    "Hệ thống đã tự động phân tích toàn bộ ảnh."
                )
            return (
                "Đã phân tích toàn bộ ảnh như một đơn vị. "
                "Leaf segmentation và leaf counting không được thực hiện."
            )
        if summary["total_leaves"] == 0 and payload.get("requested_mode") == "whole_image":
            return (
                "Không phát hiện được lá nào trong ảnh. "
                "Hãy thử ảnh rõ hơn hoặc để hệ thống tự phân tích toàn ảnh (auto)."
            )
        return None

    def _serialize_full_image_result(
        self,
        leaf: dict[str, Any],
        healthy_label: str,
        *,
        crop_url: str | None = None,
    ) -> dict[str, Any]:
        labels = list(leaf.get("predicted_labels", []))
        display_label = display_label_for_leaf(leaf, healthy_label)
        prediction = "+".join(labels) if labels else "unknown"
        return {
            "prediction": prediction,
            "labels": labels,
            "display_label": display_label,
            "is_healthy": is_healthy_leaf(leaf, healthy_label),
            "confidence": float(leaf.get("classification_confidence", 0.0)),
            "classification_confidence": float(leaf.get("classification_confidence", 0.0)),
            "accepted": bool(leaf.get("accepted", False)),
            "probabilities": {
                str(key): float(value)
                for key, value in dict(leaf.get("probabilities", {})).items()
            },
            "crop": crop_url,
        }

    def _serialize_leaf(
        self,
        leaf: dict[str, Any],
        prediction_id: str,
        healthy_label: str,
    ) -> dict[str, Any]:
        labels = list(leaf.get("predicted_labels", []))
        display_label = display_label_for_leaf(leaf, healthy_label)
        prediction = "+".join(labels) if labels else "unknown"
        leaf_id = int(leaf["leaf_id"]) + 1
        relative_root = self.history_root / prediction_id
        mask_path = leaf.get("mask_path")
        crop_path = leaf.get("crop_path")
        bbox_list = [int(value) for value in leaf["bbox_xyxy"]]
        return {
            "leaf_id": leaf_id,
            "bbox": {
                "x1": bbox_list[0],
                "y1": bbox_list[1],
                "x2": bbox_list[2],
                "y2": bbox_list[3],
            },
            # Legacy list form kept for older clients / history tooling.
            "bbox_xyxy": bbox_list,
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
            "crop_path": (
                self._media_url(
                    prediction_id,
                    Path(crop_path).relative_to(relative_root).as_posix(),
                )
                if crop_path is not None
                else None
            ),
            "prediction": prediction,
            "labels": labels,
            "display_label": display_label,
            "is_healthy": is_healthy_leaf(leaf, healthy_label),
            "confidence": float(leaf.get("classification_confidence", 0.0)),
            "classification_confidence": float(leaf.get("classification_confidence", 0.0)),
            "detector_confidence": float(leaf.get("detector_confidence", 0.0)),
            "segmentation_confidence": float(
                leaf.get("segmentation_confidence", leaf.get("detector_confidence", 0.0))
            ),
            "mask_area": int(leaf.get("mask_area", 0)),
            "accepted": bool(leaf.get("accepted", False)),
            "probabilities": {
                str(key): float(value)
                for key, value in dict(leaf.get("probabilities", {})).items()
            },
        }

    @staticmethod
    def _normalize_legacy_payload(payload: dict[str, Any]) -> dict[str, Any]:
        """Upgrade older saved results so they validate against current schemas."""
        summary = dict(payload.get("summary") or {})
        if "disease_counts" not in summary:
            class_counts = dict(summary.get("class_counts") or {})
            summary["disease_counts"] = {
                key: int(value)
                for key, value in class_counts.items()
                if key != "healthy"
            }
        summary.setdefault("multi_disease_leaves", 0)
        payload["summary"] = summary

        processing = dict(payload.get("processing") or {})
        processing.setdefault("visualization_time_ms", 0.0)
        processing.setdefault("mode", payload.get("image", {}).get("mode", "auto"))
        processing.setdefault("requested_mode", processing.get("mode", "auto"))
        processing.setdefault("resolved_mode", processing.get("mode", "auto"))
        processing.setdefault("fallback_to_single_leaf", False)
        processing.setdefault("raw_instances", 0)
        processing.setdefault(
            "valid_instances",
            int((payload.get("summary") or {}).get("total_leaves", 0)),
        )
        processing.setdefault("analysis_scope", "leaf_instances")
        processing.setdefault("segmenter", (payload.get("model") or {}).get("segmenter"))
        processing.setdefault("classifier", (payload.get("model") or {}).get("classifier"))
        payload["processing"] = processing
        payload.setdefault("full_image_result", None)

        image_info = dict(payload.get("image") or {})
        image_info.setdefault("format", Path(str(image_info.get("filename", ""))).suffix.lstrip(".").upper() or None)
        payload["image"] = image_info

        model_info = dict(payload.get("model") or {})
        model_info.setdefault("device", None)
        payload["model"] = model_info

        visualizations = dict(payload.get("visualizations") or {})
        if "mask" not in visualizations and "mask_overlay" in visualizations:
            visualizations["mask"] = visualizations["mask_overlay"]
        payload["visualizations"] = visualizations

        leaves = []
        for leaf in payload.get("leaves") or []:
            item = dict(leaf)
            bbox = item.get("bbox")
            if isinstance(bbox, list) and len(bbox) == 4:
                item["bbox"] = {
                    "x1": int(bbox[0]),
                    "y1": int(bbox[1]),
                    "x2": int(bbox[2]),
                    "y2": int(bbox[3]),
                }
                item.setdefault("bbox_xyxy", [int(v) for v in bbox])
            elif isinstance(bbox, dict):
                item.setdefault(
                    "bbox_xyxy",
                    [
                        int(bbox.get("x1", 0)),
                        int(bbox.get("y1", 0)),
                        int(bbox.get("x2", 0)),
                        int(bbox.get("y2", 0)),
                    ],
                )
            labels = list(item.get("labels") or [])
            item.setdefault("display_label", " + ".join(labels) if labels else "unknown")
            item.setdefault("is_healthy", labels == ["healthy"] or labels == [])
            item.setdefault(
                "classification_confidence",
                float(item.get("confidence", 0.0)),
            )
            item.setdefault(
                "segmentation_confidence",
                float(item.get("detector_confidence", 0.0)),
            )
            item.setdefault("mask_area", 0)
            item.setdefault("crop_path", item.get("crop"))
            leaves.append(item)
        payload["leaves"] = leaves
        payload.setdefault("message", None)
        return payload

    @staticmethod
    def _media_url(prediction_id: str, relative: str) -> str:
        return f"{STATIC_MEDIA_PREFIX}/{prediction_id}/{relative.lstrip('/')}"
