"""Response schemas for the CoffeeLeaf web API."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, field_validator


class BBox(BaseModel):
    x1: int
    y1: int
    x2: int
    y2: int


class ImageInfo(BaseModel):
    width: int
    height: int
    file_size: int
    filename: str
    mode: str = Field(description="auto | whole_image | single_leaf (legacy: tree)")


class ProcessingInfo(BaseModel):
    preprocess_time_ms: float
    segmentation_time_ms: float
    classification_time_ms: float
    total_time_ms: float
    visualization_time_ms: float = 0.0
    mode: str | None = None
    fallback_to_single_leaf: bool = False
    segmenter: str | None = None
    classifier: str | None = None


class ModelInfo(BaseModel):
    segmenter: str
    classifier: str
    version: str
    deploy_ready: bool
    detector_confidence: float
    classifier_confidence: float


class SummaryInfo(BaseModel):
    total_leaves: int
    healthy_leaves: int
    diseased_leaves: int
    average_confidence: float
    detected_diseases: list[str] = Field(default_factory=list)
    class_counts: dict[str, int] = Field(default_factory=dict)
    disease_counts: dict[str, int] = Field(default_factory=dict)
    multi_disease_leaves: int = 0
    mode: str | None = None
    fallback_to_single_leaf: bool = False


class LeafPrediction(BaseModel):
    leaf_id: int
    bbox: BBox
    bbox_xyxy: list[int] | None = None
    mask: str | None = None
    crop: str | None = None
    crop_path: str | None = None
    prediction: str
    labels: list[str]
    display_label: str | None = None
    is_healthy: bool | None = None
    confidence: float
    classification_confidence: float | None = None
    detector_confidence: float
    segmentation_confidence: float | None = None
    mask_area: int = 0
    accepted: bool
    probabilities: dict[str, float]

    @field_validator("bbox", mode="before")
    @classmethod
    def normalize_bbox(cls, value: Any) -> Any:
        if isinstance(value, (list, tuple)) and len(value) == 4:
            return {
                "x1": int(value[0]),
                "y1": int(value[1]),
                "x2": int(value[2]),
                "y2": int(value[3]),
            }
        return value


class VisualizationPaths(BaseModel):
    original: str
    overlay: str
    mask_overlay: str
    boxes: str
    thumbnail: str
    mask: str | None = None


class PredictResponse(BaseModel):
    id: str
    image: ImageInfo
    processing: ProcessingInfo
    model: ModelInfo
    summary: SummaryInfo
    leaves: list[LeafPrediction]
    visualizations: VisualizationPaths
    result_path: str
    message: str | None = None


class HistoryItem(BaseModel):
    id: str
    filename: str
    created_at: str
    processing_time_ms: float
    model: str
    summary: dict[str, Any]
    thumbnail: str | None = None
    total_leaves: int = 0


class HistoryListResponse(BaseModel):
    items: list[HistoryItem]
    total: int


class HistoryDetailResponse(PredictResponse):
    created_at: str
    filename: str


class DeleteResponse(BaseModel):
    id: str
    deleted: bool


class HealthResponse(BaseModel):
    status: str
    models_loaded: bool
    segmenter: str | None = None
    classifier: str | None = None
    device: str | None = None
    error: str | None = None
