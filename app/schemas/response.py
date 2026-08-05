"""Response schemas for the CoffeeLeaf web API."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class ImageInfo(BaseModel):
    width: int
    height: int
    file_size: int
    filename: str
    mode: str = Field(description="tree | single_leaf")


class ProcessingInfo(BaseModel):
    preprocess_time_ms: float
    segmentation_time_ms: float
    classification_time_ms: float
    total_time_ms: float


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


class LeafPrediction(BaseModel):
    leaf_id: int
    bbox: list[int]
    mask: str | None = None
    crop: str | None = None
    prediction: str
    labels: list[str]
    confidence: float
    detector_confidence: float
    accepted: bool
    probabilities: dict[str, float]


class VisualizationPaths(BaseModel):
    original: str
    overlay: str
    mask_overlay: str
    boxes: str
    thumbnail: str


class PredictResponse(BaseModel):
    id: str
    image: ImageInfo
    processing: ProcessingInfo
    model: ModelInfo
    summary: SummaryInfo
    leaves: list[LeafPrediction]
    visualizations: VisualizationPaths
    result_path: str


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
