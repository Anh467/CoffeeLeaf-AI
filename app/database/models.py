"""Database model helpers (kept thin; SQLite rows are plain dicts)."""

from __future__ import annotations

from typing import TypedDict


class PredictionRecord(TypedDict):
    id: str
    filename: str
    created_at: str
    image_path: str
    result_path: str
    processing_time: float
    model: str
    summary: dict
