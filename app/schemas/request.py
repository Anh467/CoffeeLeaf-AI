"""Request-side schemas for the CoffeeLeaf web API."""

from __future__ import annotations

from pydantic import BaseModel, Field


class PredictOptions(BaseModel):
    """Optional predict overrides; currently unused but reserved for clients."""

    allow_single_leaf_fallback: bool = Field(
        default=True,
        description="If the segmenter finds no leaves, classify the whole image.",
    )
