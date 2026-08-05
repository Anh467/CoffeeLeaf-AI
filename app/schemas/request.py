"""Request-side schemas for the CoffeeLeaf web API."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

PredictMode = Literal["auto", "whole_image", "single_leaf"]


class PredictOptions(BaseModel):
    """Optional predict overrides for multipart form clients."""

    mode: PredictMode = Field(
        default="auto",
        description=(
            "auto: segment then fallback to single_leaf; "
            "whole_image: segment only; "
            "single_leaf: classify whole image as one leaf."
        ),
    )
    allow_single_leaf_fallback: bool | None = Field(
        default=None,
        description=(
            "Legacy flag. True maps to mode=auto, False to mode=whole_image "
            "when mode is omitted."
        ),
    )
