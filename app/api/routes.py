"""FastAPI routes for dashboard and REST API."""

from __future__ import annotations

import logging
from typing import Annotated

from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from app.schemas.response import (
    DeleteResponse,
    HealthResponse,
    HistoryDetailResponse,
    HistoryListResponse,
    PredictResponse,
)
from app.services.inference_service import InferenceService
from pipeline import ROOT

LOGGER = logging.getLogger(__name__)

router = APIRouter()
templates = Jinja2Templates(directory=str(ROOT / "app" / "templates"))


def get_inference_service(request: Request) -> InferenceService:
    service = getattr(request.app.state, "inference", None)
    if service is None:
        raise HTTPException(status_code=503, detail="Inference service is unavailable")
    return service


InferenceDep = Annotated[InferenceService, Depends(get_inference_service)]


@router.get("/", response_class=HTMLResponse)
async def dashboard(request: Request, service: InferenceDep) -> HTMLResponse:
    return templates.TemplateResponse(
        request,
        "index.html",
        {
            "title": "Coffee Leaf Disease Dashboard",
            "health": service.health(),
        },
    )


@router.get("/history-page", response_class=HTMLResponse)
async def history_page(request: Request, service: InferenceDep) -> HTMLResponse:
    history = service.list_history(limit=100)
    return templates.TemplateResponse(
        request,
        "history.html",
        {
            "title": "Prediction History",
            "history_items": history["items"],
            "history_total": history["total"],
            "health": service.health(),
        },
    )


@router.get("/health", response_model=HealthResponse)
async def health(service: InferenceDep) -> HealthResponse:
    return HealthResponse(**service.health())


@router.post("/predict", response_model=PredictResponse)
async def predict(
    service: InferenceDep,
    image: UploadFile = File(...),
    mode: str = Form(default="auto"),
    allow_single_leaf_fallback: bool | None = Form(default=None),
) -> PredictResponse:
    if not image.filename:
        raise HTTPException(status_code=400, detail="Uploaded file must have a filename")
    try:
        content = await image.read()
        if not content:
            raise HTTPException(status_code=400, detail="Uploaded image is empty")
        result = service.predict_upload(
            image_bytes=content,
            filename=image.filename,
            mode=mode,
            allow_single_leaf_fallback=allow_single_leaf_fallback,
        )
        return PredictResponse(**result)
    except HTTPException:
        raise
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    except RuntimeError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error
    except Exception as error:  # noqa: BLE001
        LOGGER.exception("Prediction failed")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {error}") from error


@router.get("/history", response_model=HistoryListResponse)
async def list_history(
    service: InferenceDep,
    limit: int = 50,
    offset: int = 0,
) -> HistoryListResponse:
    payload = service.list_history(limit=max(1, min(limit, 200)), offset=max(0, offset))
    return HistoryListResponse(**payload)


@router.get("/history/{prediction_id}", response_model=HistoryDetailResponse)
async def get_history(prediction_id: str, service: InferenceDep) -> HistoryDetailResponse:
    try:
        payload = service.get_history(prediction_id)
        return HistoryDetailResponse(**payload)
    except KeyError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    except FileNotFoundError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error


@router.delete("/history/{prediction_id}", response_model=DeleteResponse)
async def delete_history(prediction_id: str, service: InferenceDep) -> DeleteResponse:
    deleted = service.delete_history(prediction_id)
    if not deleted:
        raise HTTPException(status_code=404, detail=f"Prediction not found: {prediction_id}")
    return DeleteResponse(id=prediction_id, deleted=True)
