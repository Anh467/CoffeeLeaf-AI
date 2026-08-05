"""CoffeeLeaf-AI FastAPI entrypoint."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from app.api.routes import router
from app.services.inference_service import InferenceService
from pipeline import ROOT

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
LOGGER = logging.getLogger("coffeeleaf.web")


@asynccontextmanager
async def lifespan(app: FastAPI):
    history_root = ROOT / "runs" / "history"
    history_root.mkdir(parents=True, exist_ok=True)
    service = InferenceService(bundle_dir=ROOT / "deployment", history_root=history_root)
    service.startup()
    app.state.inference = service
    LOGGER.info("CoffeeLeaf web app ready (models_loaded=%s)", service.models_loaded)
    try:
        yield
    finally:
        service.shutdown()


app = FastAPI(
    title="CoffeeLeaf-AI API",
    description="REST API and dashboard for coffee-leaf disease inference.",
    version="1.0.0",
    lifespan=lifespan,
)

static_dir = ROOT / "app" / "static"
history_dir = ROOT / "runs" / "history"
static_dir.mkdir(parents=True, exist_ok=True)
history_dir.mkdir(parents=True, exist_ok=True)
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")
app.mount(
    "/media/history",
    StaticFiles(directory=str(history_dir)),
    name="history_media",
)
app.include_router(router)


def create_app() -> FastAPI:
    """Factory used by tests or alternative ASGI loaders."""
    return app
