"""
RoadSight AI — FastAPI Application Entry Point.

Intelligent Road Defect Detection & Video Analytics.
"""

import logging
import sys
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.core.config import settings
from app.core.model import model_manager
from app.api.routes import router

# --- Logging ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("roadsight_ai")


# --- Lifespan (model loading at startup) ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model at startup, clean up at shutdown."""
    logger.info("Starting RoadSight AI API v%s", settings.API_VERSION)
    logger.info("Model path: %s", settings.full_model_path)
    logger.info("Model format: %s", settings.MODEL_FORMAT)

    try:
        model_manager.load()
        logger.info("Model loaded successfully")
    except FileNotFoundError as e:
        logger.error("Model file not found: %s", e)
        logger.error("Please ensure the model file exists at the configured path")
    except Exception as e:
        logger.error("Failed to load model: %s", e)

    yield

    # Cleanup
    logger.info("Shutting down RoadSight AI API")


# --- FastAPI App ---
app = FastAPI(
    title="RoadSight AI API",
    description=(
        "Intelligent Road Defect Detection & Video Analytics. "
        "Detect, segment, and analyze road defects using YOLOv8 computer vision."
    ),
    version=settings.API_VERSION,
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
)

# --- CORS ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins_list,
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

# --- Routes ---
app.include_router(router)


# --- Root redirect ---
@app.get("/", tags=["Root"])
async def root():
    """API root — redirects to documentation."""
    return {
        "service": "RoadSight AI API",
        "version": settings.API_VERSION,
        "docs": "/docs",
        "health": "/api/v1/health",
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
    )
