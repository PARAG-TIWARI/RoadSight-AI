"""
REST API routes for RoadSight AI.
All endpoints are under /api/v1.
"""

import logging
import os
import sys
import time
from datetime import datetime, timezone

import cv2
import numpy as np
from fastapi import APIRouter, File, UploadFile, HTTPException, Query
from fastapi.responses import FileResponse

from app.core.config import settings
from app.core.model import model_manager
from app.services.image_inference import run_image_inference
from app.services.video_inference import run_video_inference
from app.services.benchmark import run_benchmark
from app.schemas.responses import (
    HealthResponse,
    ModelInfoResponse,
    ImagePredictionResponse,
    VideoPredictionResponse,
    BenchmarkResponse,
    MetricsResponse,
    SystemInfoResponse,
)

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1", tags=["RoadSight AI API v1"])

# --- Runtime counters (in-memory, session-based) ---
_start_time = time.time()
_total_predictions = 0
_total_images = 0
_total_videos = 0
_total_inference_ms = 0.0


# ============================================================
# GET /api/v1/health
# ============================================================
@router.get("/health", response_model=HealthResponse)
async def health_check():
    """Check API health and model status."""
    return HealthResponse(
        status="ok",
        service="roadsight-ai-api",
        model_loaded=model_manager.is_loaded,
        version=settings.API_VERSION,
    )


# ============================================================
# GET /api/v1/model
# ============================================================
@router.get("/model", response_model=ModelInfoResponse)
async def get_model_info():
    """Get information about the loaded model."""
    if not model_manager.is_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded")

    metadata = model_manager.get_metadata()
    return ModelInfoResponse(
        name=metadata.name,
        version=metadata.version,
        format=metadata.format,
        input_size=metadata.input_size,
        classes=metadata.classes,
        backend=metadata.backend,
        provider=metadata.provider,
        load_time_ms=round(model_manager.load_time_ms, 2),
    )


# ============================================================
# POST /api/v1/predict/image
# ============================================================
@router.post("/predict/image", response_model=ImagePredictionResponse)
async def predict_image(
    file: UploadFile = File(...),
    confidence: float = Query(default=None, ge=0.0, le=1.0, description="Confidence threshold"),
):
    """
    Run pothole detection and segmentation on a single image.

    Accepts JPEG, PNG, BMP, or WebP images.
    Returns detections, masks, visualizations, and metrics.
    """
    global _total_predictions, _total_images, _total_inference_ms

    try:
        image_bytes = await file.read()

        result = run_image_inference(
            image_bytes=image_bytes,
            filename=file.filename or "",
            confidence_threshold=confidence,
        )

        _total_predictions += 1
        _total_images += 1
        _total_inference_ms += result.processing_time_ms

        return result

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except RuntimeError as e:
        if "Model not loaded" in str(e):
            raise HTTPException(status_code=503, detail=str(e))
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")
    except Exception as e:
        logger.error("Image prediction failed: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


# ============================================================
# POST /api/v1/predict/video
# ============================================================
@router.post("/predict/video", response_model=VideoPredictionResponse)
async def predict_video(
    file: UploadFile = File(...),
    confidence: float = Query(default=None, ge=0.0, le=1.0, description="Confidence threshold"),
):
    """
    Run pothole detection on a video file, frame by frame.

    Accepts MP4, AVI, MOV, or MKV videos.
    Returns per-frame results, temporal events, and aggregate metrics.
    """
    global _total_predictions, _total_videos, _total_inference_ms

    try:
        video_bytes = await file.read()

        # Validate
        from app.utils.preprocessing import validate_video_bytes
        validate_video_bytes(video_bytes, file.filename or "")

        result = run_video_inference(
            video_bytes=video_bytes,
            confidence_threshold=confidence,
        )

        _total_predictions += 1
        _total_videos += 1
        _total_inference_ms += result.total_processing_time_sec * 1000

        return result

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except RuntimeError as e:
        if "Model not loaded" in str(e):
            raise HTTPException(status_code=503, detail=str(e))
        raise HTTPException(status_code=500, detail=f"Video processing failed: {str(e)}")
    except Exception as e:
        logger.error("Video prediction failed: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Video processing failed: {str(e)}")


# ============================================================
# GET /api/v1/predict/video/{video_id}/download
# ============================================================
@router.get("/predict/video/{video_id}/download")
async def download_processed_video(video_id: str):
    """Download the processed (annotated) video file."""
    # Sanitize video_id to prevent path traversal
    safe_id = "".join(c for c in video_id if c.isalnum() or c in "-_")
    video_path = settings.TEMP_DIR / f"output_{safe_id}.mp4"

    if not video_path.exists():
        raise HTTPException(status_code=404, detail="Processed video not found or has expired")

    return FileResponse(
        path=str(video_path),
        media_type="video/mp4",
        filename=f"roadsight_processed_{safe_id}.mp4",
    )


# ============================================================
# GET /api/v1/benchmark
# ============================================================
@router.get("/benchmark", response_model=BenchmarkResponse)
async def get_benchmark(
    iterations: int = Query(default=10, ge=1, le=100, description="Number of benchmark iterations"),
):
    """
    Run inference benchmarks comparing available backends.

    Measures real latency, FPS, CPU, and memory usage.
    Warning: This endpoint blocks while benchmarking. Use reasonable iteration counts.
    """
    try:
        return run_benchmark(iterations=iterations)
    except Exception as e:
        logger.error("Benchmark failed: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Benchmark failed: {str(e)}")


# ============================================================
# GET /api/v1/metrics
# ============================================================
@router.get("/metrics", response_model=MetricsResponse)
async def get_metrics():
    """Get runtime application metrics (session-based, in-memory)."""
    uptime = time.time() - _start_time
    avg_inference = _total_inference_ms / _total_predictions if _total_predictions > 0 else 0.0

    return MetricsResponse(
        total_predictions=_total_predictions,
        total_images=_total_images,
        total_videos=_total_videos,
        average_inference_ms=round(avg_inference, 2),
        uptime_seconds=round(uptime, 2),
    )


# ============================================================
# GET /api/v1/system
# ============================================================
@router.get("/system", response_model=SystemInfoResponse)
async def get_system_info():
    """Get system and dependency version information."""
    if not model_manager.is_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded")

    metadata = model_manager.get_metadata()

    return SystemInfoResponse(
        python_version=sys.version.split()[0],
        opencv_version=cv2.__version__,
        numpy_version=np.__version__,
        api_version=settings.API_VERSION,
        backend_version="1.0.0",
        model_info=ModelInfoResponse(
            name=metadata.name,
            version=metadata.version,
            format=metadata.format,
            input_size=metadata.input_size,
            classes=metadata.classes,
            backend=metadata.backend,
            provider=metadata.provider,
            load_time_ms=round(model_manager.load_time_ms, 2),
        ),
    )
