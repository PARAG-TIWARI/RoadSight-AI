"""
Pydantic response models for all API endpoints.
"""

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str = "ok"
    service: str = "roadsight-ai-api"
    model_loaded: bool
    version: str


class ModelInfoResponse(BaseModel):
    name: str
    version: str
    format: str
    input_size: int
    classes: list[str]
    backend: str
    provider: str
    load_time_ms: float


class DetectionResult(BaseModel):
    """A single detection within an image."""
    bbox: list[float] = Field(description="Bounding box [x1, y1, x2, y2]")
    confidence: float = Field(ge=0.0, le=1.0)
    class_name: str = "pothole"
    class_id: int = 0
    area_pixels: int = Field(default=0, description="Mask area in pixels")
    area_ratio: float = Field(default=0.0, description="Mask area / image area")
    severity: str = Field(default="unknown", description="Severity category based on area")


class ImagePredictionResponse(BaseModel):
    """Response for single image inference."""
    detections: list[DetectionResult]
    detection_count: int
    average_confidence: float
    processing_time_ms: float
    fps: float
    image_width: int
    image_height: int
    model_backend: str

    # Base64-encoded visualization images
    annotated_image: str | None = Field(
        default=None, description="Image with bounding boxes drawn, base64 PNG"
    )
    mask_overlay: str | None = Field(
        default=None, description="Image with segmentation mask overlay, base64 PNG"
    )
    mask_image: str | None = Field(
        default=None, description="Binary segmentation mask, base64 PNG"
    )

    # Aggregate metrics
    total_area_pixels: int = 0
    total_area_ratio: float = 0.0


class VideoFrameResult(BaseModel):
    """Per-frame detection summary in video processing."""
    frame_index: int
    timestamp_sec: float
    detection_count: int
    average_confidence: float
    processing_time_ms: float


class EventEntry(BaseModel):
    """A temporal event detected during video analysis."""
    frame_index: int
    timestamp_sec: float
    event_type: str = Field(description="Event type: new_detection, persistent, high_confidence, large_area, lost")
    description: str
    confidence: float = 0.0
    area_ratio: float = 0.0


class VideoPredictionResponse(BaseModel):
    """Response for video inference."""
    total_frames: int
    processed_frames: int
    detections_total: int
    average_confidence: float
    processing_fps: float
    average_latency_ms: float
    total_processing_time_sec: float
    video_duration_sec: float
    video_fps: float
    model_backend: str

    # Frame-level results
    frame_results: list[VideoFrameResult]

    # Temporal events
    events: list[EventEntry]

    # Processed video download path (relative)
    processed_video_id: str | None = None


class BenchmarkResult(BaseModel):
    """Result from a single benchmark run."""
    backend: str
    iterations: int
    avg_latency_ms: float
    p50_latency_ms: float = 0.0
    p90_latency_ms: float = 0.0
    p99_latency_ms: float = 0.0
    min_latency_ms: float
    max_latency_ms: float
    fps: float
    cpu_percent: float
    memory_mb: float
    input_resolution: str
    model_format: str


class BenchmarkResponse(BaseModel):
    """Full benchmark comparison response."""
    results: list[BenchmarkResult]
    system_info: dict
    timestamp: str


class MetricsResponse(BaseModel):
    """Runtime metrics for the application."""
    total_predictions: int
    total_images: int
    total_videos: int
    average_inference_ms: float
    uptime_seconds: float


class SystemInfoResponse(BaseModel):
    """System and dependency version information."""
    python_version: str
    opencv_version: str
    numpy_version: str
    api_version: str
    backend_version: str
    model_info: ModelInfoResponse
