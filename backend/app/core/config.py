"""
Application configuration using Pydantic Settings.
All configurable values are loaded from environment variables.
"""

import os
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # --- Paths ---
    PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent.parent
    MODEL_DIR: Path | None = None
    MODEL_PATH: str = Field(default="best.onnx", description="Model filename inside MODEL_DIR")
    MODEL_FORMAT: str = Field(default="onnx", description="Model format: 'onnx' or 'pytorch'")

    # --- Inference ---
    CONFIDENCE_THRESHOLD: float = Field(default=0.4, ge=0.0, le=1.0)
    NMS_IOU_THRESHOLD: float = Field(default=0.5, ge=0.0, le=1.0)
    INPUT_SIZE: int = Field(default=640, description="Model input resolution (square)")
    NUM_CLASSES: int = Field(default=1, description="Number of detection classes")
    CLASS_NAMES: list[str] = Field(default=["pothole"])

    # --- Event Processing ---
    MIN_AREA_RATIO: float = Field(default=0.005, description="Minimum area ratio to consider a valid detection")
    MIN_AREA_PIXELS: int = Field(default=300, description="Minimum mask area in pixels")
    PERSISTENCE_FRAMES: int = Field(default=3, description="Frames a detection must persist to be confirmed")
    HIGH_CONFIDENCE_THRESHOLD: float = Field(default=0.75, description="Threshold for high-confidence event")
    LARGE_AREA_RATIO: float = Field(default=0.05, description="Threshold for large-area event")

    # --- Video ---
    MAX_VIDEO_FRAMES: int = Field(default=500, description="Maximum frames to process from a video")
    VIDEO_SKIP_FRAMES: int = Field(default=1, description="Process every Nth frame (1 = all frames)")

    # --- Upload ---
    MAX_UPLOAD_SIZE_MB: int = Field(default=50, description="Maximum upload size in megabytes")
    ALLOWED_IMAGE_EXTENSIONS: list[str] = Field(default=[".jpg", ".jpeg", ".png", ".bmp", ".webp"])
    ALLOWED_VIDEO_EXTENSIONS: list[str] = Field(default=[".mp4", ".avi", ".mov", ".mkv"])

    # --- Server ---
    CORS_ORIGINS: str = Field(default="http://localhost:5173,http://localhost:3000")
    API_VERSION: str = Field(default="1.0.0")
    APP_NAME: str = Field(default="RoadSight AI API")

    # --- Temp ---
    TEMP_DIR: Path | None = None

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    def model_post_init(self, __context) -> None:
        if self.MODEL_DIR is None:
            self.MODEL_DIR = self.PROJECT_ROOT / "models"
        if self.TEMP_DIR is None:
            self.TEMP_DIR = self.PROJECT_ROOT / "tmp"
        self.TEMP_DIR.mkdir(parents=True, exist_ok=True)

    @property
    def full_model_path(self) -> Path:
        return self.MODEL_DIR / self.MODEL_PATH

    @property
    def cors_origins_list(self) -> list[str]:
        return [o.strip() for o in self.CORS_ORIGINS.split(",") if o.strip()]

    @property
    def max_upload_bytes(self) -> int:
        return self.MAX_UPLOAD_SIZE_MB * 1024 * 1024


settings = Settings()
