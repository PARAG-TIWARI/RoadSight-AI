"""
Model loader supporting ONNX Runtime and PyTorch (ultralytics) backends.
Selected via MODEL_FORMAT environment variable.
"""

import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from app.core.config import settings

logger = logging.getLogger(__name__)


@dataclass
class ModelMetadata:
    """Describes the loaded model's properties."""
    name: str = "YOLOv8-seg Pothole"
    version: str = "1.0.0"
    format: str = "onnx"
    input_size: int = 640
    classes: list[str] = field(default_factory=lambda: ["pothole"])
    backend: str = "ONNX Runtime"
    provider: str = "CPUExecutionProvider"


class BaseModelBackend(ABC):
    """Abstract interface for model inference backends."""

    @abstractmethod
    def predict(self, input_tensor: np.ndarray) -> list[np.ndarray]:
        """Run inference and return raw outputs."""
        ...

    @abstractmethod
    def get_metadata(self) -> ModelMetadata:
        """Return model metadata."""
        ...


class ONNXBackend(BaseModelBackend):
    """ONNX Runtime inference backend."""

    def __init__(self, model_path: Path):
        import onnxruntime as ort

        available = ort.get_available_providers()
        providers = []
        if "CUDAExecutionProvider" in available:
            providers.append("CUDAExecutionProvider")
        providers.append("CPUExecutionProvider")

        self.session = ort.InferenceSession(str(model_path), providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        self.input_shape = self.session.get_inputs()[0].shape

        active_providers = self.session.get_providers()
        self._provider = active_providers[0] if active_providers else "CPUExecutionProvider"

        logger.info(
            "ONNX model loaded: %s | Provider: %s", model_path.name, self._provider
        )

    def predict(self, input_tensor: np.ndarray) -> list[np.ndarray]:
        return self.session.run(None, {self.input_name: input_tensor})

    def get_metadata(self) -> ModelMetadata:
        return ModelMetadata(
            format="onnx",
            input_size=settings.INPUT_SIZE,
            classes=settings.CLASS_NAMES,
            backend="ONNX Runtime",
            provider=self._provider,
        )


class PyTorchBackend(BaseModelBackend):
    """Ultralytics YOLO PyTorch inference backend."""

    def __init__(self, model_path: Path):
        from ultralytics import YOLO
        import torch

        self.model = YOLO(str(model_path))
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info("PyTorch model loaded: %s | Device: %s", model_path.name, self._device)

    def predict(self, input_tensor: np.ndarray) -> list[np.ndarray]:
        """Run ultralytics prediction. Returns raw numpy outputs for pipeline compat."""
        import cv2

        # Ultralytics expects BGR HWC uint8 image
        img = input_tensor.squeeze(0).transpose(1, 2, 0)  # CHW -> HWC
        img = (img * 255).clip(0, 255).astype(np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        results = self.model.predict(img, verbose=False, conf=settings.CONFIDENCE_THRESHOLD)
        return results

    def get_metadata(self) -> ModelMetadata:
        return ModelMetadata(
            format="pytorch",
            input_size=settings.INPUT_SIZE,
            classes=settings.CLASS_NAMES,
            backend="PyTorch (Ultralytics)",
            provider=self._device,
        )


class ModelManager:
    """
    Singleton-style model manager.
    Loads the appropriate backend based on configuration.
    """

    def __init__(self):
        self._backend: BaseModelBackend | None = None
        self._load_time_ms: float = 0.0

    def load(self) -> None:
        """Load the model. Called once at application startup."""
        model_path = settings.full_model_path

        if not model_path.exists():
            raise FileNotFoundError(
                f"Model file not found: {model_path}. "
                f"Ensure MODEL_PATH is set correctly in .env"
            )

        start = time.perf_counter()

        if settings.MODEL_FORMAT == "pytorch":
            pt_path = model_path.with_suffix(".pt")
            if not pt_path.exists():
                pt_path = model_path
            self._backend = PyTorchBackend(pt_path)
        else:
            self._backend = ONNXBackend(model_path)

        self._load_time_ms = (time.perf_counter() - start) * 1000
        logger.info("Model loaded in %.1f ms", self._load_time_ms)

    @property
    def backend(self) -> BaseModelBackend:
        if self._backend is None:
            raise RuntimeError("Model not loaded. Call model_manager.load() first.")
        return self._backend

    @property
    def is_loaded(self) -> bool:
        return self._backend is not None

    @property
    def load_time_ms(self) -> float:
        return self._load_time_ms

    def predict(self, input_tensor: np.ndarray) -> Any:
        return self.backend.predict(input_tensor)

    def get_metadata(self) -> ModelMetadata:
        return self.backend.get_metadata()


# Global model manager instance
model_manager = ModelManager()
