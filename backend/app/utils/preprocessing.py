"""
Image preprocessing utilities for the inference pipeline.
"""

import cv2
import numpy as np

from app.core.config import settings


def decode_image(image_bytes: bytes) -> np.ndarray:
    """
    Decode raw bytes into a BGR uint8 image (OpenCV format).
    Raises ValueError if decoding fails.
    """
    buf = np.frombuffer(image_bytes, dtype=np.uint8)
    img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Failed to decode image. File may be corrupted or unsupported.")
    return img


def preprocess_for_model(
    image_bgr: np.ndarray,
    input_size: int | None = None,
) -> np.ndarray:
    """
    Preprocess a BGR image for YOLOv8 inference.

    Steps:
    1. Resize to input_size x input_size
    2. BGR -> RGB
    3. Normalize to [0, 1]
    4. HWC -> CHW
    5. Add batch dimension

    Returns: float32 tensor of shape [1, 3, input_size, input_size]
    """
    size = input_size or settings.INPUT_SIZE

    resized = cv2.resize(image_bgr, (size, size), interpolation=cv2.INTER_LINEAR)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    normalized = rgb.astype(np.float32) / 255.0
    transposed = normalized.transpose(2, 0, 1)  # HWC -> CHW
    batched = transposed[np.newaxis, ...]  # Add batch dim

    return batched


def validate_image_bytes(image_bytes: bytes, filename: str = "") -> None:
    """
    Validate uploaded image bytes.
    Raises ValueError with descriptive messages on failure.
    """
    if not image_bytes:
        raise ValueError("Empty file received.")

    if len(image_bytes) > settings.max_upload_bytes:
        raise ValueError(
            f"File too large: {len(image_bytes) / (1024*1024):.1f}MB. "
            f"Maximum: {settings.MAX_UPLOAD_SIZE_MB}MB."
        )

    if filename:
        import os
        ext = os.path.splitext(filename)[1].lower()
        if ext and ext not in settings.ALLOWED_IMAGE_EXTENSIONS:
            raise ValueError(
                f"Unsupported image format: '{ext}'. "
                f"Allowed: {', '.join(settings.ALLOWED_IMAGE_EXTENSIONS)}"
            )


def validate_video_bytes(video_bytes: bytes, filename: str = "") -> None:
    """Validate uploaded video file."""
    if not video_bytes:
        raise ValueError("Empty file received.")

    if len(video_bytes) > settings.max_upload_bytes:
        raise ValueError(
            f"File too large: {len(video_bytes) / (1024*1024):.1f}MB. "
            f"Maximum: {settings.MAX_UPLOAD_SIZE_MB}MB."
        )

    if filename:
        import os
        ext = os.path.splitext(filename)[1].lower()
        if ext and ext not in settings.ALLOWED_VIDEO_EXTENSIONS:
            raise ValueError(
                f"Unsupported video format: '{ext}'. "
                f"Allowed: {', '.join(settings.ALLOWED_VIDEO_EXTENSIONS)}"
            )
