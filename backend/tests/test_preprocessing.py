"""
Tests for preprocessing utilities.
"""

import cv2
import numpy as np
import pytest

from app.utils.preprocessing import (
    decode_image,
    preprocess_for_model,
    validate_image_bytes,
    validate_video_bytes,
)


class TestDecodeImage:
    def test_decode_valid_jpeg(self, sample_image_bytes):
        img = decode_image(sample_image_bytes)
        assert img is not None
        assert img.ndim == 3
        assert img.shape[2] == 3

    def test_decode_valid_png(self, sample_png_bytes):
        img = decode_image(sample_png_bytes)
        assert img is not None
        assert img.ndim == 3

    def test_decode_invalid_bytes(self):
        with pytest.raises(ValueError, match="Failed to decode"):
            decode_image(b"not an image")


class TestPreprocessForModel:
    def test_output_shape(self):
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        tensor = preprocess_for_model(img, input_size=640)
        assert tensor.shape == (1, 3, 640, 640)
        assert tensor.dtype == np.float32

    def test_output_range(self):
        img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        tensor = preprocess_for_model(img)
        assert tensor.min() >= 0.0
        assert tensor.max() <= 1.0

    def test_custom_size(self):
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        tensor = preprocess_for_model(img, input_size=320)
        assert tensor.shape == (1, 3, 320, 320)


class TestValidation:
    def test_empty_image(self):
        with pytest.raises(ValueError, match="Empty"):
            validate_image_bytes(b"")

    def test_unsupported_extension(self):
        with pytest.raises(ValueError, match="Unsupported"):
            validate_image_bytes(b"data", filename="test.exe")

    def test_valid_extensions(self):
        # Should not raise
        validate_image_bytes(b"data", filename="test.jpg")
        validate_image_bytes(b"data", filename="test.png")

    def test_empty_video(self):
        with pytest.raises(ValueError, match="Empty"):
            validate_video_bytes(b"")

    def test_video_unsupported_ext(self):
        with pytest.raises(ValueError, match="Unsupported"):
            validate_video_bytes(b"data", filename="test.txt")
