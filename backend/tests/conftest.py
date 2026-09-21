"""
Pytest fixtures for backend tests.
"""

import io
import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.main import app


@pytest.fixture(scope="session")
def client():
    """FastAPI test client with active lifespan."""
    from app.core.model import model_manager
    if not model_manager.is_loaded:
        try:
            model_manager.load()
        except Exception:
            pass
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def sample_image_bytes() -> bytes:
    """Generate a simple test image as JPEG bytes."""
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    # Add some features to make it non-trivial
    cv2.rectangle(img, (100, 100), (300, 300), (128, 128, 128), -1)
    cv2.circle(img, (400, 200), 50, (200, 200, 200), -1)
    _, buf = cv2.imencode(".jpg", img)
    return buf.tobytes()


@pytest.fixture
def sample_png_bytes() -> bytes:
    """Generate a simple test image as PNG bytes."""
    img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    _, buf = cv2.imencode(".png", img)
    return buf.tobytes()
