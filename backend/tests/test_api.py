"""
API endpoint tests.
"""

import io
import pytest


class TestHealthEndpoint:
    def test_health_returns_ok(self, client):
        response = client.get("/api/v1/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert data["service"] == "roadsight-ai-api"
        assert "version" in data

    def test_root_returns_info(self, client):
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert "service" in data
        assert "docs" in data


class TestModelEndpoint:
    def test_model_info_returns_metadata(self, client):
        response = client.get("/api/v1/model")
        if response.status_code == 503:
            pytest.skip("Model not loaded in test environment")
        assert response.status_code == 200
        data = response.json()
        assert "name" in data
        assert "format" in data
        assert "input_size" in data
        assert "classes" in data
        assert isinstance(data["classes"], list)


class TestImagePrediction:
    def test_predict_image_valid(self, client, sample_image_bytes):
        response = client.post(
            "/api/v1/predict/image",
            files={"file": ("test.jpg", io.BytesIO(sample_image_bytes), "image/jpeg")},
        )
        if response.status_code == 503:
            pytest.skip("Model not loaded in test environment")
        assert response.status_code == 200
        data = response.json()
        assert "detections" in data
        assert "detection_count" in data
        assert "processing_time_ms" in data
        assert "fps" in data
        assert isinstance(data["detections"], list)
        assert data["processing_time_ms"] > 0

    def test_predict_image_empty_file(self, client):
        response = client.post(
            "/api/v1/predict/image",
            files={"file": ("test.jpg", io.BytesIO(b""), "image/jpeg")},
        )
        assert response.status_code == 400

    def test_predict_image_invalid_format(self, client):
        response = client.post(
            "/api/v1/predict/image",
            files={"file": ("test.exe", io.BytesIO(b"not an image"), "application/octet-stream")},
        )
        assert response.status_code in (400, 500)

    def test_predict_image_with_confidence(self, client, sample_image_bytes):
        response = client.post(
            "/api/v1/predict/image?confidence=0.8",
            files={"file": ("test.jpg", io.BytesIO(sample_image_bytes), "image/jpeg")},
        )
        if response.status_code == 503:
            pytest.skip("Model not loaded in test environment")
        assert response.status_code == 200


class TestMetrics:
    def test_metrics_returns_data(self, client):
        response = client.get("/api/v1/metrics")
        assert response.status_code == 200
        data = response.json()
        assert "total_predictions" in data
        assert "uptime_seconds" in data
        assert data["uptime_seconds"] >= 0


class TestSystemInfo:
    def test_system_info(self, client):
        response = client.get("/api/v1/system")
        if response.status_code == 503:
            pytest.skip("Model not loaded in test environment")
        assert response.status_code == 200
        data = response.json()
        assert "python_version" in data
        assert "opencv_version" in data
