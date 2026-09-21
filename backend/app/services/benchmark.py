"""
Benchmark service: measures inference performance with real timing.
Compares ONNX Runtime and PyTorch (if available) backends.
"""

import logging
import platform
import sys
import time

import cv2
import numpy as np
import psutil

from app.core.config import settings
from app.utils.preprocessing import preprocess_for_model
from app.schemas.responses import BenchmarkResult, BenchmarkResponse

logger = logging.getLogger(__name__)


def _create_test_image(size: int = 640) -> np.ndarray:
    """Create a realistic-looking test image for benchmarking."""
    # Use a gradient + noise pattern (more realistic than blank)
    img = np.zeros((size, size, 3), dtype=np.uint8)
    for c in range(3):
        img[:, :, c] = np.random.randint(40, 200, (size, size), dtype=np.uint8)
    # Add some structure
    cv2.rectangle(img, (100, 100), (400, 400), (128, 128, 128), -1)
    cv2.circle(img, (320, 320), 80, (80, 80, 80), -1)
    return img


def run_benchmark(
    iterations: int = 20,
    warmup: int = 3,
) -> BenchmarkResponse:
    """
    Run inference benchmarks and return measured results.

    Performs warmup runs, then timed iterations.
    Measures latency, FPS, CPU, and memory usage.
    Only benchmarks backends that are genuinely available.
    """
    results: list[BenchmarkResult] = []
    test_image = _create_test_image(settings.INPUT_SIZE)
    input_tensor = preprocess_for_model(test_image)

    # --- ONNX Runtime Benchmark ---
    try:
        import onnxruntime as ort

        onnx_path = settings.MODEL_DIR / "best.onnx"
        if onnx_path.exists():
            available = ort.get_available_providers()
            providers = []
            if "CUDAExecutionProvider" in available:
                providers.append("CUDAExecutionProvider")
            providers.append("CPUExecutionProvider")

            session = ort.InferenceSession(str(onnx_path), providers=providers)
            input_name = session.get_inputs()[0].name
            active_provider = session.get_providers()[0]

            # Warmup
            for _ in range(warmup):
                session.run(None, {input_name: input_tensor})

            # Timed runs
            latencies = []
            cpu_samples = []
            mem_samples = []
            process = psutil.Process()

            for _ in range(iterations):
                cpu_before = process.cpu_percent()
                mem_before = process.memory_info().rss / (1024 * 1024)

                t0 = time.perf_counter()
                session.run(None, {input_name: input_tensor})
                t1 = time.perf_counter()

                latencies.append((t1 - t0) * 1000)
                cpu_samples.append(process.cpu_percent())
                mem_samples.append(process.memory_info().rss / (1024 * 1024))

            results.append(BenchmarkResult(
                backend=f"ONNX Runtime ({active_provider})",
                iterations=iterations,
                avg_latency_ms=round(float(np.mean(latencies)), 2),
                p50_latency_ms=round(float(np.percentile(latencies, 50)), 2),
                p90_latency_ms=round(float(np.percentile(latencies, 90)), 2),
                p99_latency_ms=round(float(np.percentile(latencies, 99)), 2),
                min_latency_ms=round(float(np.min(latencies)), 2),
                max_latency_ms=round(float(np.max(latencies)), 2),
                fps=round(1000.0 / float(np.mean(latencies)), 2),
                cpu_percent=round(float(np.mean(cpu_samples)), 1),
                memory_mb=round(float(np.mean(mem_samples)), 1),
                input_resolution=f"{settings.INPUT_SIZE}x{settings.INPUT_SIZE}",
                model_format="ONNX",
            ))

            del session

    except ImportError:
        logger.info("ONNX Runtime not available — skipping benchmark")
    except Exception as e:
        logger.warning("ONNX benchmark failed: %s", e)

    # --- PyTorch Benchmark ---
    try:
        from ultralytics import YOLO
        import torch

        pt_path = settings.MODEL_DIR / "best.pt"
        if pt_path.exists():
            model = YOLO(str(pt_path))
            device = "cuda" if torch.cuda.is_available() else "cpu"

            # Ultralytics expects BGR HWC uint8
            test_bgr = test_image.copy()

            # Warmup
            for _ in range(warmup):
                model.predict(test_bgr, verbose=False)

            # Timed runs
            latencies = []
            cpu_samples = []
            mem_samples = []
            process = psutil.Process()

            for _ in range(iterations):
                t0 = time.perf_counter()
                model.predict(test_bgr, verbose=False)
                t1 = time.perf_counter()

                latencies.append((t1 - t0) * 1000)
                cpu_samples.append(process.cpu_percent())
                mem_samples.append(process.memory_info().rss / (1024 * 1024))

            results.append(BenchmarkResult(
                backend=f"PyTorch ({device})",
                iterations=iterations,
                avg_latency_ms=round(float(np.mean(latencies)), 2),
                p50_latency_ms=round(float(np.percentile(latencies, 50)), 2),
                p90_latency_ms=round(float(np.percentile(latencies, 90)), 2),
                p99_latency_ms=round(float(np.percentile(latencies, 99)), 2),
                min_latency_ms=round(float(np.min(latencies)), 2),
                max_latency_ms=round(float(np.max(latencies)), 2),
                fps=round(1000.0 / float(np.mean(latencies)), 2),
                cpu_percent=round(float(np.mean(cpu_samples)), 1),
                memory_mb=round(float(np.mean(mem_samples)), 1),
                input_resolution=f"{settings.INPUT_SIZE}x{settings.INPUT_SIZE}",
                model_format="PyTorch",
            ))

            del model

    except ImportError:
        logger.info("PyTorch/Ultralytics not available — skipping benchmark")
    except Exception as e:
        logger.warning("PyTorch benchmark failed: %s", e)

    # --- System Info ---
    system_info = {
        "platform": platform.platform(),
        "processor": platform.processor() or "Unknown",
        "python_version": sys.version.split()[0],
        "cpu_count": psutil.cpu_count(logical=True),
        "total_ram_gb": round(psutil.virtual_memory().total / (1024**3), 1),
        "opencv_version": cv2.__version__,
        "numpy_version": np.__version__,
    }

    try:
        import onnxruntime as ort
        system_info["onnxruntime_version"] = ort.__version__
    except ImportError:
        pass

    try:
        import torch
        system_info["pytorch_version"] = torch.__version__
        system_info["cuda_available"] = torch.cuda.is_available()
    except ImportError:
        pass

    from datetime import datetime, timezone
    timestamp = datetime.now(timezone.utc).isoformat()

    return BenchmarkResponse(
        results=results,
        system_info=system_info,
        timestamp=timestamp,
    )
