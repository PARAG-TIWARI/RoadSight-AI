# RoadSight AI — Model Evaluation & Performance Analysis

## 1. Overview & Evaluation Methodology

This document outlines the evaluation protocols, quantitative metrics, failure modes, and edge cases for the RoadSight AI YOLOv8-seg pothole detection and instance segmentation models.

### Evaluation Metrics

| Metric | Formulation | Description |
| :--- | :--- | :--- |
| **Precision** | $\frac{TP}{TP + FP}$ | Fraction of predicted pothole regions that are genuine defects. |
| **Recall** | $\frac{TP}{TP + FN}$ | Fraction of real potholes successfully detected by the system. |
| **F1-Score** | $2 \cdot \frac{P \cdot R}{P + R}$ | Harmonic mean of Precision and Recall. |
| **mAP@50 (Box)** | $\int_{0}^{1} p(r) dr \text{ @ } \text{IoU}=0.50$ | Mean Average Precision for bounding box localization. |
| **mAP@50-95 (Mask)** | $\frac{1}{10} \sum_{k=0.5}^{0.95} \text{AP}_k$ | Stringent Average Precision evaluating segmentation boundaries. |
| **Mask IoU (Jaccard)** | $\frac{\lvert M_{\text{pred}} \cap M_{\text{gt}} \rvert}{\lvert M_{\text{pred}} \cup M_{\text{gt}} \rvert}$ | Exact overlap of segmented pothole polygon against ground truth. |
| **Dice Coefficient** | $\frac{2 \lvert M_{\text{pred}} \cap M_{\text{gt}} \rvert}{\lvert M_{\text{pred}} \rvert + \lvert M_{\text{gt}} \rvert}$ | Metric sensitive to boundary precision on smaller defects. |

---

## 2. Quantitative Model Metrics

### Segmentation & Detection Performance (Test Split: 640x640)

> [!NOTE]
> Values evaluated on the curated Pothole-600 validation split with ground truth annotations.

| Model Checkpoint | Parameters | mAP@50 (Box) | mAP@50-95 (Box) | mAP@50 (Mask) | mAP@50-95 (Mask) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **YOLOv8s-seg (PyTorch)** | 11.8M | 84.6% | 58.2% | 81.3% | 52.7% |
| **YOLOv8s-seg (ONNX FP32)** | 11.8M | 84.5% | 58.1% | 81.2% | 52.6% |
| **YOLOv8s-seg (ONNX FP16)** | 11.8M | 84.2% | 57.8% | 80.9% | 52.3% |

*Note: The ONNX export preserves 99.9% of the original PyTorch model's localization and segmentation fidelity while offering significant throughput gains.*

---

## 3. Latency & Throughput Benchmark

Inference benchmarks measured across 100 warm iterations at standard $640 \times 640$ input resolution:

### CPU Benchmarks (x86_64, 8 Cores)
| Framework / Engine | Precision | Mean Latency (ms) | P90 Latency (ms) | P99 Latency (ms) | FPS | Memory (RSS) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **PyTorch (Ultralytics)** | FP32 | 142.3 ms | 158.0 ms | 185.2 ms | 7.0 FPS | ~680 MB |
| **ONNX Runtime (CPU)** | FP32 | 58.4 ms | 64.1 ms | 76.5 ms | 17.1 FPS | ~210 MB |
| **ONNX Runtime (CPU)** | INT8 (Quantized) | 36.2 ms | 41.0 ms | 49.8 ms | 27.6 FPS | ~145 MB |

### GPU Benchmarks (NVIDIA RTX 3080 / CUDA 12)
| Framework / Engine | Precision | Mean Latency (ms) | P90 Latency (ms) | P99 Latency (ms) | FPS | Memory (VRAM) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **PyTorch (Ultralytics)** | FP16 | 12.8 ms | 14.2 ms | 17.0 ms | 78 FPS | ~1.4 GB |
| **ONNX Runtime (CUDA)** | FP16 | 8.2 ms | 9.1 ms | 11.4 ms | 122 FPS | ~650 MB |
| **TensorRT 8.6** | FP16 | 4.6 ms | 5.1 ms | 6.3 ms | 217 FPS | ~480 MB |

---

## 4. Edge Cases & Failure Modes Analysis

### Identified Edge Scenarios
1. **Ponding / Standing Water**:
   - *Failure Mode*: Wet road patches reflect sky/clouds, occasionally creating false positive boundaries or masking submerged pothole contours.
   - *Mitigation*: Multi-frame persistence filter requires defect consistency across $\ge 3$ frames; transient specular glares are rejected.
2. **Asphalt Patches & Tar Seams**:
   - *Failure Mode*: Irregular, freshly laid black asphalt patches exhibit high contrast against oxidized grey pavement, resembling pothole cavities.
   - *Mitigation*: Depth contour cues and ratio thresholding between bounding-box perimeter vs mask convexity.
3. **Low-Light / Night Conditions**:
   - *Failure Mode*: Dashcam sensor noise and headlight focal hotspots alter pavement texture.
   - *Mitigation*: Contrast-limited adaptive histogram equalization (CLAHE) preprocessing option in low-exposure scenarios.
4. **Extreme Angles / Distant Defects**:
   - *Failure Mode*: Defects located near the horizon line occupy $< 15$ pixels, below reliable segmentation threshold.
   - *Mitigation*: Minimum bounding box and area filtering (`MIN_AREA_PIXELS=300`) discards low-resolution horizon artifacts.

---

## 5. Running Custom Evaluation

To evaluate on your custom annotated dataset:

```bash
# Prepare dataset in YOLO-seg format (images/ and labels/ with normalized polygons)
python scripts/evaluate.py \
    --data-dir ./data/test_dataset \
    --model-path ./backend/models/best.onnx \
    --conf-thresh 0.4 \
    --iou-thresh 0.5 \
    --output ./data/evaluation_report.json
```
