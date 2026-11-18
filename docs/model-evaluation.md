# Model evaluation and benchmarking

This guide describes what the current repository can measure, how to run it and what evidence should be retained when reporting results.

## 1. Evaluation supported by the current script

The current `scripts/evaluate.py` implementation performs **binary-mask evaluation**. It does not calculate YOLO detection mAP, box AP, per-instance mask AP or class-wise object-detection metrics.

For each image/mask pair, it:

1. Loads the input image.
2. Runs the configured RoadSight model through the image inference pipeline.
3. Decodes the returned combined mask image.
4. Resizes the predicted mask to the ground-truth mask dimensions if required.
5. Calculates mask IoU, Dice, pixel-level precision, pixel-level recall, pixel-level F1 and relative mask-area error.

The aggregate report contains the number of pairs and the mean of those per-image metrics. Individual per-image results are also written when an output path is supplied.

### Expected directory layout

The evaluator expects images and ground-truth masks in separate directories:

```text
evaluation-data/
├── images/
│   ├── frame_001.jpg
│   ├── frame_002.jpg
│   └── frame_003.png
└── masks/
    ├── frame_001.png
    ├── frame_002.png
    └── frame_003.png
```

Image extensions currently discovered by the script are `.jpg` and `.png`. For every image, the matching ground-truth filename must use the same stem and a `.png` extension in the masks directory. Masks are read as grayscale and thresholded at 127 to create a binary foreground mask.

Keep the image and mask pairs aligned. Use a held-out test set that was not used to train or tune the model. Record the dataset source, licence, split and annotation procedure.

### Run the evaluator

From the repository root, with the backend dependencies installed and the configured model file available:

```bash
python scripts/evaluate.py \
  --images ./evaluation-data/images \
  --masks ./evaluation-data/masks \
  --output ./evaluation-report.json
```

The script loads the model through the backend model manager. Ensure that the backend configuration points to the checkpoint you intend to evaluate. The default is the ONNX model configured in `backend/app/core/config.py`.

### Metrics

| Metric | Meaning |
| --- | --- |
| Mask IoU | Intersection divided by union between predicted and ground-truth foreground pixels |
| Dice | Overlap score based on twice the intersection divided by the sum of foreground pixels |
| Pixel precision | Share of predicted foreground pixels that are ground-truth foreground |
| Pixel recall | Share of ground-truth foreground pixels recovered by the prediction |
| Pixel F1 | Harmonic mean of pixel precision and recall |
| Relative area error | Absolute difference in predicted and ground-truth mask area divided by ground-truth mask area |

Empty-mask cases are handled by the metric functions in `backend/app/services/metrics.py`. Interpret results in the context of the dataset and annotation quality.

## 2. Detection metrics not currently reproduced by this repository

Earlier versions of this documentation included YOLOv8 box/mask mAP figures and attributed them to a Pothole-600 split. The current repository does not include that dataset or a YOLO-format mAP evaluation runner. The checked-in evaluator cannot reproduce those mAP figures.

For that reason, those figures are not repeated here as validated results. They should only be published again when the exact dataset version, split, annotation files, checkpoint, evaluation command and raw output are available for review.

If you run Ultralytics validation separately, retain its full console output or JSON/CSV results and document the exact command and package version. Keep that result distinct from the mask-level metrics produced by `scripts/evaluate.py`.

## 3. Inference benchmark included in the project

The current `scripts/benchmark.py` calls `backend/app/services/benchmark.py`. That service creates a synthetic 640-by-640 test image and measures the inference engines that are available:

- ONNX Runtime, using CUDA when its CUDA execution provider is installed and otherwise CPU.
- PyTorch/Ultralytics, when `torch`, `ultralytics` and `backend/models/best.pt` are available.

The timer measures model inference calls. It does not measure image upload, request parsing, preprocessing, post-processing, mask rendering, video processing or frontend latency. The input is synthetic, not a representative road image.

The current implementation does not provide a TensorRT benchmark or a separate INT8-quantized ONNX benchmark. Do not report those rows as results from this script.

### Run a benchmark

```bash
python scripts/benchmark.py --iterations 50 --warmup 5 --format all
```

Options:
- `--iterations`: number of timed runs (default 20).
- `--warmup`: number of warm-up runs (default 3).
- `--format`: output filter for `all`, `onnx` or `pytorch`. The current service attempts each available backend before the CLI filters the returned list.
- `--output`: optional path for a JSON result file.

The API benchmark endpoint is also available at:

```text
GET /api/v1/benchmark?iterations=20
```

The endpoint runs in the request process and blocks while the benchmark executes. Use a modest iteration count and avoid running it during latency-sensitive use.

### What to record with every benchmark

Include all of the following with a published result:

- Commit SHA and model file/checksum.
- Operating system and Python version.
- CPU model, core count and available RAM.
- GPU model, driver and CUDA version when applicable.
- ONNX Runtime, PyTorch and Ultralytics versions.
- Active ONNX execution provider or PyTorch device.
- Input resolution, iteration count and warm-up count.
- Mean, P50, P90, P99 latency, FPS and memory measurement.
- Whether the reported timing is model-only or end-to-end.

Do not compare two measurements if the environment or timing scope differs materially.

## 4. Failure cases to test

Use a held-out set that includes varied road surfaces and capture conditions. At minimum, review:

- Wet pavement, standing water and reflections.
- Dark asphalt repairs, tar seams and road markings.
- Low-light and night footage.
- Motion blur, camera shake and compression artefacts.
- Small or distant defects.
- Images with no potholes.
- Different camera angles and road materials.

The event processor is based on frame-level detection summaries. Its persistence event means a detection was present for the configured number of consecutive processed frames; it is not proof that the same physical pothole was tracked across frames.

## 5. Dataset and result provenance

The `data/` directory currently contains preparation guidance, not the full training or test dataset. Before describing a result as independently reproducible, add or link to:

- Dataset source and licence.
- Version/date and split definition.
- Annotation instructions and class mapping.
- Image/mask counts and any exclusions.
- The exact model checkpoint used.
- Raw evaluator or benchmark output.

Avoid including private, unlicensed or sensitive road footage in the repository.
