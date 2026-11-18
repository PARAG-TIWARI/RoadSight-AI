# Dataset preparation guide

This directory currently contains dataset-format guidance. It does not contain the full training or evaluation dataset used to produce the checked-in model weights.

Use this guide when preparing a dataset that you are authorised to use. Record its source, licence, version and split before training or evaluation.

## Expected YOLO segmentation layout

For model training or an Ultralytics validation workflow, a dataset may be organised as:

```text
data/
├── images/
│   ├── train/
│   ├── val/
│   └── test/
├── labels/
│   ├── train/
│   ├── val/
│   └── test/
└── dataset.yaml
```

Example `dataset.yaml`:

```yaml
path: ./data
train: images/train
val: images/val
test: images/test

names:
  0: pothole
```

Each YOLO segmentation label row represents one object instance:

```text
class_id x1 y1 x2 y2 x3 y3 ... xn yn
```

Coordinates are normalised to the image width and height. A polygon requires at least three coordinate pairs. The current model configuration uses one class, `pothole`, at class index `0`.

## Binary-mask layout for the included evaluator

The repository's `scripts/evaluate.py` uses a different format: image files and paired binary PNG masks in separate directories.

```text
evaluation-data/
├── images/
│   ├── frame_001.jpg
│   └── frame_002.png
└── masks/
    ├── frame_001.png
    └── frame_002.png
```

For each input image, the evaluator looks for a PNG mask with the same filename stem. The mask is read as grayscale and thresholded at 127. White/foreground pixels represent the pothole region; black/background pixels represent the rest of the image.

See [Model evaluation](../docs/model-evaluation.md) for the exact command and metrics.

## Dataset provenance checklist

For every dataset used in training or testing, record:

- Dataset title, source URL and version/date.
- Licence and any usage restrictions.
- Number of images and instances per split.
- Class names and annotation conventions.
- Train/validation/test split method.
- Any filtering, deduplication or exclusion applied.
- Capture conditions and known coverage gaps.

Do not put personal, sensitive or unlicensed imagery in the repository.

## Possible public research datasets

RDD2022, CRACK500 and Pothole-600 are examples of road-damage or pavement-distress datasets that may be relevant to future experiments. Their class definitions and annotation formats differ, so check each dataset's current documentation and licence before using them. Listing a dataset here does not mean it was used to train or validate the model in this repository.

## Training considerations

Road footage can vary substantially by camera angle, lighting, weather, road material and motion blur. When building a training or validation set, avoid leakage between splits (for example, adjacent frames from the same recording appearing in both training and test sets). Record any augmentation pipeline and keep a genuinely held-out test split for final reporting.
