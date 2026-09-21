# RoadSight AI — Dataset & Data Pipeline Guide

## 1. Overview

This directory provides guidelines, specifications, and sample schemas for datasets used to train, validate, and benchmark the RoadSight AI YOLOv8-seg pothole detection and instance segmentation models.

---

## 2. Dataset Structure

When preparing custom training or evaluation datasets, format files according to the standard YOLOv8 segmentation structure:

```
data/
├── images/
│   ├── train/
│   │   ├── img_0001.jpg
│   │   └── img_0002.jpg
│   └── val/
│       ├── img_0500.jpg
│       └── img_0501.jpg
├── labels/
│   ├── train/
│   │   ├── img_0001.txt
│   │   └── img_0002.txt
│   └── val/
│       ├── img_0500.txt
│       └── img_0501.txt
└── dataset.yaml
```

### `dataset.yaml` Schema
```yaml
path: ./data
train: images/train
val: images/val
test: images/test

names:
  0: pothole
```

---

## 3. Annotation Format

Each line in a `.txt` label file represents one segmented object instance:

$$\text{class\_index } x_1\ y_1\ x_2\ y_2\ x_3\ y_3\ \dots\ x_n\ y_n$$

- Coordinates are normalized between `0.0` and `1.0` relative to image width and height.
- Minimum 3 vertex pairs defining the closed defect polygon.
- Class index `0` corresponds to `pothole`.

Example:
```text
0 0.352 0.612 0.380 0.630 0.410 0.645 0.440 0.635 0.420 0.610 0.370 0.605
```

---

## 4. Recommended Public Datasets

For retraining, transfer learning, or extended benchmarking:

1. **RDD2022 (Road Damage Dataset)**:
   - Multi-country road damage dataset (Japan, India, Czech Republic, China, USA).
   - Covers longitudinal cracks, transverse cracks, alligator cracking, and potholes (D40).
2. **CRACK500**:
   - High-resolution pavement distress dataset for structural surface crack segmentation.
3. **Pothole-600**:
   - 600+ high-quality annotated pothole images captured from varying camera elevations, wet pavement conditions, and gravel/asphalt roads.

---

## 5. Augmentation Strategy for Dashcam Domain Adaptation

Dashcam video presents specific challenges (motion blur, windshield glare, lens distortion, low light). The following augmentation pipeline is recommended during training:

- **Mosaic & MixUp**: Increases scene complexity and small-target defect density.
- **Random Perspective & Shear**: Simulates varying camera mounting heights and windshield angles.
- **HSV Jitter**: Simulates variable lighting, overcast skies, and direct sunlight glare.
- **Gaussian Blur & Motion Blur**: Simulates vehicle vibration and speed-induced camera blur.
