"""
Evaluate model performance against ground-truth annotations.

Calculates: IoU, Dice score, Precision, Recall, F1, Area Error.

Usage:
    python scripts/evaluate.py --images data/images/ --masks data/masks/

Ground truth masks should be binary PNG images with the same filenames as the input images.
If no ground-truth data is available, this script will report that metrics cannot be computed.
"""

import argparse
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np

# Add backend to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'backend'))

from app.services.metrics import (
    intersection_over_union,
    dice_coefficient,
    precision_recall_f1,
    area_error,
)
from app.core.config import settings


def main():
    parser = argparse.ArgumentParser(description="Evaluate model against ground truth")
    parser.add_argument("--images", type=str, default="data/images", help="Directory with input images")
    parser.add_argument("--masks", type=str, default="data/masks", help="Directory with ground-truth masks")
    parser.add_argument("--output", type=str, default=None, help="Output JSON file")
    args = parser.parse_args()

    images_dir = Path(args.images)
    masks_dir = Path(args.masks)

    if not images_dir.exists():
        print(f"Images directory not found: {images_dir}")
        print("Evaluation requires ground-truth data. See data/README.md for instructions.")
        print("\nStatus: METRICS UNAVAILABLE (no ground-truth data)")
        sys.exit(0)

    if not masks_dir.exists():
        print(f"Masks directory not found: {masks_dir}")
        print("Evaluation requires paired ground-truth masks.")
        print("\nStatus: METRICS UNAVAILABLE (no ground-truth masks)")
        sys.exit(0)

    # Find paired images
    image_files = sorted(images_dir.glob("*.jpg")) + sorted(images_dir.glob("*.png"))
    pairs = []
    for img_path in image_files:
        mask_path = masks_dir / img_path.with_suffix(".png").name
        if mask_path.exists():
            pairs.append((img_path, mask_path))

    if not pairs:
        print("No paired image/mask files found.")
        print("\nStatus: METRICS UNAVAILABLE (no matching pairs)")
        sys.exit(0)

    print(f"Found {len(pairs)} image/mask pairs")

    # Run evaluation
    from app.core.model import model_manager
    from app.services.image_inference import run_image_inference

    model_manager.load()

    results = []
    for img_path, mask_path in pairs:
        print(f"  Evaluating: {img_path.name}...", end=" ")

        # Load ground truth mask
        gt_mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        gt_mask = (gt_mask > 127).astype(np.uint8)

        # Run inference
        with open(img_path, "rb") as f:
            image_bytes = f.read()

        prediction = run_image_inference(image_bytes)

        # Create predicted mask from detections
        pred_mask = np.zeros_like(gt_mask)
        if prediction.mask_image:
            import base64
            mask_bytes = base64.b64decode(prediction.mask_image)
            mask_arr = cv2.imdecode(np.frombuffer(mask_bytes, np.uint8), cv2.IMREAD_GRAYSCALE)
            if mask_arr is not None:
                pred_mask = cv2.resize(mask_arr, (gt_mask.shape[1], gt_mask.shape[0]))
                pred_mask = (pred_mask > 127).astype(np.uint8)

        # Calculate metrics
        iou = intersection_over_union(pred_mask, gt_mask)
        dice = dice_coefficient(pred_mask, gt_mask)
        p, r, f1 = precision_recall_f1(pred_mask, gt_mask)
        ae = area_error(pred_mask, gt_mask)

        results.append({
            "image": img_path.name,
            "iou": round(iou, 4),
            "dice": round(dice, 4),
            "precision": round(p, 4),
            "recall": round(r, 4),
            "f1": round(f1, 4),
            "area_error": round(ae, 4),
        })
        print(f"IoU={iou:.3f}, Dice={dice:.3f}, F1={f1:.3f}")

    # Aggregate
    avg = {
        "count": len(results),
        "mean_iou": round(np.mean([r["iou"] for r in results]), 4),
        "mean_dice": round(np.mean([r["dice"] for r in results]), 4),
        "mean_precision": round(np.mean([r["precision"] for r in results]), 4),
        "mean_recall": round(np.mean([r["recall"] for r in results]), 4),
        "mean_f1": round(np.mean([r["f1"] for r in results]), 4),
        "mean_area_error": round(np.mean([r["area_error"] for r in results]), 4),
    }

    output = {"summary": avg, "per_image": results}

    print(f"\n--- Evaluation Summary ({len(results)} images) ---")
    for k, v in avg.items():
        print(f"  {k}: {v}")

    if args.output:
        with open(args.output, "w") as f:
            json.dump(output, f, indent=2)
        print(f"\nResults saved to: {args.output}")


if __name__ == "__main__":
    main()
