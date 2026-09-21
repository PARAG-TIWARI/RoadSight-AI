"""
Evaluation metrics for segmentation quality assessment.
These require ground-truth annotations and are used by scripts/evaluate.py.
"""

import numpy as np


def intersection_over_union(pred_mask: np.ndarray, gt_mask: np.ndarray) -> float:
    """
    Compute Intersection over Union (IoU / Jaccard Index).

    Both masks should be binary (0/1) with the same shape.
    """
    pred = pred_mask.astype(bool)
    gt = gt_mask.astype(bool)

    intersection = np.logical_and(pred, gt).sum()
    union = np.logical_or(pred, gt).sum()

    if union == 0:
        return 1.0 if intersection == 0 else 0.0

    return float(intersection / union)


def dice_coefficient(pred_mask: np.ndarray, gt_mask: np.ndarray) -> float:
    """
    Compute Dice coefficient (F1 for segmentation).

    Dice = 2 * |P ∩ G| / (|P| + |G|)
    """
    pred = pred_mask.astype(bool)
    gt = gt_mask.astype(bool)

    intersection = np.logical_and(pred, gt).sum()
    total = pred.sum() + gt.sum()

    if total == 0:
        return 1.0 if intersection == 0 else 0.0

    return float(2 * intersection / total)


def precision_recall_f1(
    pred_mask: np.ndarray, gt_mask: np.ndarray
) -> tuple[float, float, float]:
    """
    Compute pixel-level precision, recall, and F1 score.

    Returns: (precision, recall, f1)
    """
    pred = pred_mask.astype(bool)
    gt = gt_mask.astype(bool)

    tp = np.logical_and(pred, gt).sum()
    fp = np.logical_and(pred, ~gt).sum()
    fn = np.logical_and(~pred, gt).sum()

    precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    recall = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    f1 = float(2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    return precision, recall, f1


def area_error(pred_mask: np.ndarray, gt_mask: np.ndarray) -> float:
    """
    Relative error in predicted mask area vs ground truth area.
    Returns: |pred_area - gt_area| / gt_area
    """
    pred_area = pred_mask.astype(bool).sum()
    gt_area = gt_mask.astype(bool).sum()

    if gt_area == 0:
        return 0.0 if pred_area == 0 else float("inf")

    return float(abs(pred_area - gt_area) / gt_area)
