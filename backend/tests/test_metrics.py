"""
Tests for evaluation metrics.
"""

import numpy as np
import pytest

from app.services.metrics import (
    intersection_over_union,
    dice_coefficient,
    precision_recall_f1,
    area_error,
)


class TestIoU:
    def test_perfect_match(self):
        mask = np.ones((100, 100), dtype=np.uint8)
        assert abs(intersection_over_union(mask, mask) - 1.0) < 1e-6

    def test_no_overlap(self):
        pred = np.zeros((100, 100), dtype=np.uint8)
        pred[:50, :] = 1
        gt = np.zeros((100, 100), dtype=np.uint8)
        gt[50:, :] = 1
        assert intersection_over_union(pred, gt) < 1e-6

    def test_partial_overlap(self):
        pred = np.zeros((100, 100), dtype=np.uint8)
        pred[:60, :] = 1
        gt = np.zeros((100, 100), dtype=np.uint8)
        gt[:40, :] = 1
        iou = intersection_over_union(pred, gt)
        # Intersection = 40*100=4000, Union = 60*100=6000
        assert abs(iou - 4000 / 6000) < 1e-4

    def test_both_empty(self):
        mask = np.zeros((10, 10), dtype=np.uint8)
        assert intersection_over_union(mask, mask) == 1.0


class TestDice:
    def test_perfect_match(self):
        mask = np.ones((50, 50), dtype=np.uint8)
        assert abs(dice_coefficient(mask, mask) - 1.0) < 1e-6

    def test_no_overlap(self):
        pred = np.zeros((100, 100), dtype=np.uint8)
        pred[:50, :] = 1
        gt = np.zeros((100, 100), dtype=np.uint8)
        gt[50:, :] = 1
        assert dice_coefficient(pred, gt) < 1e-6


class TestPrecisionRecallF1:
    def test_perfect(self):
        mask = np.ones((10, 10), dtype=np.uint8)
        p, r, f1 = precision_recall_f1(mask, mask)
        assert abs(p - 1.0) < 1e-6
        assert abs(r - 1.0) < 1e-6
        assert abs(f1 - 1.0) < 1e-6

    def test_all_false_positive(self):
        pred = np.ones((10, 10), dtype=np.uint8)
        gt = np.zeros((10, 10), dtype=np.uint8)
        p, r, f1 = precision_recall_f1(pred, gt)
        assert p < 1e-6  # No true positives
        assert r == 0.0   # No GT positives


class TestAreaError:
    def test_same_area(self):
        mask = np.ones((10, 10), dtype=np.uint8)
        assert area_error(mask, mask) < 1e-6

    def test_double_area(self):
        pred = np.ones((10, 20), dtype=np.uint8)
        gt = np.ones((10, 10), dtype=np.uint8)
        err = area_error(pred, gt)
        assert abs(err - 1.0) < 1e-4  # 200-100/100 = 1.0
