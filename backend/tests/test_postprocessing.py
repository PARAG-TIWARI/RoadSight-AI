"""
Tests for postprocessing utilities.
"""

import numpy as np
import pytest

from app.core.postprocessing import (
    xywh_to_xyxy,
    compute_iou,
    non_max_suppression,
    scale_boxes,
)


class TestXywhToXyxy:
    def test_single_box(self):
        boxes = np.array([[50, 50, 20, 30]])  # center 50,50, w=20, h=30
        result = xywh_to_xyxy(boxes)
        np.testing.assert_array_almost_equal(result, [[40, 35, 60, 65]])

    def test_multiple_boxes(self):
        boxes = np.array([
            [100, 100, 50, 50],
            [200, 200, 100, 80],
        ])
        result = xywh_to_xyxy(boxes)
        assert result.shape == (2, 4)
        # First box: x1=75, y1=75, x2=125, y2=125
        assert result[0][0] == 75
        assert result[0][2] == 125


class TestComputeIou:
    def test_perfect_overlap(self):
        box = np.array([0, 0, 10, 10])
        boxes = np.array([[0, 0, 10, 10]])
        iou = compute_iou(box, boxes)
        assert abs(iou[0] - 1.0) < 1e-5

    def test_no_overlap(self):
        box = np.array([0, 0, 10, 10])
        boxes = np.array([[20, 20, 30, 30]])
        iou = compute_iou(box, boxes)
        assert iou[0] < 1e-5

    def test_partial_overlap(self):
        box = np.array([0, 0, 10, 10])
        boxes = np.array([[5, 5, 15, 15]])
        iou = compute_iou(box, boxes)
        # Intersection = 5*5=25, Union = 100+100-25=175
        expected = 25.0 / 175.0
        assert abs(iou[0] - expected) < 1e-4


class TestNMS:
    def test_no_suppression(self):
        boxes = np.array([
            [0, 0, 10, 10],
            [50, 50, 60, 60],
        ])
        scores = np.array([0.9, 0.8])
        keep = non_max_suppression(boxes, scores, iou_threshold=0.5)
        assert len(keep) == 2

    def test_suppression(self):
        boxes = np.array([
            [0, 0, 10, 10],
            [1, 1, 11, 11],  # Almost identical
        ])
        scores = np.array([0.9, 0.8])
        keep = non_max_suppression(boxes, scores, iou_threshold=0.5)
        assert len(keep) == 1
        assert keep[0] == 0  # Higher score kept

    def test_empty_input(self):
        boxes = np.empty((0, 4))
        scores = np.empty(0)
        keep = non_max_suppression(boxes, scores)
        assert len(keep) == 0


class TestScaleBoxes:
    def test_scale_up(self):
        boxes = np.array([[0, 0, 320, 320]])
        scaled = scale_boxes(boxes, from_size=(640, 640), to_size=(1280, 1280))
        np.testing.assert_array_almost_equal(scaled, [[0, 0, 640, 640]])

    def test_scale_down(self):
        boxes = np.array([[0, 0, 640, 640]])
        scaled = scale_boxes(boxes, from_size=(640, 640), to_size=(320, 320))
        np.testing.assert_array_almost_equal(scaled, [[0, 0, 320, 320]])
