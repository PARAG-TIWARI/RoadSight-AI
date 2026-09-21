"""
Visualization utilities for drawing detection results on images.
"""

import base64
import cv2
import numpy as np


# Color palette for detections (BGR format)
DETECTION_COLOR = (0, 200, 255)  # Amber/yellow
MASK_COLOR = (50, 50, 255)       # Red tint for masks
SEVERITY_COLORS = {
    "low": (0, 200, 100),       # Green
    "medium": (0, 200, 255),    # Yellow
    "high": (0, 100, 255),      # Orange
    "critical": (0, 0, 255),    # Red
}


def draw_detections(
    image_bgr: np.ndarray,
    boxes: np.ndarray,
    scores: np.ndarray,
    class_names: list[str] | None = None,
    class_ids: np.ndarray | None = None,
    thickness: int = 2,
) -> np.ndarray:
    """
    Draw bounding boxes and confidence labels on an image.
    Returns a copy with annotations drawn.
    """
    annotated = image_bgr.copy()
    names = class_names or ["pothole"]

    for i in range(len(boxes)):
        x1, y1, x2, y2 = boxes[i].astype(int)
        score = float(scores[i])
        cls_id = int(class_ids[i]) if class_ids is not None else 0
        label = f"{names[cls_id]} {score:.2f}"

        # Box
        cv2.rectangle(annotated, (x1, y1), (x2, y2), DETECTION_COLOR, thickness)

        # Label background
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(annotated, (x1, y1 - th - 8), (x1 + tw + 4, y1), DETECTION_COLOR, -1)
        cv2.putText(
            annotated, label, (x1 + 2, y1 - 4),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA
        )

    return annotated


def draw_mask_overlay(
    image_bgr: np.ndarray,
    mask: np.ndarray,
    alpha: float = 0.45,
    color: tuple[int, int, int] = MASK_COLOR,
) -> np.ndarray:
    """
    Overlay a translucent color on masked regions.
    mask: binary uint8 mask (same H, W as image).
    """
    overlay = image_bgr.copy()
    color_arr = np.array(color, dtype=np.float32)

    idx = mask > 0
    if np.any(idx):
        base = overlay[idx].astype(np.float32)
        blended = (1.0 - alpha) * base + alpha * color_arr
        overlay[idx] = blended.clip(0, 255).astype(np.uint8)

    return overlay


def encode_image_base64(image_bgr: np.ndarray, format: str = ".png") -> str:
    """Encode an image to base64 string."""
    ok, buf = cv2.imencode(format, image_bgr)
    if not ok:
        raise ValueError(f"Failed to encode image as {format}")
    return base64.b64encode(buf.tobytes()).decode("utf-8")


def create_mask_visualization(mask: np.ndarray) -> np.ndarray:
    """
    Create a 3-channel visualization of a binary mask.
    White = detected region, black = background.
    """
    if mask.ndim == 2:
        vis = np.zeros((*mask.shape, 3), dtype=np.uint8)
        vis[mask > 0] = (255, 255, 255)
        return vis
    return mask


def get_severity_category(area_ratio: float) -> str:
    """
    Categorize pothole severity based on area ratio.

    Rule-based categorization:
    - low: < 1% of image area
    - medium: 1-3% of image area
    - high: 3-8% of image area
    - critical: > 8% of image area

    These thresholds are heuristic and based on the assumption
    that a larger portion of the frame indicates more severe damage.
    """
    if area_ratio < 0.01:
        return "low"
    elif area_ratio < 0.03:
        return "medium"
    elif area_ratio < 0.08:
        return "high"
    else:
        return "critical"
