"""
YOLOv8-seg postprocessing: box decoding, NMS, and mask construction.
Operates on raw ONNX Runtime outputs.
"""

import numpy as np
import cv2


def xywh_to_xyxy(boxes: np.ndarray) -> np.ndarray:
    """Convert [x_center, y_center, w, h] to [x1, y1, x2, y2]."""
    result = np.zeros_like(boxes)
    result[:, 0] = boxes[:, 0] - boxes[:, 2] / 2  # x1
    result[:, 1] = boxes[:, 1] - boxes[:, 3] / 2  # y1
    result[:, 2] = boxes[:, 0] + boxes[:, 2] / 2  # x2
    result[:, 3] = boxes[:, 1] + boxes[:, 3] / 2  # y2
    return result


def compute_iou(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    """Compute IoU between one box and an array of boxes."""
    x1 = np.maximum(box[0], boxes[:, 0])
    y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[2], boxes[:, 2])
    y2 = np.minimum(box[3], boxes[:, 3])

    intersection = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
    area_box = (box[2] - box[0]) * (box[3] - box[1])
    area_boxes = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    union = area_box + area_boxes - intersection

    return intersection / (union + 1e-6)


def non_max_suppression(
    boxes: np.ndarray, scores: np.ndarray, iou_threshold: float = 0.5
) -> list[int]:
    """
    Greedy NMS: keep highest-score boxes and suppress overlapping ones.
    Returns indices of kept boxes.
    """
    order = scores.argsort()[::-1]
    keep = []

    while order.size > 0:
        idx = order[0]
        keep.append(int(idx))

        if order.size == 1:
            break

        ious = compute_iou(boxes[idx], boxes[order[1:]])
        remaining = np.where(ious < iou_threshold)[0]
        order = order[remaining + 1]

    return keep


def decode_yolov8_seg(
    outputs: list[np.ndarray],
    conf_threshold: float = 0.4,
    iou_threshold: float = 0.5,
    num_classes: int = 1,
) -> dict | None:
    """
    Decode raw YOLOv8-seg ONNX outputs into boxes, scores, masks.

    YOLOv8-seg produces two outputs:
    - detection tensor: [1, N, 4 + num_classes + mask_dim]  (transposed in newer exports)
    - prototype masks: [1, mask_dim, H, W]

    Returns dict with keys: boxes, scores, class_ids, mask_coeffs, proto
    or None if no detections pass the threshold.
    """
    # Determine which output is detection vs proto based on shape
    if len(outputs) != 2:
        # Fallback: some models may only output detections
        return None

    # YOLOv8 ONNX: output0 = [1, 4+nc+mask_dim, N], output1 = [1, mask_dim, H, W]
    out0, out1 = outputs

    # Handle transposed outputs: YOLOv8 exports as [1, features, anchors]
    if out0.ndim == 3 and out0.shape[1] < out0.shape[2]:
        # [1, features, N] -> [1, N, features]
        out0 = out0.transpose(0, 2, 1)

    if out0.ndim == 3:
        det = out0.squeeze(0)  # [N, features]
    else:
        det = out0

    proto = out1.squeeze(0)  # [mask_dim, H, W]
    mask_dim = proto.shape[0]

    # Slice detection tensor
    boxes_xywh = det[:, :4]
    class_scores = det[:, 4 : 4 + num_classes]
    mask_coeffs = det[:, 4 + num_classes : 4 + num_classes + mask_dim]

    # Best class score per detection
    class_ids = class_scores.argmax(axis=1)
    scores = class_scores.max(axis=1)

    # Confidence filter
    mask = scores > conf_threshold
    if mask.sum() == 0:
        return None

    boxes_xywh = boxes_xywh[mask]
    scores = scores[mask]
    class_ids = class_ids[mask]
    mask_coeffs = mask_coeffs[mask]

    # Convert to xyxy
    boxes = xywh_to_xyxy(boxes_xywh)

    # NMS
    keep = non_max_suppression(boxes, scores, iou_threshold)
    if not keep:
        return None

    return {
        "boxes": boxes[keep],
        "scores": scores[keep],
        "class_ids": class_ids[keep],
        "mask_coeffs": mask_coeffs[keep],
        "proto": proto,
    }


def build_instance_masks(
    mask_coeffs: np.ndarray,
    proto: np.ndarray,
    boxes: np.ndarray,
    target_shape: tuple[int, int],
) -> np.ndarray:
    """
    Construct per-instance binary masks from mask coefficients and prototypes.

    Args:
        mask_coeffs: [N, mask_dim] coefficient vectors
        proto: [mask_dim, H, W] prototype masks
        boxes: [N, 4] bounding boxes in xyxy format (at input_size scale)
        target_shape: (height, width) of the target output

    Returns:
        Combined binary mask of shape target_shape, uint8.
    """
    # Matrix multiply: [N, mask_dim] @ [mask_dim, H*W] -> [N, H*W]
    proto_h, proto_w = proto.shape[1], proto.shape[2]
    raw_masks = mask_coeffs @ proto.reshape(proto.shape[0], -1)
    raw_masks = raw_masks.reshape(-1, proto_h, proto_w)

    combined = np.zeros(target_shape, dtype=np.uint8)

    for i in range(raw_masks.shape[0]):
        # Resize mask to target size
        mask_resized = cv2.resize(
            raw_masks[i], (target_shape[1], target_shape[0]),
            interpolation=cv2.INTER_LINEAR
        )
        # Sigmoid activation + threshold
        mask_binary = (1 / (1 + np.exp(-mask_resized)) > 0.5).astype(np.uint8)

        # Clip to bounding box region
        # Scale boxes from input_size to target_shape
        scale_x = target_shape[1] / 640  # assuming 640 input
        scale_y = target_shape[0] / 640

        x1 = max(0, int(boxes[i][0] * scale_x))
        y1 = max(0, int(boxes[i][1] * scale_y))
        x2 = min(target_shape[1], int(boxes[i][2] * scale_x))
        y2 = min(target_shape[0], int(boxes[i][3] * scale_y))

        clipped = np.zeros_like(mask_binary)
        clipped[y1:y2, x1:x2] = mask_binary[y1:y2, x1:x2]

        combined = np.maximum(combined, clipped)

    return combined


def scale_boxes(
    boxes: np.ndarray,
    from_size: tuple[int, int],
    to_size: tuple[int, int],
) -> np.ndarray:
    """Scale bounding boxes from one image size to another."""
    if len(boxes) == 0:
        return boxes.copy()

    scale_x = to_size[1] / from_size[1]
    scale_y = to_size[0] / from_size[0]

    scaled = boxes.astype(np.float32, copy=True)
    scaled[:, [0, 2]] *= scale_x
    scaled[:, [1, 3]] *= scale_y

    # Clip to target boundaries
    scaled[:, [0, 2]] = np.clip(scaled[:, [0, 2]], 0, to_size[1])
    scaled[:, [1, 3]] = np.clip(scaled[:, [1, 3]], 0, to_size[0])

    return scaled
