"""
Single image inference pipeline.
Handles the full flow: validate → preprocess → infer → decode → visualize → response.
"""

import logging
import time

import cv2
import numpy as np

from app.core.config import settings
from app.core.model import model_manager
from app.core.postprocessing import (
    decode_yolov8_seg,
    build_instance_masks,
    scale_boxes,
)
from app.utils.preprocessing import decode_image, preprocess_for_model, validate_image_bytes
from app.utils.visualization import (
    draw_detections,
    draw_mask_overlay,
    encode_image_base64,
    create_mask_visualization,
    get_severity_category,
)
from app.schemas.responses import ImagePredictionResponse, DetectionResult

logger = logging.getLogger(__name__)


def run_image_inference(
    image_bytes: bytes,
    filename: str = "",
    confidence_threshold: float | None = None,
) -> ImagePredictionResponse:
    """
    Run the full image inference pipeline.

    Returns an ImagePredictionResponse with detections, visualizations, and metrics.
    """
    # --- Validate ---
    validate_image_bytes(image_bytes, filename)
    conf_thresh = confidence_threshold or settings.CONFIDENCE_THRESHOLD

    # --- Decode original image ---
    original_bgr = decode_image(image_bytes)
    orig_h, orig_w = original_bgr.shape[:2]

    # --- Preprocess ---
    input_tensor = preprocess_for_model(original_bgr)

    # --- Inference ---
    start_time = time.perf_counter()
    raw_outputs = model_manager.predict(input_tensor)
    inference_time = time.perf_counter() - start_time

    processing_time_ms = inference_time * 1000
    fps = 1.0 / inference_time if inference_time > 0 else 0.0

    metadata = model_manager.get_metadata()

    # --- Handle PyTorch backend (returns ultralytics Results) ---
    if settings.MODEL_FORMAT == "pytorch":
        return _process_pytorch_results(
            raw_outputs, original_bgr, orig_h, orig_w,
            processing_time_ms, fps, metadata.backend, conf_thresh,
        )

    # --- Decode ONNX outputs ---
    decoded = decode_yolov8_seg(
        raw_outputs,
        conf_threshold=conf_thresh,
        iou_threshold=settings.NMS_IOU_THRESHOLD,
        num_classes=settings.NUM_CLASSES,
    )

    if decoded is None:
        return ImagePredictionResponse(
            detections=[],
            detection_count=0,
            average_confidence=0.0,
            processing_time_ms=processing_time_ms,
            fps=fps,
            image_width=orig_w,
            image_height=orig_h,
            model_backend=metadata.backend,
        )

    # --- Build masks ---
    boxes = decoded["boxes"]
    scores = decoded["scores"]
    class_ids = decoded["class_ids"]
    mask_coeffs = decoded["mask_coeffs"]
    proto = decoded["proto"]

    # Build combined mask at original image resolution
    combined_mask = build_instance_masks(
        mask_coeffs, proto, boxes, (orig_h, orig_w)
    )

    # Scale boxes from model input to original image size
    scaled_boxes = scale_boxes(
        boxes,
        from_size=(settings.INPUT_SIZE, settings.INPUT_SIZE),
        to_size=(orig_h, orig_w),
    )

    # --- Build detection results ---
    detections = []
    total_area = 0

    for i in range(len(scaled_boxes)):
        # Per-detection mask area
        x1, y1, x2, y2 = scaled_boxes[i].astype(int)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(orig_w, x2), min(orig_h, y2)

        region_mask = combined_mask[y1:y2, x1:x2]
        area_px = int(region_mask.sum())
        total_area += area_px
        area_ratio = area_px / (orig_h * orig_w) if (orig_h * orig_w) > 0 else 0.0

        detections.append(DetectionResult(
            bbox=[float(x1), float(y1), float(x2), float(y2)],
            confidence=float(scores[i]),
            class_name=settings.CLASS_NAMES[int(class_ids[i])] if int(class_ids[i]) < len(settings.CLASS_NAMES) else "pothole",
            class_id=int(class_ids[i]),
            area_pixels=area_px,
            area_ratio=round(area_ratio, 6),
            severity=get_severity_category(area_ratio),
        ))

    total_area_ratio = total_area / (orig_h * orig_w) if (orig_h * orig_w) > 0 else 0.0
    avg_conf = float(np.mean(scores)) if len(scores) > 0 else 0.0

    # --- Visualizations ---
    annotated = draw_detections(original_bgr, scaled_boxes, scores, settings.CLASS_NAMES, class_ids)
    overlay = draw_mask_overlay(original_bgr, combined_mask)
    mask_vis = create_mask_visualization(combined_mask)

    return ImagePredictionResponse(
        detections=detections,
        detection_count=len(detections),
        average_confidence=round(avg_conf, 4),
        processing_time_ms=round(processing_time_ms, 2),
        fps=round(fps, 2),
        image_width=orig_w,
        image_height=orig_h,
        model_backend=metadata.backend,
        annotated_image=encode_image_base64(annotated),
        mask_overlay=encode_image_base64(overlay),
        mask_image=encode_image_base64(mask_vis),
        total_area_pixels=total_area,
        total_area_ratio=round(total_area_ratio, 6),
    )


def _process_pytorch_results(
    results, original_bgr, orig_h, orig_w,
    processing_time_ms, fps, backend_name, conf_thresh,
) -> ImagePredictionResponse:
    """Process ultralytics YOLO Results objects."""
    if not results or len(results) == 0:
        return ImagePredictionResponse(
            detections=[], detection_count=0, average_confidence=0.0,
            processing_time_ms=processing_time_ms, fps=fps,
            image_width=orig_w, image_height=orig_h, model_backend=backend_name,
        )

    result = results[0]
    detections = []
    combined_mask = np.zeros((orig_h, orig_w), dtype=np.uint8)

    if result.boxes is not None:
        boxes_data = result.boxes.xyxy.cpu().numpy()
        scores_data = result.boxes.conf.cpu().numpy()
        class_ids_data = result.boxes.cls.cpu().numpy().astype(int)

        # Get masks if available
        masks_data = None
        if result.masks is not None:
            masks_data = result.masks.data.cpu().numpy()

        for i in range(len(boxes_data)):
            if scores_data[i] < conf_thresh:
                continue

            x1, y1, x2, y2 = boxes_data[i]
            area_px = 0

            if masks_data is not None and i < len(masks_data):
                mask_resized = cv2.resize(masks_data[i], (orig_w, orig_h))
                mask_binary = (mask_resized > 0.5).astype(np.uint8)
                combined_mask = np.maximum(combined_mask, mask_binary)
                area_px = int(mask_binary.sum())

            area_ratio = area_px / (orig_h * orig_w) if (orig_h * orig_w) > 0 else 0.0

            detections.append(DetectionResult(
                bbox=[float(x1), float(y1), float(x2), float(y2)],
                confidence=float(scores_data[i]),
                class_name=settings.CLASS_NAMES[int(class_ids_data[i])] if int(class_ids_data[i]) < len(settings.CLASS_NAMES) else "pothole",
                class_id=int(class_ids_data[i]),
                area_pixels=area_px,
                area_ratio=round(area_ratio, 6),
                severity=get_severity_category(area_ratio),
            ))

    avg_conf = np.mean([d.confidence for d in detections]) if detections else 0.0

    annotated = draw_detections(
        original_bgr,
        np.array([d.bbox for d in detections]) if detections else np.empty((0, 4)),
        np.array([d.confidence for d in detections]) if detections else np.empty(0),
    )
    overlay = draw_mask_overlay(original_bgr, combined_mask)
    mask_vis = create_mask_visualization(combined_mask)

    total_area = sum(d.area_pixels for d in detections)
    total_ratio = total_area / (orig_h * orig_w) if (orig_h * orig_w) > 0 else 0.0

    return ImagePredictionResponse(
        detections=detections,
        detection_count=len(detections),
        average_confidence=round(float(avg_conf), 4),
        processing_time_ms=round(processing_time_ms, 2),
        fps=round(fps, 2),
        image_width=orig_w,
        image_height=orig_h,
        model_backend=backend_name,
        annotated_image=encode_image_base64(annotated),
        mask_overlay=encode_image_base64(overlay),
        mask_image=encode_image_base64(mask_vis),
        total_area_pixels=total_area,
        total_area_ratio=round(total_ratio, 6),
    )
