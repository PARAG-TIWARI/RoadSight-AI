"""
Video inference pipeline: frame-by-frame processing with OpenCV.
Streams frames from video, runs inference, aggregates metrics, generates events.
"""

import logging
import os
import tempfile
import time
import uuid

import cv2
import numpy as np

from app.core.config import settings
from app.core.model import model_manager
from app.core.postprocessing import decode_yolov8_seg, build_instance_masks, scale_boxes
from app.utils.preprocessing import preprocess_for_model
from app.utils.visualization import draw_detections, draw_mask_overlay
from app.services.events import EventProcessor
from app.schemas.responses import (
    VideoPredictionResponse,
    VideoFrameResult,
    EventEntry,
)

logger = logging.getLogger(__name__)


def run_video_inference(
    video_bytes: bytes,
    confidence_threshold: float | None = None,
) -> VideoPredictionResponse:
    """
    Process a video file frame-by-frame through the inference pipeline.

    Uses streaming frame extraction (never loads entire video into memory).
    Generates temporal events and per-frame metrics.
    """
    conf_thresh = confidence_threshold or settings.CONFIDENCE_THRESHOLD
    metadata = model_manager.get_metadata()

    # Write video to temp file for OpenCV access
    video_id = str(uuid.uuid4())[:8]
    temp_input = os.path.join(str(settings.TEMP_DIR), f"input_{video_id}.mp4")
    temp_output = os.path.join(str(settings.TEMP_DIR), f"output_{video_id}.mp4")

    try:
        with open(temp_input, "wb") as f:
            f.write(video_bytes)

        cap = cv2.VideoCapture(temp_input)
        if not cap.isOpened():
            raise ValueError("Failed to open video file. Format may be unsupported.")

        # Video properties
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        video_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        video_duration = total_frames / video_fps if video_fps > 0 else 0.0

        # Output video writer
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(temp_output, fourcc, video_fps, (frame_width, frame_height))

        # Processing state
        frame_results: list[VideoFrameResult] = []
        event_processor = EventProcessor(
            persistence_frames=settings.PERSISTENCE_FRAMES,
            high_confidence_threshold=settings.HIGH_CONFIDENCE_THRESHOLD,
            large_area_ratio=settings.LARGE_AREA_RATIO,
            video_fps=video_fps,
        )

        total_detections = 0
        all_confidences = []
        all_latencies = []
        processed_count = 0
        total_start = time.perf_counter()

        frame_idx = 0
        max_frames = min(total_frames, settings.MAX_VIDEO_FRAMES)

        while cap.isOpened() and processed_count < max_frames:
            ret, frame = cap.read()
            if not ret:
                break

            # Skip frames if configured
            if frame_idx % settings.VIDEO_SKIP_FRAMES != 0:
                writer.write(frame)
                frame_idx += 1
                continue

            timestamp = frame_idx / video_fps if video_fps > 0 else 0.0

            # Preprocess
            input_tensor = preprocess_for_model(frame)

            # Inference
            t0 = time.perf_counter()
            raw_outputs = model_manager.predict(input_tensor)
            latency = (time.perf_counter() - t0) * 1000

            # Decode
            frame_detections = 0
            frame_confidences = []
            annotated_frame = frame.copy()

            if settings.MODEL_FORMAT == "pytorch":
                # PyTorch path
                if raw_outputs and len(raw_outputs) > 0:
                    result = raw_outputs[0]
                    if result.boxes is not None and len(result.boxes) > 0:
                        boxes_np = result.boxes.xyxy.cpu().numpy()
                        scores_np = result.boxes.conf.cpu().numpy()

                        valid = scores_np >= conf_thresh
                        if valid.any():
                            boxes_np = boxes_np[valid]
                            scores_np = scores_np[valid]
                            frame_detections = len(boxes_np)
                            frame_confidences = scores_np.tolist()

                            annotated_frame = draw_detections(frame, boxes_np, scores_np)

                            if result.masks is not None:
                                masks_np = result.masks.data.cpu().numpy()
                                combined = np.zeros((frame_height, frame_width), dtype=np.uint8)
                                for m_idx in range(min(len(masks_np), len(boxes_np))):
                                    mr = cv2.resize(masks_np[m_idx], (frame_width, frame_height))
                                    combined = np.maximum(combined, (mr > 0.5).astype(np.uint8))
                                annotated_frame = draw_mask_overlay(annotated_frame, combined)
            else:
                # ONNX path
                decoded = decode_yolov8_seg(
                    raw_outputs,
                    conf_threshold=conf_thresh,
                    iou_threshold=settings.NMS_IOU_THRESHOLD,
                    num_classes=settings.NUM_CLASSES,
                )

                if decoded is not None:
                    boxes = decoded["boxes"]
                    scores = decoded["scores"]
                    mask_coeffs = decoded["mask_coeffs"]
                    proto = decoded["proto"]

                    scaled = scale_boxes(
                        boxes,
                        from_size=(settings.INPUT_SIZE, settings.INPUT_SIZE),
                        to_size=(frame_height, frame_width),
                    )

                    frame_detections = len(boxes)
                    frame_confidences = scores.tolist()

                    annotated_frame = draw_detections(frame, scaled, scores)

                    combined_mask = build_instance_masks(
                        mask_coeffs, proto, boxes, (frame_height, frame_width)
                    )
                    annotated_frame = draw_mask_overlay(annotated_frame, combined_mask)

            writer.write(annotated_frame)

            avg_frame_conf = float(np.mean(frame_confidences)) if frame_confidences else 0.0

            # Compute area ratio for event processing
            area_ratio = 0.0
            if decoded is not None and settings.MODEL_FORMAT != "pytorch":
                area_ratio = float(combined_mask.sum()) / (frame_height * frame_width)

            frame_results.append(VideoFrameResult(
                frame_index=frame_idx,
                timestamp_sec=round(timestamp, 3),
                detection_count=frame_detections,
                average_confidence=round(avg_frame_conf, 4),
                processing_time_ms=round(latency, 2),
            ))

            # Feed to event processor
            event_processor.process_frame(
                frame_index=frame_idx,
                timestamp_sec=timestamp,
                detection_count=frame_detections,
                avg_confidence=avg_frame_conf,
                area_ratio=area_ratio,
            )

            total_detections += frame_detections
            all_confidences.extend(frame_confidences)
            all_latencies.append(latency)
            processed_count += 1
            frame_idx += 1

        cap.release()
        writer.release()

        total_time = time.perf_counter() - total_start
        processing_fps = processed_count / total_time if total_time > 0 else 0.0
        avg_latency = float(np.mean(all_latencies)) if all_latencies else 0.0
        avg_confidence = float(np.mean(all_confidences)) if all_confidences else 0.0

        return VideoPredictionResponse(
            total_frames=total_frames,
            processed_frames=processed_count,
            detections_total=total_detections,
            average_confidence=round(avg_confidence, 4),
            processing_fps=round(processing_fps, 2),
            average_latency_ms=round(avg_latency, 2),
            total_processing_time_sec=round(total_time, 2),
            video_duration_sec=round(video_duration, 2),
            video_fps=round(video_fps, 2),
            model_backend=metadata.backend,
            frame_results=frame_results,
            events=event_processor.get_events(),
            processed_video_id=video_id,
        )

    except Exception:
        # Clean up temp files on error
        for f in [temp_input, temp_output]:
            if os.path.exists(f):
                try:
                    os.remove(f)
                except OSError:
                    pass
        raise
    finally:
        # Always clean up input temp file
        if os.path.exists(temp_input):
            try:
                os.remove(temp_input)
            except OSError:
                pass
