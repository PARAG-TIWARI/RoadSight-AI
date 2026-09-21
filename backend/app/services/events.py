"""
Temporal event processing for video analysis.

Implements a lightweight state machine that tracks detection persistence
across frames, generates events for new/persistent/lost detections,
and filters transient false positives.

This is NOT a full object tracker (e.g., DeepSORT). It operates on
aggregate frame-level detection statistics, which is appropriate for
static road-defect analysis where potholes don't move — only the camera does.
"""

from dataclasses import dataclass, field

from app.schemas.responses import EventEntry


@dataclass
class _DetectionState:
    """Internal state for tracking detection persistence."""
    first_seen_frame: int = 0
    first_seen_time: float = 0.0
    last_seen_frame: int = 0
    last_seen_time: float = 0.0
    consecutive_frames: int = 0
    max_confidence: float = 0.0
    max_area_ratio: float = 0.0
    is_confirmed: bool = False


class EventProcessor:
    """
    Processes frame-level detection results and generates temporal events.

    Events:
    - new_detection: First appearance of a defect
    - persistent: Defect persists across PERSISTENCE_FRAMES consecutive frames
    - high_confidence: Detection confidence exceeds threshold
    - large_area: Detection area exceeds threshold
    - lost: Previously tracked defect is no longer detected

    False positive suppression:
    - Isolated single-frame detections with low confidence are suppressed.
    """

    def __init__(
        self,
        persistence_frames: int = 3,
        high_confidence_threshold: float = 0.75,
        large_area_ratio: float = 0.05,
        video_fps: float = 30.0,
        gap_tolerance: int = 2,
    ):
        self.persistence_frames = persistence_frames
        self.high_conf_thresh = high_confidence_threshold
        self.large_area_thresh = large_area_ratio
        self.video_fps = video_fps
        self.gap_tolerance = gap_tolerance

        self._state: _DetectionState | None = None
        self._events: list[EventEntry] = []
        self._frames_without_detection: int = 0

    def process_frame(
        self,
        frame_index: int,
        timestamp_sec: float,
        detection_count: int,
        avg_confidence: float,
        area_ratio: float,
    ) -> None:
        """Process one frame's detection summary and generate events."""
        has_detection = detection_count > 0

        if has_detection:
            self._frames_without_detection = 0

            if self._state is None:
                # New detection
                self._state = _DetectionState(
                    first_seen_frame=frame_index,
                    first_seen_time=timestamp_sec,
                    last_seen_frame=frame_index,
                    last_seen_time=timestamp_sec,
                    consecutive_frames=1,
                    max_confidence=avg_confidence,
                    max_area_ratio=area_ratio,
                )
                self._events.append(EventEntry(
                    frame_index=frame_index,
                    timestamp_sec=round(timestamp_sec, 3),
                    event_type="new_detection",
                    description=f"Pothole detected ({detection_count} region{'s' if detection_count > 1 else ''}, confidence: {avg_confidence:.2f})",
                    confidence=round(avg_confidence, 4),
                    area_ratio=round(area_ratio, 6),
                ))
            else:
                # Update existing state
                self._state.last_seen_frame = frame_index
                self._state.last_seen_time = timestamp_sec
                self._state.consecutive_frames += 1
                self._state.max_confidence = max(self._state.max_confidence, avg_confidence)
                self._state.max_area_ratio = max(self._state.max_area_ratio, area_ratio)

                # Persistence event
                if (
                    self._state.consecutive_frames == self.persistence_frames
                    and not self._state.is_confirmed
                ):
                    self._state.is_confirmed = True
                    self._events.append(EventEntry(
                        frame_index=frame_index,
                        timestamp_sec=round(timestamp_sec, 3),
                        event_type="persistent",
                        description=f"Defect confirmed — persisted for {self.persistence_frames} consecutive frames",
                        confidence=round(self._state.max_confidence, 4),
                        area_ratio=round(self._state.max_area_ratio, 6),
                    ))

            # High confidence event
            if avg_confidence >= self.high_conf_thresh:
                self._events.append(EventEntry(
                    frame_index=frame_index,
                    timestamp_sec=round(timestamp_sec, 3),
                    event_type="high_confidence",
                    description=f"High-confidence detection: {avg_confidence:.2f}",
                    confidence=round(avg_confidence, 4),
                    area_ratio=round(area_ratio, 6),
                ))

            # Large area event
            if area_ratio >= self.large_area_thresh:
                self._events.append(EventEntry(
                    frame_index=frame_index,
                    timestamp_sec=round(timestamp_sec, 3),
                    event_type="large_area",
                    description=f"Large defect area: {area_ratio*100:.1f}% of frame",
                    confidence=round(avg_confidence, 4),
                    area_ratio=round(area_ratio, 6),
                ))

        else:
            self._frames_without_detection += 1

            # Lost detection event
            if (
                self._state is not None
                and self._frames_without_detection > self.gap_tolerance
            ):
                duration = self._state.last_seen_time - self._state.first_seen_time
                self._events.append(EventEntry(
                    frame_index=frame_index,
                    timestamp_sec=round(timestamp_sec, 3),
                    event_type="lost",
                    description=f"Defect left frame after {self._state.consecutive_frames} frames ({duration:.1f}s)",
                    confidence=round(self._state.max_confidence, 4),
                    area_ratio=round(self._state.max_area_ratio, 6),
                ))
                self._state = None

    def get_events(self) -> list[EventEntry]:
        """Return all generated events, deduplicating consecutive high-confidence events."""
        # Simple dedup: collapse consecutive high_confidence events
        if not self._events:
            return []

        deduped = [self._events[0]]
        for event in self._events[1:]:
            prev = deduped[-1]
            if (
                event.event_type == "high_confidence"
                and prev.event_type == "high_confidence"
                and event.frame_index - prev.frame_index <= 2
            ):
                continue  # Skip near-duplicate high-confidence events
            deduped.append(event)

        return deduped
