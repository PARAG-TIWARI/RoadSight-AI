/* ================================================
   RoadSight AI TypeScript Type Definitions
   ================================================ */

export interface DetectionResult {
  bbox: number[];
  confidence: number;
  class_name: string;
  class_id: number;
  area_pixels: number;
  area_ratio: number;
  severity: string;
}

export interface ImagePredictionResponse {
  detections: DetectionResult[];
  detection_count: number;
  average_confidence: number;
  processing_time_ms: number;
  fps: number;
  image_width: number;
  image_height: number;
  model_backend: string;
  annotated_image: string | null;
  mask_overlay: string | null;
  mask_image: string | null;
  total_area_pixels: number;
  total_area_ratio: number;
}

export interface VideoFrameResult {
  frame_index: number;
  timestamp_sec: number;
  detection_count: number;
  average_confidence: number;
  processing_time_ms: number;
}

export interface EventEntry {
  frame_index: number;
  timestamp_sec: number;
  event_type: string;
  description: string;
  confidence: number;
  area_ratio: number;
}

export interface VideoPredictionResponse {
  total_frames: number;
  processed_frames: number;
  detections_total: number;
  average_confidence: number;
  processing_fps: number;
  average_latency_ms: number;
  total_processing_time_sec: number;
  video_duration_sec: number;
  video_fps: number;
  model_backend: string;
  frame_results: VideoFrameResult[];
  events: EventEntry[];
  processed_video_id: string | null;
}

export interface ModelInfo {
  name: string;
  version: string;
  format: string;
  input_size: number;
  classes: string[];
  backend: string;
  provider: string;
  load_time_ms: number;
}

export interface HealthResponse {
  status: string;
  service: string;
  model_loaded: boolean;
  version: string;
}

export interface BenchmarkResult {
  backend: string;
  iterations: number;
  avg_latency_ms: number;
  min_latency_ms: number;
  max_latency_ms: number;
  fps: number;
  cpu_percent: number;
  memory_mb: number;
  input_resolution: string;
  model_format: string;
}

export interface BenchmarkResponse {
  results: BenchmarkResult[];
  system_info: Record<string, string | number | boolean>;
  timestamp: string;
}

export interface MetricsResponse {
  total_predictions: number;
  total_images: number;
  total_videos: number;
  average_inference_ms: number;
  uptime_seconds: number;
}

export interface SystemInfoResponse {
  python_version: string;
  opencv_version: string;
  numpy_version: string;
  api_version: string;
  backend_version: string;
  model_info: ModelInfo;
}
