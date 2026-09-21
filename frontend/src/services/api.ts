/**
 * RoadSight AI API Client
 * Typed HTTP client for all backend endpoints.
 */

import axios, { AxiosInstance } from 'axios';
import type {
  HealthResponse,
  ModelInfo,
  ImagePredictionResponse,
  VideoPredictionResponse,
  BenchmarkResponse,
  MetricsResponse,
  SystemInfoResponse,
} from '../types';

const API_BASE = import.meta.env.VITE_API_BASE_URL || import.meta.env.VITE_API_URL || '';

const client: AxiosInstance = axios.create({
  baseURL: `${API_BASE}/api/v1`,
  timeout: 300000, // 5 min for video processing
});

export const api = {
  /** Health check */
  health: () => client.get<HealthResponse>('/health').then(r => r.data),

  /** Model information */
  modelInfo: () => client.get<ModelInfo>('/model').then(r => r.data),

  /** Image inference */
  predictImage: (file: File, confidence?: number) => {
    const form = new FormData();
    form.append('file', file);
    const params = confidence != null ? { confidence } : {};
    return client
      .post<ImagePredictionResponse>('/predict/image', form, { params })
      .then(r => r.data);
  },

  /** Video inference */
  predictVideo: (file: File, confidence?: number) => {
    const form = new FormData();
    form.append('file', file);
    const params = confidence != null ? { confidence } : {};
    return client
      .post<VideoPredictionResponse>('/predict/video', form, { params })
      .then(r => r.data);
  },

  /** Download processed video */
  getProcessedVideoUrl: (videoId: string) =>
    `${API_BASE}/api/v1/predict/video/${videoId}/download`,

  /** Run benchmark */
  benchmark: (iterations = 10) =>
    client
      .get<BenchmarkResponse>('/benchmark', { params: { iterations } })
      .then(r => r.data),

  /** Runtime metrics */
  metrics: () => client.get<MetricsResponse>('/metrics').then(r => r.data),

  /** System information */
  systemInfo: () => client.get<SystemInfoResponse>('/system').then(r => r.data),
};
