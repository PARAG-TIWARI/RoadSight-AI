import { ExternalLink, BookOpen } from 'lucide-react';

const API_BASE = import.meta.env.VITE_API_BASE_URL || import.meta.env.VITE_API_URL || '';

const endpoints = [
  {
    method: 'GET',
    path: '/api/v1/health',
    description: 'Health check — returns API status and model load state',
    sample: '{ "status": "ok", "model_loaded": true, "version": "1.0.0" }',
  },
  {
    method: 'GET',
    path: '/api/v1/model',
    description: 'Model metadata — name, format, input size, classes, backend',
    sample: '{ "name": "YOLOv8-seg Pothole", "format": "onnx", "input_size": 640, "classes": ["pothole"] }',
  },
  {
    method: 'POST',
    path: '/api/v1/predict/image',
    description: 'Run pothole detection on a single image. Accepts multipart/form-data with a "file" field.',
    sample: '{ "detections": [...], "detection_count": 2, "average_confidence": 0.87, "processing_time_ms": 45.2, "fps": 22.1 }',
  },
  {
    method: 'POST',
    path: '/api/v1/predict/video',
    description: 'Process a video file frame-by-frame. Returns per-frame results, events, and aggregate metrics.',
    sample: '{ "total_frames": 300, "processed_frames": 150, "detections_total": 42, "events": [...] }',
  },
  {
    method: 'GET',
    path: '/api/v1/benchmark',
    description: 'Run inference benchmarks comparing available backends. Accepts ?iterations=N query param.',
    sample: '{ "results": [{ "backend": "ONNX Runtime", "fps": 28.5, "avg_latency_ms": 35.1 }] }',
  },
  {
    method: 'GET',
    path: '/api/v1/metrics',
    description: 'Runtime application metrics — total predictions, average latency, uptime.',
    sample: '{ "total_predictions": 15, "average_inference_ms": 41.3, "uptime_seconds": 3600 }',
  },
  {
    method: 'GET',
    path: '/api/v1/system',
    description: 'System and dependency versions — Python, OpenCV, NumPy, model info.',
    sample: '{ "python_version": "3.11.9", "opencv_version": "4.10.0", "model_info": {...} }',
  },
];

export default function ApiDocsPage() {
  return (
    <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 py-10">
      <div className="mb-8">
        <h1 className="section-title">API Documentation</h1>
        <p className="section-subtitle">REST API reference for the RoadSight AI backend</p>
      </div>

      {/* Links to interactive docs */}
      <div className="grid grid-cols-1 sm:grid-cols-2 gap-4 mb-8">
        <a
          href={`${API_BASE}/docs`}
          target="_blank"
          rel="noopener noreferrer"
          className="glass-card-interactive p-5 flex items-center gap-4"
        >
          <div className="w-10 h-10 rounded-lg bg-accent/10 flex items-center justify-center">
            <BookOpen className="w-5 h-5 text-accent" />
          </div>
          <div>
            <h3 className="text-sm font-semibold text-white">Swagger UI</h3>
            <p className="text-xs text-slate-500">Interactive API documentation</p>
          </div>
          <ExternalLink className="w-4 h-4 text-slate-500 ml-auto" />
        </a>
        <a
          href={`${API_BASE}/redoc`}
          target="_blank"
          rel="noopener noreferrer"
          className="glass-card-interactive p-5 flex items-center gap-4"
        >
          <div className="w-10 h-10 rounded-lg bg-purple-500/10 flex items-center justify-center">
            <BookOpen className="w-5 h-5 text-purple-400" />
          </div>
          <div>
            <h3 className="text-sm font-semibold text-white">ReDoc</h3>
            <p className="text-xs text-slate-500">Alternative API documentation</p>
          </div>
          <ExternalLink className="w-4 h-4 text-slate-500 ml-auto" />
        </a>
      </div>

      {/* Endpoint reference */}
      <div className="space-y-4">
        {endpoints.map(ep => (
          <div key={ep.path} className="glass-card p-5">
            <div className="flex items-center gap-3 mb-2">
              <span className={`text-xs font-bold px-2 py-0.5 rounded ${
                ep.method === 'GET' ? 'bg-emerald-500/20 text-emerald-400' : 'bg-amber-500/20 text-amber-400'
              }`}>
                {ep.method}
              </span>
              <code className="text-sm font-mono text-white">{ep.path}</code>
            </div>
            <p className="text-sm text-slate-400 mb-3">{ep.description}</p>
            <div className="bg-surface-200/50 rounded-lg p-3 overflow-x-auto">
              <pre className="text-xs font-mono text-slate-400 whitespace-pre-wrap">{ep.sample}</pre>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
