import { useState, useRef } from 'react';
import { Video, Loader2, Download, AlertCircle, Clock, Zap } from 'lucide-react';
import { api } from '../services/api';
import { MetricCard } from '../components/dashboard/MetricCard';
import type { VideoPredictionResponse } from '../types';

const EVENT_COLORS: Record<string, string> = {
  new_detection: 'border-accent text-accent',
  persistent: 'border-emerald-500 text-emerald-400',
  high_confidence: 'border-amber-500 text-amber-400',
  large_area: 'border-red-500 text-red-400',
  lost: 'border-slate-500 text-slate-400',
};

export default function VideoAnalysisPage() {
  const [file, setFile] = useState<File | null>(null);
  const [result, setResult] = useState<VideoPredictionResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [confidence, setConfidence] = useState(0.4);
  const inputRef = useRef<HTMLInputElement>(null);

  const handleFile = (f: File) => {
    setFile(f);
    setResult(null);
    setError(null);
  };

  const analyze = async () => {
    if (!file) return;
    setLoading(true);
    setError(null);
    try {
      const res = await api.predictVideo(file, confidence);
      setResult(res);
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : 'Video processing failed';
      setError(msg);
    } finally {
      setLoading(false);
    }
  };

  const formatTime = (sec: number) => {
    const m = Math.floor(sec / 60);
    const s = Math.floor(sec % 60);
    return `${m.toString().padStart(2, '0')}:${s.toString().padStart(2, '0')}`;
  };

  return (
    <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-10">
      <div className="mb-8">
        <h1 className="section-title">Video Analysis</h1>
        <p className="section-subtitle">Process video files frame-by-frame for road defect detection</p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Left: Upload & Controls */}
        <div className="space-y-6">
          <div
            className="upload-zone"
            onClick={() => inputRef.current?.click()}
            onDrop={e => { e.preventDefault(); const f = e.dataTransfer.files[0]; if (f) handleFile(f); }}
            onDragOver={e => e.preventDefault()}
          >
            <input
              ref={inputRef}
              type="file"
              accept="video/*"
              className="hidden"
              onChange={e => e.target.files?.[0] && handleFile(e.target.files[0])}
            />
            <Video className="w-10 h-10 text-slate-500" />
            <p className="text-sm text-slate-400">{file ? file.name : 'Drop a video or click to browse'}</p>
            <p className="text-xs text-slate-600">MP4, AVI, MOV, MKV</p>
          </div>

          {/* Confidence slider */}
          <div className="glass-card p-5">
            <label className="flex items-center justify-between mb-3">
              <span className="text-sm font-medium text-slate-300">Confidence Threshold</span>
              <span className="text-sm font-mono text-accent">{confidence.toFixed(2)}</span>
            </label>
            <input
              type="range"
              min="0.1"
              max="0.95"
              step="0.05"
              value={confidence}
              onChange={e => setConfidence(parseFloat(e.target.value))}
              className="w-full accent-blue-500"
            />
          </div>

          <button
            onClick={analyze}
            disabled={!file || loading}
            className="btn-primary w-full flex items-center justify-center gap-2"
          >
            {loading ? (
              <><Loader2 className="w-5 h-5 animate-spin" />Processing Video...</>
            ) : (
              <><Video className="w-5 h-5" />Analyze Video</>
            )}
          </button>

          {loading && (
            <div className="glass-card p-4 text-center">
              <div className="w-full h-1.5 bg-surface-300 rounded-full overflow-hidden mb-3">
                <div className="h-full bg-accent rounded-full animate-pulse" style={{ width: '60%' }} />
              </div>
              <p className="text-xs text-slate-500">Processing frames... This may take a moment.</p>
            </div>
          )}

          {error && (
            <div className="flex items-start gap-2 p-4 rounded-lg bg-red-500/10 border border-red-500/20 text-red-400 text-sm">
              <AlertCircle className="w-4 h-4 mt-0.5 flex-shrink-0" />
              <span>{error}</span>
            </div>
          )}
        </div>

        {/* Right: Results */}
        <div className="lg:col-span-2 space-y-6">
          {result ? (
            <>
              {/* Summary metrics */}
              <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
                <MetricCard label="Total Frames" value={result.total_frames} />
                <MetricCard label="Processed" value={result.processed_frames} />
                <MetricCard label="Total Detections" value={result.detections_total} />
                <MetricCard label="Avg Confidence" value={(result.average_confidence * 100).toFixed(1)} unit="%" />
              </div>

              <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
                <MetricCard label="Processing FPS" value={result.processing_fps.toFixed(1)} icon={<Zap className="w-4 h-4" />} />
                <MetricCard label="Avg Latency" value={result.average_latency_ms.toFixed(1)} unit="ms" icon={<Clock className="w-4 h-4" />} />
                <MetricCard label="Total Time" value={result.total_processing_time_sec.toFixed(1)} unit="s" />
                <MetricCard label="Video Duration" value={result.video_duration_sec.toFixed(1)} unit="s" />
              </div>

              {/* Event timeline */}
              {result.events.length > 0 && (
                <div className="glass-card p-6">
                  <h3 className="text-sm font-semibold text-slate-300 mb-4">Event Timeline</h3>
                  <div className="space-y-3 max-h-80 overflow-y-auto pr-2">
                    {result.events.map((event, i) => (
                      <div key={i} className={`flex items-start gap-3 pl-3 border-l-2 ${EVENT_COLORS[event.event_type] || 'border-slate-600 text-slate-400'}`}>
                        <div className="flex-1 min-w-0">
                          <div className="flex items-center gap-2 mb-0.5">
                            <span className="text-xs font-mono text-slate-500">
                              {formatTime(event.timestamp_sec)}
                            </span>
                            <span className="text-xs px-1.5 py-0.5 rounded bg-surface-300 font-medium">
                              {event.event_type.replace('_', ' ')}
                            </span>
                          </div>
                          <p className="text-sm text-slate-400">{event.description}</p>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              )}

              {/* Download processed video */}
              {result.processed_video_id && (
                <a
                  href={api.getProcessedVideoUrl(result.processed_video_id)}
                  download
                  className="btn-secondary w-full flex items-center justify-center gap-2"
                >
                  <Download className="w-4 h-4" />
                  Download Processed Video
                </a>
              )}

              <div className="text-xs text-slate-600 text-center">
                {result.model_backend} • {result.video_fps.toFixed(1)} FPS source video
              </div>
            </>
          ) : (
            <div className="glass-card flex items-center justify-center min-h-[400px]">
              <div className="text-center text-slate-500">
                <Video className="w-12 h-12 mx-auto mb-3 opacity-30" />
                <p className="text-sm">Upload a video and click Analyze</p>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
