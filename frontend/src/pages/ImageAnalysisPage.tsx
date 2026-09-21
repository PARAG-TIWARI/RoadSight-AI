import { useState, useCallback, useRef } from 'react';
import { Upload, Image as ImageIcon, Loader2, Download, AlertCircle, ChevronDown, ChevronUp } from 'lucide-react';
import { api } from '../services/api';
import { MetricCard } from '../components/dashboard/MetricCard';
import type { ImagePredictionResponse } from '../types';

export default function ImageAnalysisPage() {
  const [file, setFile] = useState<File | null>(null);
  const [preview, setPreview] = useState<string | null>(null);
  const [result, setResult] = useState<ImagePredictionResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [confidence, setConfidence] = useState(0.4);
  const [activeView, setActiveView] = useState<'annotated' | 'overlay' | 'mask'>('annotated');
  const [showDetails, setShowDetails] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);

  const handleFile = useCallback((f: File) => {
    setFile(f);
    setPreview(URL.createObjectURL(f));
    setResult(null);
    setError(null);
  }, []);

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    const f = e.dataTransfer.files[0];
    if (f && f.type.startsWith('image/')) handleFile(f);
  }, [handleFile]);

  const analyze = async () => {
    if (!file) return;
    setLoading(true);
    setError(null);
    try {
      const res = await api.predictImage(file, confidence);
      setResult(res);
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : 'Analysis failed';
      setError(msg);
    } finally {
      setLoading(false);
    }
  };

  const downloadResult = () => {
    if (!result?.annotated_image) return;
    const a = document.createElement('a');
    a.href = `data:image/png;base64,${result.annotated_image}`;
    a.download = 'roadsight_detection.png';
    a.click();
  };

  return (
    <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-10">
      <div className="mb-8">
        <h1 className="section-title">Image Analysis</h1>
        <p className="section-subtitle">Upload an image for pothole detection and segmentation</p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Left: Upload & Controls */}
        <div className="space-y-6">
          {/* Upload zone */}
          <div
            className="upload-zone"
            onDrop={handleDrop}
            onDragOver={e => e.preventDefault()}
            onClick={() => inputRef.current?.click()}
          >
            <input
              ref={inputRef}
              type="file"
              accept="image/*"
              className="hidden"
              onChange={e => e.target.files?.[0] && handleFile(e.target.files[0])}
            />
            {preview ? (
              <img src={preview} alt="Preview" className="max-h-64 rounded-lg object-contain" />
            ) : (
              <>
                <Upload className="w-10 h-10 text-slate-500" />
                <p className="text-sm text-slate-400">Drop an image here or click to browse</p>
                <p className="text-xs text-slate-600">JPEG, PNG, BMP, WebP</p>
              </>
            )}
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
            <div className="flex justify-between text-xs text-slate-600 mt-1">
              <span>0.10</span>
              <span>0.95</span>
            </div>
          </div>

          {/* Analyze button */}
          <button
            onClick={analyze}
            disabled={!file || loading}
            className="btn-primary w-full flex items-center justify-center gap-2"
          >
            {loading ? (
              <><Loader2 className="w-5 h-5 animate-spin" />Analyzing...</>
            ) : (
              <><ImageIcon className="w-5 h-5" />Analyze Image</>
            )}
          </button>

          {error && (
            <div className="flex items-center gap-2 p-4 rounded-lg bg-red-500/10 border border-red-500/20 text-red-400 text-sm">
              <AlertCircle className="w-4 h-4 flex-shrink-0" />
              {error}
            </div>
          )}
        </div>

        {/* Right: Results */}
        <div className="space-y-6">
          {result ? (
            <>
              {/* View switcher */}
              <div className="glass-card p-1 flex gap-1">
                {(['annotated', 'overlay', 'mask'] as const).map(view => (
                  <button
                    key={view}
                    onClick={() => setActiveView(view)}
                    className={`flex-1 py-2 px-3 rounded-lg text-sm font-medium transition-colors ${
                      activeView === view
                        ? 'bg-accent/10 text-accent'
                        : 'text-slate-400 hover:text-white'
                    }`}
                  >
                    {view === 'annotated' ? 'Detections' : view === 'overlay' ? 'Segmentation' : 'Mask'}
                  </button>
                ))}
              </div>

              {/* Result image */}
              <div className="glass-card p-4">
                {activeView === 'annotated' && result.annotated_image && (
                  <img src={`data:image/png;base64,${result.annotated_image}`} alt="Detections" className="w-full rounded-lg" />
                )}
                {activeView === 'overlay' && result.mask_overlay && (
                  <img src={`data:image/png;base64,${result.mask_overlay}`} alt="Segmentation" className="w-full rounded-lg" />
                )}
                {activeView === 'mask' && result.mask_image && (
                  <img src={`data:image/png;base64,${result.mask_image}`} alt="Mask" className="w-full rounded-lg" />
                )}
                {!result.annotated_image && !result.mask_overlay && !result.mask_image && (
                  <div className="text-center py-12 text-slate-500">
                    No detections found in this image
                  </div>
                )}
              </div>

              {/* Metrics */}
              <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
                <MetricCard label="Detections" value={result.detection_count} />
                <MetricCard label="Avg Confidence" value={(result.average_confidence * 100).toFixed(1)} unit="%" />
                <MetricCard label="Latency" value={result.processing_time_ms.toFixed(1)} unit="ms" />
                <MetricCard label="FPS" value={result.fps.toFixed(1)} />
              </div>

              {/* Detection details */}
              {result.detections.length > 0 && (
                <div className="glass-card">
                  <button
                    onClick={() => setShowDetails(!showDetails)}
                    className="w-full flex items-center justify-between p-4 text-sm font-medium text-slate-300 hover:text-white"
                  >
                    Detection Details ({result.detections.length})
                    {showDetails ? <ChevronUp className="w-4 h-4" /> : <ChevronDown className="w-4 h-4" />}
                  </button>
                  {showDetails && (
                    <div className="px-4 pb-4">
                      <div className="overflow-x-auto">
                        <table className="w-full text-sm">
                          <thead>
                            <tr className="border-b border-white/[0.06]">
                              <th className="text-left py-2 text-slate-500 font-medium">#</th>
                              <th className="text-left py-2 text-slate-500 font-medium">Confidence</th>
                              <th className="text-left py-2 text-slate-500 font-medium">Area</th>
                              <th className="text-left py-2 text-slate-500 font-medium">Severity</th>
                            </tr>
                          </thead>
                          <tbody>
                            {result.detections.map((d, i) => (
                              <tr key={i} className="border-b border-white/[0.03]">
                                <td className="py-2 text-slate-400">{i + 1}</td>
                                <td className="py-2 font-mono text-white">{(d.confidence * 100).toFixed(1)}%</td>
                                <td className="py-2 font-mono text-slate-300">{d.area_pixels.toLocaleString()}px</td>
                                <td className="py-2">
                                  <span className={`px-2 py-0.5 rounded text-xs font-medium ${
                                    d.severity === 'critical' ? 'bg-red-500/20 text-red-400' :
                                    d.severity === 'high' ? 'bg-orange-500/20 text-orange-400' :
                                    d.severity === 'medium' ? 'bg-amber-500/20 text-amber-400' :
                                    'bg-emerald-500/20 text-emerald-400'
                                  }`}>
                                    {d.severity}
                                  </span>
                                </td>
                              </tr>
                            ))}
                          </tbody>
                        </table>
                      </div>
                    </div>
                  )}
                </div>
              )}

              {/* Download */}
              {result.annotated_image && (
                <button onClick={downloadResult} className="btn-secondary w-full flex items-center justify-center gap-2">
                  <Download className="w-4 h-4" />
                  Download Result
                </button>
              )}

              {/* Meta */}
              <div className="text-xs text-slate-600 text-center">
                {result.image_width}×{result.image_height} • {result.model_backend}
              </div>
            </>
          ) : (
            <div className="glass-card flex items-center justify-center min-h-[400px]">
              <div className="text-center text-slate-500">
                <ImageIcon className="w-12 h-12 mx-auto mb-3 opacity-30" />
                <p className="text-sm">Upload an image and click Analyze</p>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
