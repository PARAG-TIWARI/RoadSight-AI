import { useState, useEffect } from 'react';
import { Loader2, Server, Box, Cpu } from 'lucide-react';
import { api } from '../services/api';
import type { SystemInfoResponse } from '../types';

export default function ModelInfoPage() {
  const [data, setData] = useState<SystemInfoResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    api.systemInfo()
      .then(setData)
      .catch(() => setError('Failed to load system info. Ensure backend is running.'))
      .finally(() => setLoading(false));
  }, []);

  if (loading) {
    return (
      <div className="flex items-center justify-center min-h-[60vh]">
        <Loader2 className="w-6 h-6 text-accent animate-spin" />
      </div>
    );
  }

  if (error || !data) {
    return (
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-10">
        <div className="glass-card p-8 text-center text-slate-400">
          <Server className="w-10 h-10 mx-auto mb-3 opacity-30" />
          <p>{error || 'Unable to load system info'}</p>
        </div>
      </div>
    );
  }

  const { model_info } = data;

  return (
    <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 py-10">
      <div className="mb-8">
        <h1 className="section-title">Model & System</h1>
        <p className="section-subtitle">Loaded model metadata and runtime environment</p>
      </div>

      {/* Model Info */}
      <div className="glass-card p-6 mb-6">
        <div className="flex items-center gap-3 mb-5">
          <div className="w-10 h-10 rounded-lg bg-accent/10 flex items-center justify-center">
            <Box className="w-5 h-5 text-accent" />
          </div>
          <div>
            <h2 className="text-lg font-semibold text-white">{model_info.name}</h2>
            <p className="text-xs text-slate-500">v{model_info.version}</p>
          </div>
        </div>

        <div className="grid grid-cols-2 md:grid-cols-3 gap-y-5 gap-x-6">
          <InfoRow label="Format" value={model_info.format.toUpperCase()} />
          <InfoRow label="Input Size" value={`${model_info.input_size}×${model_info.input_size}`} />
          <InfoRow label="Classes" value={model_info.classes.join(', ')} />
          <InfoRow label="Backend" value={model_info.backend} />
          <InfoRow label="Provider" value={model_info.provider} />
          <InfoRow label="Load Time" value={`${model_info.load_time_ms.toFixed(1)} ms`} />
        </div>
      </div>

      {/* System Info */}
      <div className="glass-card p-6">
        <div className="flex items-center gap-3 mb-5">
          <div className="w-10 h-10 rounded-lg bg-purple-500/10 flex items-center justify-center">
            <Cpu className="w-5 h-5 text-purple-400" />
          </div>
          <div>
            <h2 className="text-lg font-semibold text-white">Runtime Environment</h2>
          </div>
        </div>

        <div className="grid grid-cols-2 md:grid-cols-3 gap-y-5 gap-x-6">
          <InfoRow label="API Version" value={data.api_version} />
          <InfoRow label="Backend Version" value={data.backend_version} />
          <InfoRow label="Python" value={data.python_version} />
          <InfoRow label="OpenCV" value={data.opencv_version} />
          <InfoRow label="NumPy" value={data.numpy_version} />
        </div>
      </div>
    </div>
  );
}

function InfoRow({ label, value }: { label: string; value: string }) {
  return (
    <div>
      <p className="text-xs text-slate-500 mb-1">{label}</p>
      <p className="text-sm font-mono text-white">{value}</p>
    </div>
  );
}
