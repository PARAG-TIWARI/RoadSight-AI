import { useState, useEffect } from 'react';
import { Loader2, Zap, Play } from 'lucide-react';
import { api } from '../services/api';
import type { BenchmarkResponse } from '../types';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Cell } from 'recharts';

const COLORS = ['#3b82f6', '#a855f7', '#22c55e', '#f59e0b'];

export default function BenchmarksPage() {
  const [data, setData] = useState<BenchmarkResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [iterations, setIterations] = useState(10);

  const runBenchmark = async () => {
    setLoading(true);
    setError(null);
    try {
      const res = await api.benchmark(iterations);
      setData(res);
    } catch {
      setError('Failed to run benchmark. Ensure backend is running.');
    } finally {
      setLoading(false);
    }
  };

  // Load cached benchmark on mount
  useEffect(() => {
    api.benchmark(5).then(setData).catch(() => {});
  }, []);

  const fpsData = data?.results.map(r => ({ name: r.backend, fps: r.fps })) || [];
  const latencyData = data?.results.map(r => ({ name: r.backend, latency: r.avg_latency_ms })) || [];

  return (
    <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-10">
      <div className="flex items-start justify-between mb-8">
        <div>
          <h1 className="section-title">Performance Benchmarks</h1>
          <p className="section-subtitle">Real measured inference performance — no fabricated numbers</p>
        </div>
        <div className="flex items-center gap-3">
          <div className="glass-card flex items-center gap-2 px-3 py-2">
            <span className="text-xs text-slate-500">Iterations:</span>
            <input
              type="number"
              min={1}
              max={100}
              value={iterations}
              onChange={e => setIterations(Math.max(1, parseInt(e.target.value) || 1))}
              className="w-14 bg-transparent text-sm text-white font-mono text-center border-b border-white/10 focus:outline-none focus:border-accent"
            />
          </div>
          <button onClick={runBenchmark} disabled={loading} className="btn-primary flex items-center gap-2 text-sm py-2.5">
            {loading ? <Loader2 className="w-4 h-4 animate-spin" /> : <Play className="w-4 h-4" />}
            {loading ? 'Running...' : 'Run Benchmark'}
          </button>
        </div>
      </div>

      {error && (
        <div className="mb-6 p-4 rounded-lg bg-red-500/10 border border-red-500/20 text-red-400 text-sm">
          {error}
        </div>
      )}

      {data && (
        <>
          {/* Comparison cards */}
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-8">
            {data.results.map((r, i) => (
              <div key={r.backend} className="glass-card p-6">
                <div className="flex items-center gap-3 mb-5">
                  <div className={`w-3 h-3 rounded-full`} style={{ backgroundColor: COLORS[i % COLORS.length] }} />
                  <h3 className="text-lg font-semibold text-white">{r.backend}</h3>
                  <span className="ml-auto text-xs text-slate-500">{r.model_format} • {r.input_resolution}</span>
                </div>
                <div className="grid grid-cols-2 gap-4">
                  <div>
                    <p className="metric-label">Avg Latency</p>
                    <p className="text-2xl font-bold text-white">{r.avg_latency_ms.toFixed(1)}<span className="text-sm text-slate-500 ml-1">ms</span></p>
                  </div>
                  <div>
                    <p className="metric-label">FPS</p>
                    <p className="text-2xl font-bold text-white">{r.fps.toFixed(1)}</p>
                  </div>
                  <div>
                    <p className="metric-label">CPU</p>
                    <p className="text-lg font-semibold text-slate-300">{r.cpu_percent.toFixed(1)}%</p>
                  </div>
                  <div>
                    <p className="metric-label">Memory</p>
                    <p className="text-lg font-semibold text-slate-300">{r.memory_mb.toFixed(0)}<span className="text-sm text-slate-500 ml-1">MB</span></p>
                  </div>
                </div>
                <div className="mt-4 pt-4 border-t border-white/[0.06] flex justify-between text-xs text-slate-500">
                  <span>Min: {r.min_latency_ms.toFixed(1)}ms</span>
                  <span>Max: {r.max_latency_ms.toFixed(1)}ms</span>
                  <span>{r.iterations} iterations</span>
                </div>
              </div>
            ))}
          </div>

          {/* Charts */}
          {data.results.length > 0 && (
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-8">
              {/* FPS chart */}
              <div className="glass-card p-6">
                <h3 className="text-sm font-semibold text-slate-300 mb-4">Throughput (FPS)</h3>
                <div className="h-48">
                  <ResponsiveContainer width="100%" height="100%">
                    <BarChart data={fpsData}>
                      <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" />
                      <XAxis dataKey="name" tick={{ fill: '#64748b', fontSize: 11 }} axisLine={{ stroke: 'rgba(255,255,255,0.06)' }} />
                      <YAxis tick={{ fill: '#64748b', fontSize: 11 }} axisLine={{ stroke: 'rgba(255,255,255,0.06)' }} />
                      <Tooltip contentStyle={{ background: '#12121a', border: '1px solid rgba(255,255,255,0.06)', borderRadius: '8px', fontSize: 12 }} />
                      <Bar dataKey="fps" radius={[4, 4, 0, 0]}>
                        {fpsData.map((_, i) => <Cell key={i} fill={COLORS[i % COLORS.length]} />)}
                      </Bar>
                    </BarChart>
                  </ResponsiveContainer>
                </div>
              </div>

              {/* Latency chart */}
              <div className="glass-card p-6">
                <h3 className="text-sm font-semibold text-slate-300 mb-4">Average Latency (ms)</h3>
                <div className="h-48">
                  <ResponsiveContainer width="100%" height="100%">
                    <BarChart data={latencyData}>
                      <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" />
                      <XAxis dataKey="name" tick={{ fill: '#64748b', fontSize: 11 }} axisLine={{ stroke: 'rgba(255,255,255,0.06)' }} />
                      <YAxis tick={{ fill: '#64748b', fontSize: 11 }} axisLine={{ stroke: 'rgba(255,255,255,0.06)' }} />
                      <Tooltip contentStyle={{ background: '#12121a', border: '1px solid rgba(255,255,255,0.06)', borderRadius: '8px', fontSize: 12 }} />
                      <Bar dataKey="latency" radius={[4, 4, 0, 0]}>
                        {latencyData.map((_, i) => <Cell key={i} fill={COLORS[i % COLORS.length]} />)}
                      </Bar>
                    </BarChart>
                  </ResponsiveContainer>
                </div>
              </div>
            </div>
          )}

          {/* System info */}
          <div className="glass-card p-6">
            <h3 className="text-sm font-semibold text-slate-300 mb-4">Benchmark Environment</h3>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
              {Object.entries(data.system_info).map(([key, value]) => (
                <div key={key}>
                  <p className="text-xs text-slate-500 mb-0.5">{key.replace(/_/g, ' ')}</p>
                  <p className="text-sm text-white font-mono">{String(value)}</p>
                </div>
              ))}
            </div>
            <div className="mt-4 pt-3 border-t border-white/[0.06] text-xs text-slate-600">
              Benchmark timestamp: {data.timestamp}
            </div>
          </div>
        </>
      )}

      {!data && !loading && (
        <div className="glass-card flex items-center justify-center min-h-[300px]">
          <div className="text-center text-slate-500">
            <Zap className="w-12 h-12 mx-auto mb-3 opacity-30" />
            <p className="text-sm">Click "Run Benchmark" to measure inference performance</p>
          </div>
        </div>
      )}
    </div>
  );
}
