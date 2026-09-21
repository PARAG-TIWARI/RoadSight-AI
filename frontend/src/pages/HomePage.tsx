import { Link } from 'react-router-dom';
import { useEffect, useState } from 'react';
import { Camera, Video, Activity, Cpu, ChevronRight, ArrowRight, Zap, BarChart3, Eye, Box } from 'lucide-react';
import { StatusBadge } from '../components/dashboard/StatusBadge';
import { api } from '../services/api';
import type { HealthResponse, ModelInfo } from '../types';


const pipelineSteps = [
  { icon: <Camera className="w-4 h-4" />, label: 'Input' },
  { icon: <Eye className="w-4 h-4" />, label: 'YOLO' },
  { icon: <Box className="w-4 h-4" />, label: 'Segmentation' },
  { icon: <Activity className="w-4 h-4" />, label: 'Analysis' },
  { icon: <BarChart3 className="w-4 h-4" />, label: 'Metrics' },
];

export default function HomePage() {
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [model, setModel] = useState<ModelInfo | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    Promise.allSettled([api.health(), api.modelInfo()])
      .then(([h, m]) => {
        if (h.status === 'fulfilled') setHealth(h.value);
        if (m.status === 'fulfilled') setModel(m.value);
      })
      .finally(() => setLoading(false));
  }, []);

  const isOnline = health?.status === 'ok';

  return (
    <div className="min-h-screen">
      {/* Hero */}
      <section className="relative overflow-hidden">
        {/* Background gradient */}
        <div className="absolute inset-0 bg-gradient-to-br from-accent/5 via-transparent to-purple-500/5" />
        <div className="absolute top-1/4 left-1/4 w-96 h-96 bg-accent/5 rounded-full blur-3xl" />
        <div className="absolute bottom-1/4 right-1/4 w-64 h-64 bg-purple-500/5 rounded-full blur-3xl" />

        <div className="relative max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 pt-20 pb-16">
          {/* Status bar */}
          <div className="flex items-center justify-center mb-8">
            <div className="glass-card px-4 py-2 flex items-center gap-3">
              <StatusBadge status={loading ? 'loading' : isOnline ? 'online' : 'offline'} label={loading ? 'Connecting...' : isOnline ? 'System Online' : 'System Offline'} />
              {model && (
                <>
                  <div className="w-px h-4 bg-white/10" />
                  <span className="text-xs text-slate-500">{model.format.toUpperCase()} • {model.backend}</span>
                </>
              )}
            </div>
          </div>

          {/* Title */}
          <div className="text-center max-w-3xl mx-auto">
            <h1 className="text-5xl sm:text-6xl lg:text-7xl font-extrabold tracking-tight mb-6">
              <span className="gradient-text">RoadSight AI</span>
            </h1>
            <p className="text-xl sm:text-2xl text-slate-300 font-light mb-3">
              AI-Powered Road Defect Detection & Video Analytics
            </p>
            <p className="text-base text-slate-500 max-w-xl mx-auto mb-10">
              Detect, segment, analyze, and benchmark road defects using modern computer vision with YOLOv8 segmentation.
            </p>

            {/* CTA buttons */}
            <div className="flex flex-col sm:flex-row items-center justify-center gap-4 mb-16">
              <Link to="/analyze/image" className="btn-primary flex items-center gap-2 text-base">
                <Camera className="w-5 h-5" />
                Analyze Image
                <ArrowRight className="w-4 h-4" />
              </Link>
              <Link to="/analyze/video" className="btn-secondary flex items-center gap-2 text-base">
                <Video className="w-5 h-5" />
                Analyze Video
              </Link>
            </div>

            {/* Pipeline visualization */}
            <div className="glass-card p-6 max-w-2xl mx-auto">
              <p className="text-xs uppercase tracking-wider text-slate-500 mb-4">Computer Vision Pipeline</p>
              <div className="flex items-center justify-center flex-wrap gap-2">
                {pipelineSteps.map((step, i) => (
                  <div key={step.label} className="flex items-center gap-2">
                    <div className="flex items-center gap-2 px-3 py-2 rounded-lg bg-surface-200/50 border border-white/[0.06]">
                      <span className="text-accent">{step.icon}</span>
                      <span className="text-sm text-slate-300">{step.label}</span>
                    </div>
                    {i < pipelineSteps.length - 1 && (
                      <ChevronRight className="w-4 h-4 text-slate-600 flex-shrink-0" />
                    )}
                  </div>
                ))}
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* Features */}
      <section className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-20">
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
          <FeatureCard
            icon={<Camera className="w-5 h-5" />}
            title="Image Analysis"
            description="Upload images for instant pothole detection with segmentation masks, bounding boxes, and severity assessment."
            link="/analyze/image"
          />
          <FeatureCard
            icon={<Video className="w-5 h-5" />}
            title="Video Analysis"
            description="Process video files frame-by-frame with temporal event tracking, detection persistence, and annotated output."
            link="/analyze/video"
          />
          <FeatureCard
            icon={<Zap className="w-5 h-5" />}
            title="ONNX Inference"
            description="Production-optimized inference with ONNX Runtime. Compare performance against PyTorch with real benchmarks."
            link="/benchmarks"
          />
          <FeatureCard
            icon={<BarChart3 className="w-5 h-5" />}
            title="Performance Benchmarks"
            description="Real measured latency, FPS, CPU, and memory usage. No fabricated numbers — only genuine measurements."
            link="/benchmarks"
          />
          <FeatureCard
            icon={<Activity className="w-5 h-5" />}
            title="Event Processing"
            description="Temporal analysis detects persistent defects, filters false positives, and generates a timeline of road events."
            link="/analyze/video"
          />
          <FeatureCard
            icon={<Cpu className="w-5 h-5" />}
            title="REST API"
            description="Clean FastAPI backend with OpenAPI documentation, Pydantic validation, and typed JSON responses."
            link="/api-docs"
          />
        </div>
      </section>

      {/* Tech stack */}
      <section className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 pb-20">
        <div className="glass-card p-8">
          <h2 className="section-title mb-6">Tech Stack</h2>
          <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-6 gap-4">
            {['Python', 'FastAPI', 'YOLOv8', 'ONNX Runtime', 'OpenCV', 'React + TypeScript'].map(tech => (
              <div key={tech} className="text-center p-3 rounded-lg bg-surface-200/30 border border-white/[0.04]">
                <span className="text-sm text-slate-300">{tech}</span>
              </div>
            ))}
          </div>
        </div>
      </section>
    </div>
  );
}

function FeatureCard({ icon, title, description, link }: {
  icon: React.ReactNode;
  title: string;
  description: string;
  link: string;
}) {
  return (
    <Link to={link} className="glass-card-interactive p-6 group">
      <div className="w-10 h-10 rounded-lg bg-accent/10 flex items-center justify-center mb-4 text-accent group-hover:bg-accent/20 transition-colors">
        {icon}
      </div>
      <h3 className="text-lg font-semibold text-white mb-2">{title}</h3>
      <p className="text-sm text-slate-400 leading-relaxed">{description}</p>
    </Link>
  );
}
