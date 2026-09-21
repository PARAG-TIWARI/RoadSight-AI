import { ReactNode } from 'react';

interface MetricCardProps {
  label: string;
  value: string | number;
  unit?: string;
  icon?: ReactNode;
  trend?: 'up' | 'down' | 'neutral';
  className?: string;
}

export function MetricCard({ label, value, unit, icon, className = '' }: MetricCardProps) {
  return (
    <div className={`glass-card p-5 ${className}`}>
      <div className="flex items-start justify-between mb-3">
        <span className="metric-label">{label}</span>
        {icon && <span className="text-accent">{icon}</span>}
      </div>
      <div className="flex items-baseline gap-1.5">
        <span className="metric-value">{value}</span>
        {unit && <span className="text-sm text-slate-500 font-medium">{unit}</span>}
      </div>
    </div>
  );
}
