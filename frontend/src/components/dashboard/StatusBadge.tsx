interface StatusBadgeProps {
  status: 'online' | 'offline' | 'loading';
  label?: string;
}

export function StatusBadge({ status, label }: StatusBadgeProps) {
  const config = {
    online: { dot: 'bg-emerald-500', text: 'text-emerald-400', glow: 'shadow-[0_0_8px_rgba(34,197,94,0.5)]' },
    offline: { dot: 'bg-red-500', text: 'text-red-400', glow: 'shadow-[0_0_8px_rgba(239,68,68,0.5)]' },
    loading: { dot: 'bg-amber-500 animate-pulse', text: 'text-amber-400', glow: '' },
  };
  const c = config[status];

  return (
    <div className="flex items-center gap-2">
      <div className={`w-2 h-2 rounded-full ${c.dot} ${c.glow}`} />
      <span className={`text-xs font-medium ${c.text}`}>
        {label || status.charAt(0).toUpperCase() + status.slice(1)}
      </span>
    </div>
  );
}
