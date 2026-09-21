import { Github, ExternalLink } from 'lucide-react';

export function Footer() {
  return (
    <footer className="border-t border-white/[0.06] bg-surface-50/50 mt-auto">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        <div className="flex flex-col md:flex-row items-center justify-between gap-4">
          <div className="text-sm text-slate-500">
            <span className="font-semibold text-slate-400">RoadSight AI</span>
            {' '}— Intelligent Road Defect Detection & Video Analytics
          </div>
          <div className="flex items-center gap-4">
            <a
              href="https://github.com/Musa-Qureshi-01/RoadWatch"
              target="_blank"
              rel="noopener noreferrer"
              className="flex items-center gap-1.5 text-sm text-slate-500 hover:text-accent transition-colors"
            >
              <Github className="w-4 h-4" />
              GitHub
            </a>
            <a
              href="/api-docs"
              className="flex items-center gap-1.5 text-sm text-slate-500 hover:text-accent transition-colors"
            >
              <ExternalLink className="w-4 h-4" />
              API Docs
            </a>
          </div>
        </div>
      </div>
    </footer>
  );
}
