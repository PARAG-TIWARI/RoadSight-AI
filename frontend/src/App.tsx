import { Routes, Route } from 'react-router-dom';
import { Suspense, lazy } from 'react';
import { Navbar } from './components/layout/Navbar';
import { Footer } from './components/layout/Footer';

const HomePage = lazy(() => import('./pages/HomePage'));
const ImageAnalysisPage = lazy(() => import('./pages/ImageAnalysisPage'));
const VideoAnalysisPage = lazy(() => import('./pages/VideoAnalysisPage'));
const BenchmarksPage = lazy(() => import('./pages/BenchmarksPage'));
const ModelInfoPage = lazy(() => import('./pages/ModelInfoPage'));
const ApiDocsPage = lazy(() => import('./pages/ApiDocsPage'));

function PageLoader() {
  return (
    <div className="flex items-center justify-center min-h-[60vh]">
      <div className="flex flex-col items-center gap-3">
        <div className="w-8 h-8 border-2 border-accent/30 border-t-accent rounded-full animate-spin" />
        <span className="text-sm text-slate-500">Loading...</span>
      </div>
    </div>
  );
}

export default function App() {
  return (
    <div className="min-h-screen flex flex-col">
      <Navbar />
      <main className="flex-1">
        <Suspense fallback={<PageLoader />}>
          <Routes>
            <Route path="/" element={<HomePage />} />
            <Route path="/analyze/image" element={<ImageAnalysisPage />} />
            <Route path="/analyze/video" element={<VideoAnalysisPage />} />
            <Route path="/benchmarks" element={<BenchmarksPage />} />
            <Route path="/model" element={<ModelInfoPage />} />
            <Route path="/api-docs" element={<ApiDocsPage />} />
          </Routes>
        </Suspense>
      </main>
      <Footer />
    </div>
  );
}
