import React, { useState, useEffect } from 'react';
import Navbar from './components/Navbar';
import MediaSelector from './components/MediaSelector';
import Uploader from './components/Uploader';
import ResultView from './components/ResultView';
import HistoryTable from './components/HistoryTable';
import { getApiUrl } from './config';
import {
  analyzeMediaClientSide,
  saveLocalCase,
  getLocalCases,
  generatePrintableReportUrl,
  generateImageOverlay,
} from './utils/forensicsEngine';

export default function App() {
  const [activeModality, setActiveModality] = useState('image');
  const [healthInfo, setHealthInfo] = useState(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [currentResult, setCurrentResult] = useState(null);
  const [cases, setCases] = useState([]);
  const [errorMessage, setErrorMessage] = useState(null);

  const fetchHealth = async () => {
    try {
      const res = await fetch(getApiUrl('/api/health'));
      const contentType = res.headers.get('content-type') || '';
      if (res.ok && contentType.includes('application/json')) {
        const data = await res.json();
        setHealthInfo(data);
      } else {
        setHealthInfo({
          status: 'healthy',
          device_info: { active_device: 'Edge Forensics Engine' },
        });
      }
    } catch (e) {
      setHealthInfo({
        status: 'healthy',
        device_info: { active_device: 'Edge Forensics Engine' },
      });
    }
  };

  const fetchCases = async () => {
    const local = getLocalCases();
    try {
      const res = await fetch(getApiUrl('/api/cases'));
      const contentType = res.headers.get('content-type') || '';
      if (res.ok && contentType.includes('application/json')) {
        const apiCases = await res.json();
        const map = new Map();
        [...apiCases, ...local].forEach((c) => {
          if (c && c.case_id && !map.has(c.case_id)) {
            map.set(c.case_id, c);
          }
        });
        setCases(Array.from(map.values()));
        return;
      }
    } catch (e) {
      // API not available, use local cases
    }
    setCases(local);
  };

  useEffect(() => {
    fetchHealth();
    fetchCases();
  }, []);

  const handleAnalyze = async (file, modality) => {
    setIsAnalyzing(true);
    setErrorMessage(null);
    setCurrentResult(null);

    let resultData = null;

    // 1. Attempt API analysis (if backend or Vercel serverless is live)
    try {
      const formData = new FormData();
      formData.append('file', file);

      const res = await fetch(getApiUrl(`/api/analyze/${modality}`), {
        method: 'POST',
        body: formData,
      });

      const contentType = res.headers.get('content-type') || '';
      if (res.ok && contentType.includes('application/json')) {
        resultData = await res.json();
      } else if (!res.ok && contentType.includes('application/json')) {
        const errJson = await res.json().catch(() => null);
        if (errJson?.detail) {
          throw new Error(errJson.detail);
        }
      }
    } catch (apiErr) {
      console.warn('API call encountered error, engaging client-side forensics engine:', apiErr);
    }

    // 2. If API was unreachable, returned non-JSON, or empty, run high-fidelity client-side forensics engine
    try {
      if (!resultData) {
        resultData = await analyzeMediaClientSide(file, modality);
      }

      // Ensure affected region overlay is always generated and attached for images
      if (modality === 'image') {
        const currentOverlay = resultData.analysis_details?.explanation?.overlay_path;
        const isDataUrl = currentOverlay && (currentOverlay.startsWith('data:image') || currentOverlay.startsWith('blob:'));
        if (!isDataUrl) {
          const isManipulated = resultData.verdict === 'manipulated';
          const bbox = resultData.analysis_details?.bounding_box;
          const overlayUrl = await generateImageOverlay(file, isManipulated, bbox);
          if (overlayUrl) {
            if (!resultData.analysis_details) resultData.analysis_details = {};
            if (!resultData.analysis_details.explanation) resultData.analysis_details.explanation = {};
            resultData.analysis_details.explanation.overlay_path = overlayUrl;
          }
        }
      }

      // Attach printable report URL for instant download (which embeds the affected region)
      resultData.printable_report_url = generatePrintableReportUrl(resultData);

      setCurrentResult(resultData);
      saveLocalCase(resultData);
      fetchCases();
    } catch (err) {
      console.error('Forensic analysis error:', err);
      setErrorMessage(err.message || 'Forensic analysis could not be completed.');
    } finally {
      setIsAnalyzing(false);
    }
  };

  return (
    <div style={{ paddingBottom: '60px' }}>
      <Navbar healthInfo={healthInfo} />

      <main style={{ maxWidth: '1200px', margin: '0 auto', padding: '0 20px' }}>
        {/* Intro Hero */}
        <div style={{ textAlign: 'center', margin: '32px 0 40px 0' }}>
          <h2
            style={{
              fontSize: '2.4rem',
              fontWeight: 800,
              background: 'linear-gradient(135deg, #ffffff, #94a3b8)',
              WebkitBackgroundClip: 'text',
              WebkitTextFillColor: 'transparent',
              marginBottom: '12px',
            }}
          >
            Multimodal AI Deepfake Detection & Media Forensics
          </h2>
          <p
            style={{
              color: 'var(--text-secondary)',
              fontSize: '1.05rem',
              maxWidth: '720px',
              margin: '0 auto',
            }}
          >
            VeriFace integrates Vision Transformers (FA-ViT), Style Latent Flows (StyleGRU), and Contrastive Audio Analysis (CLAD) to detect synthetic manipulations across Images, Videos, and Audio.
          </p>
        </div>

        {/* Modality Selector */}
        <MediaSelector
          activeModality={activeModality}
          onSelectModality={(m) => {
            setActiveModality(m);
            setCurrentResult(null);
            setErrorMessage(null);
          }}
        />

        {/* Error Alert */}
        {errorMessage && (
          <div
            style={{
              background: 'rgba(244, 63, 94, 0.15)',
              border: '1px solid var(--fake-rose)',
              color: 'var(--fake-rose)',
              padding: '16px',
              borderRadius: '12px',
              marginBottom: '24px',
              textAlign: 'center',
            }}
          >
            <b>Analysis Failure:</b> {errorMessage}
          </div>
        )}

        {/* Upload Box */}
        <Uploader
          modality={activeModality}
          onAnalyze={handleAnalyze}
          isAnalyzing={isAnalyzing}
        />

        {/* Result Dashboard */}
        <ResultView result={currentResult} />

        {/* Case History Table */}
        <HistoryTable cases={cases} onSelectCase={(c) => setCurrentResult(c)} />
      </main>
    </div>
  );
}
