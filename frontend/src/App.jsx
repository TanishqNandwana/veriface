import React, { useState, useEffect } from 'react';
import Navbar from './components/Navbar';
import MediaSelector from './components/MediaSelector';
import Uploader from './components/Uploader';
import ResultView from './components/ResultView';
import HistoryTable from './components/HistoryTable';
import { getApiUrl } from './config';

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
      if (res.ok) {
        const data = await res.json();
        setHealthInfo(data);
      }
    } catch (e) {
      console.warn('Health check fetch failed:', e);
    }
  };

  const fetchCases = async () => {
    try {
      const res = await fetch(getApiUrl('/api/cases'));
      if (res.ok) {
        const data = await res.json();
        setCases(data);
      }
    } catch (e) {
      console.warn('Cases fetch failed:', e);
    }
  };

  useEffect(() => {
    fetchHealth();
    fetchCases();
  }, []);

  const handleAnalyze = async (file, modality) => {
    setIsAnalyzing(true);
    setErrorMessage(null);
    setCurrentResult(null);

    const formData = new FormData();
    formData.append('file', file);

    try {
      const res = await fetch(getApiUrl(`/api/analyze/${modality}`), {
        method: 'POST',
        body: formData,
      });

      if (!res.ok) {
        const errData = await res.json();
        throw new Error(errData.detail || 'Analysis request failed.');
      }

      const data = await res.json();
      setCurrentResult(data);
      fetchCases(); // Refresh case history
    } catch (err) {
      setErrorMessage(err.message);
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
          <h2 style={{ fontSize: '2.4rem', fontWeight: 800, background: 'linear-gradient(135deg, #ffffff, #94a3b8)', WebkitBackgroundClip: 'text', WebkitTextFillColor: 'transparent', marginBottom: '12px' }}>
            Multimodal AI Deepfake Detection & Media Forensics
          </h2>
          <p style={{ color: 'var(--text-secondary)', fontSize: '1.05rem', maxWidth: '720px', margin: '0 auto' }}>
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
          <div style={{ background: 'rgba(244, 63, 94, 0.15)', border: '1px solid var(--fake-rose)', color: 'var(--fake-rose)', padding: '16px', borderRadius: '12px', marginBottom: '24px', textAlign: 'center' }}>
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
