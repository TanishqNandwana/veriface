import React from 'react';
import { AlertTriangle, CheckCircle, FileText, Layers, Clock, Activity, Download } from 'lucide-react';
import { getApiUrl } from '../config';

export default function ResultView({ result }) {
  if (!result) return null;

  const isFake = result.verdict === 'manipulated';
  const confPct = (result.confidence * 100).toFixed(2);
  const details = result.analysis_details || result;
  const explanation = details.explanation || {};

  return (
    <div className="glass-panel" style={{ padding: '32px', marginBottom: '32px' }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', flexWrap: 'wrap', gap: '20px', marginBottom: '28px' }}>
        <div>
          <span style={{ fontSize: '0.8rem', textTransform: 'uppercase', letterSpacing: '1px', color: 'var(--text-muted)' }}>Case #{result.case_id}</span>
          <div style={{ display: 'flex', alignItems: 'center', gap: '16px', marginTop: '6px' }}>
            <span className={`badge-verdict ${isFake ? 'manipulated' : 'real'}`}>
              {isFake ? <AlertTriangle size={22} /> : <CheckCircle size={22} />}
              {result.verdict}
            </span>
            <div style={{ display: 'flex', flexDirection: 'column' }}>
              <span style={{ fontSize: '1.8rem', fontWeight: 800, color: isFake ? 'var(--fake-rose)' : 'var(--real-emerald)' }}>
                {confPct}% <span style={{ fontSize: '0.9rem', color: 'var(--text-secondary)', fontWeight: 500 }}>Confidence</span>
              </span>
            </div>
          </div>
        </div>

        <a
          href={getApiUrl(`/api/reports/${result.case_id}`)}
          target="_blank"
          rel="noopener noreferrer"
          className="btn-primary"
          style={{ textDecoration: 'none' }}
        >
          <Download size={18} />
          Download Forensic PDF Report
        </a>
      </div>

      {/* Metadata Grid */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: '16px', background: 'rgba(15, 23, 42, 0.4)', padding: '20px', borderRadius: '12px', marginBottom: '28px', border: '1px solid var(--border-card)' }}>
        <div>
          <span style={{ fontSize: '0.78rem', color: 'var(--text-muted)' }}>Model Architecture</span>
          <p style={{ fontWeight: 600, fontSize: '0.92rem', marginTop: '2px' }}>{result.model_name}</p>
        </div>
        <div>
          <span style={{ fontSize: '0.78rem', color: 'var(--text-muted)' }}>Processing Latency</span>
          <p style={{ fontWeight: 600, fontSize: '0.92rem', marginTop: '2px' }}>{result.processing_time_ms ? `${result.processing_time_ms.toFixed(0)} ms` : 'N/A'}</p>
        </div>
        <div>
          <span style={{ fontSize: '0.78rem', color: 'var(--text-muted)' }}>File Hash (SHA-256)</span>
          <p style={{ fontFamily: 'var(--font-mono)', fontSize: '0.8rem', marginTop: '2px', color: 'var(--text-secondary)' }}>{result.file_hash ? result.file_hash.substring(0, 16) + '...' : 'N/A'}</p>
        </div>
      </div>

      {/* Multimodal Late Fusion Info */}
      {details.multimodal_fusion && (
        <div style={{ background: 'rgba(6, 182, 212, 0.08)', border: '1px solid rgba(6, 182, 212, 0.3)', borderRadius: '12px', padding: '16px', marginBottom: '28px' }}>
          <h4 style={{ color: 'var(--accent-cyan)', fontSize: '0.95rem', marginBottom: '8px', display: 'flex', alignItems: 'center', gap: '8px' }}>
            <Layers size={18} /> Media-Aware Late Fusion Active
          </h4>
          <p style={{ fontSize: '0.85rem', color: 'var(--text-secondary)' }}>
            Visual Confidence: <b>{(details.multimodal_fusion.visual_score * 100).toFixed(1)}%</b> | Audio Track Confidence: <b>{(details.multimodal_fusion.audio_score * 100).toFixed(1)}%</b>
          </p>
        </div>
      )}

      {/* Forensic Evidence Visualizer */}
      <h3 style={{ fontSize: '1.2rem', fontWeight: 700, marginBottom: '16px' }}>Forensic Evidence & Explanation</h3>
      <p style={{ fontSize: '0.88rem', color: 'var(--text-secondary)', marginBottom: '20px' }}>{explanation.description}</p>

      {/* Image Heatmap */}
      {explanation.overlay_path && (
        <div style={{ display: 'flex', gap: '24px', flexWrap: 'wrap', marginBottom: '28px' }}>
          <div style={{ borderRadius: '12px', overflow: 'hidden', border: '1px solid var(--border-card)' }}>
            <img
              src={getApiUrl(`/${explanation.overlay_path}`)}
              alt="Grad-CAM Overlay"
              style={{ width: '280px', height: '280px', objectFit: 'cover' }}
            />
            <div style={{ background: 'rgba(0,0,0,0.6)', padding: '8px', textAlign: 'center', fontSize: '0.8rem', color: 'var(--text-secondary)' }}>
              Grad-CAM Suspicious Region Overlay
            </div>
          </div>
        </div>
      )}

      {/* Video Temporal Timeline */}
      {details.temporal_timeline && (
        <div style={{ marginBottom: '28px' }}>
          <h4 style={{ fontSize: '1rem', fontWeight: 600, marginBottom: '12px' }}>Frame-by-Frame Temporal Manipulation Scores</h4>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(80px, 1fr))', gap: '10px' }}>
            {details.temporal_timeline.map((f, i) => (
              <div key={i} style={{ background: 'rgba(15, 23, 42, 0.6)', border: '1px solid var(--border-card)', borderRadius: '8px', padding: '10px', textAlign: 'center' }}>
                <span style={{ fontSize: '0.72rem', color: 'var(--text-muted)' }}>Frame #{f.frame_index}</span>
                <p style={{ fontWeight: 700, fontSize: '0.9rem', color: f.score >= 0.5 ? 'var(--fake-rose)' : 'var(--real-emerald)', marginTop: '4px' }}>
                  {(f.score * 100).toFixed(0)}%
                </p>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Audio Segment Saliency */}
      {details.segment_saliency && (
        <div>
          <h4 style={{ fontSize: '1rem', fontWeight: 600, marginBottom: '12px' }}>Audio Segment Spectro-Temporal Saliency</h4>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(120px, 1fr))', gap: '10px' }}>
            {details.segment_saliency.map((s, i) => (
              <div key={i} style={{ background: 'rgba(15, 23, 42, 0.6)', border: '1px solid var(--border-card)', borderRadius: '8px', padding: '10px', textAlign: 'center' }}>
                <span style={{ fontSize: '0.72rem', color: 'var(--text-muted)' }}>{s.start_time_sec}s - {s.end_time_sec}s</span>
                <p style={{ fontWeight: 700, fontSize: '0.9rem', color: s.score >= 0.5 ? 'var(--fake-rose)' : 'var(--real-emerald)', marginTop: '4px' }}>
                  {(s.score * 100).toFixed(0)}%
                </p>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
