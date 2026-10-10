import React, { useState, useRef } from 'react';
import { UploadCloud, Loader2, Play } from 'lucide-react';

export default function Uploader({ modality, onAnalyze, isAnalyzing }) {
  const [selectedFile, setSelectedFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState(null);
  const fileInputRef = useRef(null);

  const handleFileChange = (file) => {
    if (!file) return;
    setSelectedFile(file);
    const url = URL.createObjectURL(file);
    setPreviewUrl(url);
  };

  const handleDrop = (e) => {
    e.preventDefault();
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      handleFileChange(e.dataTransfer.files[0]);
    }
  };

  const handleDragOver = (e) => {
    e.preventDefault();
  };

  const handleStartAnalysis = () => {
    if (selectedFile) {
      onAnalyze(selectedFile, modality);
    }
  };

  return (
    <div className="glass-panel" style={{ padding: '32px', marginBottom: '32px' }}>
      <input
        type="file"
        ref={fileInputRef}
        style={{ display: 'none' }}
        onChange={(e) => handleFileChange(e.target.files[0])}
        accept={modality === 'image' ? 'image/*' : modality === 'video' ? 'video/*' : 'audio/*'}
      />

      {!selectedFile ? (
        <div
          className="dropzone"
          onDrop={handleDrop}
          onDragOver={handleDragOver}
          onClick={() => fileInputRef.current.click()}
        >
          <UploadCloud size={48} color="var(--accent-cyan)" style={{ marginBottom: '16px' }} />
          <h3 style={{ fontSize: '1.2rem', fontWeight: 600, marginBottom: '8px' }}>
            Upload {modality.toUpperCase()} for Forensic Deepfake Analysis
          </h3>
          <p style={{ color: 'var(--text-secondary)', fontSize: '0.9rem', marginBottom: '16px' }}>
            Drag and drop your file here, or click to browse (Max 100MB)
          </p>
          <button className="btn-outline">Select File</button>
        </div>
      ) : (
        <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: '20px' }}>
          <div style={{ maxWidth: '400px', width: '100%', borderRadius: '12px', overflow: 'hidden', border: '1px solid var(--border-card)', background: '#000', display: 'flex', justifyContent: 'center' }}>
            {modality === 'image' && (
              <img src={previewUrl} alt="Preview" style={{ maxHeight: '260px', objectFit: 'contain' }} />
            )}
            {modality === 'video' && (
              <video src={previewUrl} controls style={{ maxHeight: '260px', width: '100%' }} />
            )}
            {modality === 'audio' && (
              <div style={{ padding: '24px', width: '100%', textCenter: 'center' }}>
                <audio src={previewUrl} controls style={{ width: '100%' }} />
              </div>
            )}
          </div>

          <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
            <span style={{ fontSize: '0.9rem', color: 'var(--text-secondary)' }}>
              File: <b>{selectedFile.name}</b> ({(selectedFile.size / (1024 * 1024)).toFixed(2)} MB)
            </span>
            <button className="btn-outline" style={{ padding: '6px 12px', fontSize: '0.8rem' }} onClick={() => { setSelectedFile(null); setPreviewUrl(null); }}>
              Change
            </button>
          </div>

          <button
            className="btn-primary"
            onClick={handleStartAnalysis}
            disabled={isAnalyzing}
            style={{ fontSize: '1.05rem', padding: '14px 32px' }}
          >
            {isAnalyzing ? (
              <>
                <Loader2 className="spin" size={20} />
                Running Deepfake Neural Inference...
              </>
            ) : (
              <>
                <Play size={20} />
                Analyze Media & Generate Forensic Report
              </>
            )}
          </button>
        </div>
      )}
    </div>
  );
}
