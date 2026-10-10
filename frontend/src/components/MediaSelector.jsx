import React from 'react';
import { Image, Video, Mic } from 'lucide-react';

export default function MediaSelector({ activeModality, onSelectModality }) {
  const modalities = [
    {
      id: 'image',
      title: 'Image Analysis',
      subtitle: 'FA-ViT Vision Transformer + GAM/LAM + FFT Forensic Cue',
      icon: Image,
      color: '#06b6d4'
    },
    {
      id: 'video',
      title: 'Video Analysis',
      subtitle: 'Style Latent Flows + StyleGRU + 3D ResNet + SAM + TTE',
      icon: Video,
      color: '#3b82f6'
    },
    {
      id: 'audio',
      title: 'Audio Analysis',
      subtitle: 'CLAD + AASIST Graph Attention + MoCo + Length Loss',
      icon: Mic,
      color: '#8b5cf6'
    }
  ];

  return (
    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(300px, 1fr))', gap: '20px', marginBottom: '28px' }}>
      {modalities.map((m) => {
        const Icon = m.icon;
        const isActive = activeModality === m.id;
        return (
          <div
            key={m.id}
            className={`modality-card ${isActive ? 'active' : ''}`}
            onClick={() => onSelectModality(m.id)}
          >
            <div style={{ background: `rgba(${m.color === '#06b6d4' ? '6, 182, 212' : m.color === '#3b82f6' ? '59, 130, 246' : '139, 92, 246'}, 0.15)`, width: '56px', height: '56px', borderRadius: '14px', display: 'flex', alignItems: 'center', justifyContent: 'center', margin: '0 auto 16px auto' }}>
              <Icon size={28} color={m.color} />
            </div>
            <h3 style={{ fontSize: '1.15rem', fontWeight: 700, marginBottom: '6px' }}>{m.title}</h3>
            <p style={{ fontSize: '0.82rem', color: 'var(--text-secondary)', lineHeight: 1.4 }}>{m.subtitle}</p>
          </div>
        );
      })}
    </div>
  );
}
