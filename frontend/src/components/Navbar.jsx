import React from 'react';
import { ShieldCheck, Cpu, Database } from 'lucide-react';

export default function Navbar({ healthInfo }) {
  return (
    <header className="glass-panel" style={{ margin: '20px auto', maxWidth: '1200px', padding: '16px 28px', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
      <div style={{ display: 'flex', alignItems: 'center', gap: '14px' }}>
        <div style={{ background: 'linear-gradient(135deg, #06b6d4, #3b82f6)', padding: '10px', borderRadius: '12px', display: 'flex' }}>
          <ShieldCheck size={28} color="#ffffff" />
        </div>
        <div>
          <h1 style={{ fontSize: '1.4rem', fontWeight: 800, letterSpacing: '-0.5px' }}>VeriFace</h1>
          <p style={{ fontSize: '0.8rem', color: 'var(--text-secondary)' }}>Verify the Media. Reveal the Truth.</p>
        </div>
      </div>

      <div style={{ display: 'flex', alignItems: 'center', gap: '16px' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '6px', fontSize: '0.85rem', color: 'var(--accent-cyan)', background: 'rgba(6, 182, 212, 0.1)', padding: '6px 12px', borderRadius: '20px', border: '1px solid rgba(6, 182, 212, 0.3)' }}>
          <Cpu size={16} />
          <span>{healthInfo?.device_info?.active_device ? `GPU Acceleration (${healthInfo.device_info.active_device.toUpperCase()})` : 'Backend Active'}</span>
        </div>
        <div style={{ display: 'flex', alignItems: 'center', gap: '6px', fontSize: '0.85rem', color: 'var(--text-secondary)' }}>
          <Database size={16} />
          <span>SQLite / PostgreSQL</span>
        </div>
      </div>
    </header>
  );
}
