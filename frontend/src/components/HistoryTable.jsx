import React from 'react';
import { History, FileText, Download } from 'lucide-react';
import { getApiUrl } from '../config';

export default function HistoryTable({ cases, onSelectCase }) {
  if (!cases || cases.length === 0) {
    return (
      <div className="glass-panel" style={{ padding: '32px', textAlign: 'center', color: 'var(--text-secondary)' }}>
        <History size={40} style={{ marginBottom: '12px', opacity: 0.5 }} />
        <p>No prior forensic cases recorded in database.</p>
      </div>
    );
  }

  return (
    <div className="glass-panel" style={{ padding: '32px' }}>
      <h3 style={{ fontSize: '1.2rem', fontWeight: 700, marginBottom: '20px', display: 'flex', alignItems: 'center', gap: '10px' }}>
        <History size={22} color="var(--accent-cyan)" />
        Case History & Persisted Forensic Records
      </h3>

      <div style={{ overflowX: 'auto' }}>
        <table style={{ width: '100%', borderCollapse: 'collapse', textAlign: 'left', fontSize: '0.9rem' }}>
          <thead>
            <tr style={{ borderBottom: '1px solid var(--border-card)', color: 'var(--text-muted)' }}>
              <th style={{ padding: '12px' }}>Case ID</th>
              <th style={{ padding: '12px' }}>Filename</th>
              <th style={{ padding: '12px' }}>Media Type</th>
              <th style={{ padding: '12px' }}>Verdict</th>
              <th style={{ padding: '12px' }}>Confidence</th>
              <th style={{ padding: '12px' }}>Timestamp</th>
              <th style={{ padding: '12px' }}>Report</th>
            </tr>
          </thead>
          <tbody>
            {cases.map((c) => {
              const isFake = c.verdict === 'manipulated';
              return (
                <tr key={c.case_id} style={{ borderBottom: '1px solid rgba(255,255,255,0.05)' }}>
                  <td style={{ padding: '12px', fontFamily: 'var(--font-mono)', fontSize: '0.82rem', color: 'var(--accent-cyan)' }}>
                    <span style={{ cursor: 'pointer', textDecoration: 'underline' }} onClick={() => onSelectCase(c)}>
                      {c.case_id}
                    </span>
                  </td>
                  <td style={{ padding: '12px', fontWeight: 500 }}>{c.filename}</td>
                  <td style={{ padding: '12px', textTransform: 'capitalize', color: 'var(--text-secondary)' }}>{c.media_type}</td>
                  <td style={{ padding: '12px' }}>
                    <span className={`badge-verdict ${isFake ? 'manipulated' : 'real'}`} style={{ fontSize: '0.75rem', padding: '4px 10px' }}>
                      {c.verdict}
                    </span>
                  </td>
                  <td style={{ padding: '12px', fontWeight: 700 }}>{(c.confidence * 100).toFixed(1)}%</td>
                  <td style={{ padding: '12px', color: 'var(--text-muted)', fontSize: '0.8rem' }}>
                    {c.created_at ? new Date(c.created_at).toLocaleString() : 'Recent'}
                  </td>
                  <td style={{ padding: '12px' }}>
                    <a
                      href={c.printable_report_url || getApiUrl(`/api/reports/${c.case_id}`)}
                      target="_blank"
                      rel="noopener noreferrer"
                      className="btn-outline"
                      style={{ padding: '4px 10px', fontSize: '0.78rem' }}
                    >
                      <Download size={14} /> PDF
                    </a>
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </div>
  );
}
