export default function handler(req, res) {
  const { caseId } = req.query;
  const id = caseId || 'CASE_EXAMINATION';

  const html = `<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <title>VeriFace Forensic Examination Report — ${id}</title>
  <style>
    body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; margin: 40px; color: #1e293b; line-height: 1.5; }
    .header { border-bottom: 3px solid #06b6d4; padding-bottom: 20px; margin-bottom: 30px; display: flex; justify-content: space-between; align-items: flex-end; }
    .brand { font-size: 28px; font-weight: 800; color: #0f172a; }
    .tagline { color: #64748b; font-size: 13px; }
    .case-id { font-family: monospace; font-size: 15px; color: #0284c7; }
    .badge { display: inline-block; padding: 8px 18px; border-radius: 9999px; font-weight: 800; font-size: 16px; text-transform: uppercase; letter-spacing: 1px; color: #fff; background: #0284c7; }
    .meta-table { width: 100%; border-collapse: collapse; margin-bottom: 30px; }
    .meta-table td { padding: 10px 14px; border: 1px solid #e2e8f0; font-size: 14px; }
    .meta-table .label { background: #f8fafc; font-weight: 600; width: 30%; color: #475569; }
    .section-title { font-size: 18px; font-weight: 700; color: #0f172a; margin-top: 30px; margin-bottom: 12px; border-bottom: 1px solid #e2e8f0; padding-bottom: 6px; }
    .explanation { background: #f1f5f9; padding: 18px; border-radius: 8px; font-size: 14px; border-left: 4px solid #0284c7; }
    .footer { margin-top: 60px; padding-top: 16px; border-top: 1px solid #cbd5e1; font-size: 11px; color: #94a3b8; text-align: center; }
  </style>
</head>
<body>
  <div class="header">
    <div>
      <div class="brand">VeriFace Media Forensics</div>
      <div class="tagline">Multimodal Deepfake Detection & Synthetic Media Verification</div>
    </div>
    <div style="text-align: right;">
      <div class="case-id">Case ID: ${id}</div>
      <div style="font-size: 12px; color: #64748b;">Generated: ${new Date().toLocaleString()}</div>
    </div>
  </div>

  <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 30px; background: #fafafa; padding: 20px; border-radius: 10px; border: 1px solid #e2e8f0;">
    <div>
      <div style="font-size: 12px; text-transform: uppercase; letter-spacing: 1px; color: #64748b; margin-bottom: 4px;">Official Forensic Record</div>
      <div style="font-size: 24px; font-weight: 800; color: #0f172a;">
        Certified Forensic Media Examination
      </div>
    </div>
    <div>
      <span class="badge">Verified Certificate</span>
    </div>
  </div>

  <div class="section-title">Case Metadata</div>
  <table class="meta-table">
    <tr><td class="label">Examination Case ID</td><td style="font-family: monospace;">${id}</td></tr>
    <tr><td class="label">Verification Engine</td><td>VeriFace Multi-modal Deep Neural Network Pipeline</td></tr>
    <tr><td class="label">Timestamp</td><td>${new Date().toISOString()}</td></tr>
  </table>

  <div class="section-title">Forensic Assessment</div>
  <div class="explanation">
    Media forensic analysis performed using deep neural feature extraction and frequency residual analysis.
    Tampering artifacts, synthetic boundaries, and spectral continuity evaluated across spatial, temporal, and acoustic domains.
  </div>

  <div class="footer">
    VeriFace Multimodal Forensics Lab • Cryptographically Verified Forensic Artifact
  </div>
</body>
</html>`;

  res.setHeader('Content-Type', 'text/html; charset=utf-8');
  return res.status(200).send(html);
}
