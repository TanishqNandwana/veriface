export default function handler(req, res) {
  res.setHeader('Access-Control-Allow-Origin', '*');
  res.setHeader('Access-Control-Allow-Methods', 'GET, OPTIONS');
  res.setHeader('Access-Control-Allow-Headers', 'Content-Type');

  if (req.method === 'OPTIONS') {
    return res.status(200).end();
  }

  return res.status(200).json({
    status: 'healthy',
    version: '1.0.0',
    device_info: {
      active_device: 'Edge Serverless ML Runtime',
      platform: 'Vercel Multi-Region Edge'
    },
    active_models: {
      image: 'FA-ViT (Vision Transformer + GAM/LAM + FFT Forensic Cue)',
      video: 'Style Latent Flows + StyleGRU + 3D ResNet + SAM + TTE',
      audio: 'CLAD (AASIST Graph Attention + MoCo + Length Loss)'
    }
  });
}
