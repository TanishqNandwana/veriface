# VeriFace — Deployment Guide & Infrastructure Documentation

**Platform**: VeriFace — Verify the Media. Reveal the Truth.  
**Frontend Deployment URL**: [https://veriface-tau.vercel.app](https://veriface-tau.vercel.app)  
**Alternative Production URL**: [https://veriface-420705aag-tanishq-462a.vercel.app](https://veriface-420705aag-tanishq-462a.vercel.app)  
**GitHub Repository**: [https://github.com/tanishqnandwana/veriface](https://github.com/tanishqnandwana/veriface)  
**Target Branch**: `main`  
**Vercel Project**: `tanishq-462a/veriface` (ID: `prj_mrEJ1lOqFcW5pjKPleJ8GIDFy2qK`)  

---

## 1. Deployment Architecture Overview

VeriFace adopts a decoupled, high-performance architecture:

```
┌────────────────────────────────────────────────────────┐
│                   CLIENT (BROWSER)                     │
└───────────────┬────────────────────────┬───────────────┘
                │                        │
       Static Assets & UI          API / Multipart
       (HTML, CSS, JS)             Inference Calls
                │                        │
                ▼                        ▼
┌───────────────────────────────┐ ┌───────────────────────────────┐
│     VERCEL EDGE PLATFORM      │ │    PYTHON ML BACKEND SERVICE  │
│  (React 18 + Vite SPA CDN)    │ │   (FastAPI + Uvicorn + PyTorch│
│                               │ │    + FFmpeg + OpenCV + Librosa│
│  • Framework: Vite            │ │                               │
│  • Root Dir: frontend         │ │  • Image: EfficientNet-B0     │
│  • Output: dist               │ │  • Video: Frame-CNN + BiGRU   │
│  • Dynamic API URL Support    │ │  • Audio: CLAD AASIST Graph   │
│  • Auto-deploy on git push    │ │  • Fusion: Late Multi-modal   │
└───────────────────────────────┘ └───────────────────────────────┘
```

### Why Decoupled Deployment?
- **Frontend on Vercel**: Vercel offers global edge caching, fast static asset delivery, instant atomic rollbacks, and automatic CI/CD integration with GitHub.
- **Backend on Dedicated ML Infrastructure**: The machine learning inference engine requires heavy binary runtime dependencies (PyTorch 2.x, OpenCV, Librosa, FFmpeg) totaling ~1.5 GB on disk, along with dedicated RAM (~1 GB peak) for multi-frame video extraction and audio spectrogram Fourier transforms. These exceed Vercel Serverless Function limits (50 MB zipped / 250 MB uncompressed, 10-60s timeout).

---

## 2. Vercel Frontend Configuration

The frontend is configured and live on Vercel:

| Parameter | Configuration Value |
|---|---|
| **Project Name** | `veriface` |
| **Framework Preset** | `Vite` |
| **Root Directory** | `frontend` |
| **Build Command** | `npm run build` (or `vite build`) |
| **Output Directory** | `dist` |
| **Install Command** | `npm install` |
| **Node Version** | `20.x` / `24.x` |
| **Live Production Domain** | `https://veriface-tau.vercel.app` |

### Frontend Environment Variables
Configure these in the Vercel Project Dashboard (`Settings` -> `Environment Variables`) or via Vercel CLI:

| Variable | Required | Description | Example |
|---|---|---|---|
| `VITE_API_BASE_URL` | Optional | Base URL of the deployed FastAPI backend. When set, all API calls (`/api/health`, `/api/analyze/*`, `/api/cases`, `/api/reports/*`) target this backend. | `https://api.veriface.yourdomain.com` |

---

## 3. Backend Deployment Instructions

The backend repository includes complete production deployment assets:
- `Dockerfile`: Production container based on `python:3.12-slim` with system packages (`ffmpeg`, `libgl1`, `libsndfile1`) and PyTorch dependencies.
- `Procfile`: Ready for platforms supporting Procfiles (`web: uvicorn backend.main:app --host 0.0.0.0 --port ${PORT:-8000}`).
- `.env.example`: Configuration template for environment variables.

### Option A: Deploying via Docker (Recommended for Any Cloud VPS / PaaS)
Run the containerized backend with Docker:
```bash
docker build -t veriface-backend .
docker run -p 8000:8000 -e PORT=8000 -e CORS_ORIGINS="https://veriface-tau.vercel.app" veriface-backend
```

### Option B: Deploying to Render
1. In [Render Dashboard](https://dashboard.render.com), click **New +** -> **Web Service**.
2. Connect repository `https://github.com/tanishqnandwana/veriface`.
3. Set:
   - **Environment**: Docker (or Python 3)
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `uvicorn backend.main:app --host 0.0.0.0 --port $PORT`
4. Set Environment Variables:
   - `CORS_ORIGINS`: `https://veriface-tau.vercel.app`
   - `DATABASE_URL`: Optional PostgreSQL URL (defaults to internal SQLite).
5. Deploy and copy the Render URL (e.g. `https://veriface-api.onrender.com`).
6. In Vercel, set `VITE_API_BASE_URL=https://veriface-api.onrender.com`.

### Option C: Deploying to Hugging Face Spaces (Free CPU & 16GB RAM)
1. Create a new Space on [Hugging Face](https://huggingface.co/new-space).
2. Choose **Docker** SDK (Blank).
3. Push or connect the VeriFace repository.
4. Hugging Face automatically builds the `Dockerfile` and assigns a public HTTPS URL.
5. In Vercel, set `VITE_API_BASE_URL` to your Hugging Face Space direct URL.

---

## 4. Connecting Frontend to Backend

Once the backend is deployed:

1. Open your Vercel project settings:
   ```bash
   vercel env add VITE_API_BASE_URL production
   ```
   (Or navigate to Vercel Dashboard -> `veriface` -> `Settings` -> `Environment Variables`).
2. Enter the backend URL:
   ```text
   https://<your-deployed-backend-url>
   ```
3. Trigger a production redeploy:
   ```bash
   vercel redeploy veriface-tau.vercel.app
   ```
4. Verify by opening [https://veriface-tau.vercel.app](https://veriface-tau.vercel.app). The top right status indicator in the Navbar will turn **green ("API Connected: Online")**!

---

## 5. Model Checkpoint Policy & Storage

| Modality | Architecture | Active Checkpoint | Size in Git |
|---|---|---|---|
| **Image** | EfficientNet-B0 Face Classifier | `checkpoints/image/efficientnet_b0_best.pth` | 16.0 MB |
| **Video** | Frame-CNN + BiGRU Temporal Aggregator | `checkpoints/video/frame_cnn___bigru_temporal_aggregator_best.pth` | 906 KB |
| **Audio** | CLAD AASIST Graph Attention | `checkpoints/audio/clad_full_model_aasist___contrastive___lengthloss_best.pth` | 702 KB |

*All three production checkpoints total ~17.6 MB and are committed directly to the GitHub repository.*  
Oversized research checkpoints (>100 MB like `fa_vit_gam_lam_fft_fal_best.pth` 333MB, `style_latent_flow_video_model_full_best.pth` 325MB) are excluded via `.gitignore` and can be retrieved from external academic storage for retraining experiments.

---

## 6. Automatic Redeployment Workflow (CI/CD)

The Vercel project is linked to the GitHub repository `tanishqnandwana/veriface`.

Future updates follow this automated workflow:
1. Make code modifications locally:
   ```bash
   # Test frontend build
   npm --prefix frontend run build

   # Test backend test suite
   PYTHONPATH=. venv/bin/pytest tests/
   ```
2. Commit and push to GitHub:
   ```bash
   git add .
   git commit -m "Your descriptive commit message"
   git push origin main
   ```
3. Vercel automatically detects the push, initiates a new build, and deploys the latest version to [https://veriface-tau.vercel.app](https://veriface-tau.vercel.app) with zero downtime.

---

## 7. Troubleshooting

| Issue | Cause | Resolution |
|---|---|---|
| Navbar shows "API Offline" | Frontend cannot reach backend | Verify `VITE_API_BASE_URL` is configured in Vercel settings and backend service is running. |
| CORS error in browser console | Backend does not permit frontend origin | Ensure backend environment variable `CORS_ORIGINS` includes `https://veriface-tau.vercel.app` (or `*`). |
| 413 Request Entity Too Large | Uploaded file > 100MB | VeriFace limits single media uploads to 100MB. Compress video or audio file. |
| PDF Report 404 | Report not generated | PDF generation requires `reportlab`. Check backend logs for generation errors. |
