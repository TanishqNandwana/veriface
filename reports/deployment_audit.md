# VeriFace — Deployment Audit Report

**Date**: October 10, 2026  
**Auditor**: Autonomous Senior DevOps & Full-Stack Deployment Engineer  
**Project**: VeriFace — Multimodal AI Deepfake Detection & Media Forensics  
**Target Repository**: `https://github.com/tanishqnandwana/veriface` (Branch: `main`)  
**Deployment Platform**: Vercel (Frontend) & Compatible Python Backend Host  

---

## 1. Executive Summary

A comprehensive architectural and deployment audit was conducted on the **VeriFace** codebase located at `/Users/tanishqnandwana/Desktop/Veriface`. 

The project is structured as a decoupled full-stack machine learning application:
- **Frontend**: Single Page Application built with **React 18** and bundled with **Vite 5**.
- **Backend**: High-performance REST API built with **FastAPI** (Python 3.12).
- **ML Inference**: PyTorch-based multimodal detection pipelines covering **Image** (EfficientNet-B0 / FA-ViT), **Video** (Frame-CNN + BiGRU / Style Latent Flows), and **Audio** (CLAD AASIST + Contrastive Learning).

---

## 2. Technical Stack Audit

| Component | Technology | Version | Details |
|---|---|---|---|
| **Frontend Framework** | React | 18.2.0 | Pure SPA, JSX, Lucide React icons, Glassmorphism CSS |
| **Frontend Tooling** | Vite | 5.1.6 | Fast ESM bundler; builds cleanly to `frontend/dist` in ~600ms |
| **Node.js Runtime** | Node.js | v20.20.2 | Managed via Homebrew Node 20 |
| **Package Manager** | npm | 10.8.2 | Lockfile: `frontend/package-lock.json` |
| **Backend Framework** | FastAPI | 0.141.1 | Async ASGI web server |
| **ASGI Server** | Uvicorn | 0.52.4 | Standard distribution with uvloop/httptools |
| **Python Runtime** | CPython | 3.12.3 | Active virtual environment at `./venv` |
| **Database** | SQLAlchemy | 2.0.52 | SQLite (`veriface.db`) default, PostgreSQL via `DATABASE_URL` |
| **Report Generation** | ReportLab | 5.0.1 | Programmatic PDF forensic report generator |
| **Hardware Acceleration** | PyTorch MPS / CUDA / CPU | 2.14.0 | Automatic device detection (Apple Silicon `mps` active locally) |

---

## 3. API Contract & Endpoints

The backend exposes the following REST API under `/api`:
- `GET /api/health`: Health status, device acceleration info (`mps`/`cuda`/`cpu`), and active model catalogue.
- `POST /api/analyze/image`: Multipart image upload; validates bounds (<100MB), runs neural classifier, extracts Grad-CAM heatmap, records case to database, produces PDF forensic report.
- `POST /api/analyze/video`: Multipart video upload; extracts frame sequence, extracts audio track via FFmpeg for late fusion (60% visual + 40% audio), generates frame timeline and PDF report.
- `POST /api/analyze/audio`: Multipart audio upload; extracts log-mel spectrogram, executes CLAD inference, computes temporal segment saliency, generates PDF report.
- `GET /api/cases`: Retrieves historical forensic cases (paginated/limited).
- `GET /api/cases/{case_id}`: Retrieves single case forensic metadata and analysis results.
- `GET /api/reports/{case_id}`: Downloads generated PDF forensic report.
- Static mounts: `/storage` (media & Grad-CAM overlays) and `/forensic_reports` (PDF reports).

---

## 4. Model Checkpoint Audit & GitHub Policy

| Checkpoint Path | Architecture | Size | Status & Version Control Decision |
|---|---|---|---|
| `checkpoints/image/efficientnet_b0_best.pth` | EfficientNet-B0 Face Classifier | **16 MB** | **Include in Git** (active production image model) |
| `checkpoints/video/frame_cnn___bigru_temporal_aggregator_best.pth` | Frame-CNN + BiGRU Video Model | **906 KB** | **Include in Git** (active production video model) |
| `checkpoints/audio/clad_full_model_aasist___contrastive___lengthloss_best.pth` | CLAD AASIST Audio Detector | **702 KB** | **Include in Git** (active production audio model) |
| `checkpoints/audio/aasist_baseline_classifier_best.pth` | AASIST Baseline | 538 KB | Include in Git (baseline model) |
| `checkpoints/audio/spectrogram_cnn_baseline_best.pth` | Spectrogram CNN Baseline | 114 KB | Include in Git (baseline model) |
| `checkpoints/video/3d_resnet_video_classifier_best.pth` | 3D ResNet Video Classifier | 4.4 MB | Include in Git (baseline model) |
| `checkpoints/image/resnet_50_best.pth` | ResNet-50 Baseline | 91 MB | **Exclude from Git** (>50MB, trigger warning) |
| `checkpoints/image/fa_vit_gam_lam_fft_fal_best.pth` | FA-ViT Full Transformer | 333 MB | **Exclude from Git** (>100MB GitHub hard rejection) |
| `checkpoints/image/vanilla_vit_b_16_best.pth` | Vanilla ViT-B/16 | 328 MB | **Exclude from Git** (>100MB GitHub hard rejection) |
| `checkpoints/video/style_latent_flow_video_model_full_best.pth` | Style Latent Flow Full Model | 325 MB | **Exclude from Git** (>100MB GitHub hard rejection) |

**Conclusion on Checkpoints:**  
All three active production inference models (`efficientnet_b0_best.pth` at 16MB, `frame_cnn___bigru...` at 906KB, and `clad_full_model...` at 702KB) total **~17.6 MB**, perfectly within GitHub version control limits without requiring LFS or external download dependencies. The oversized research checkpoints (>100MB) will be excluded via `.gitignore` while strictly preserved in the local environment.

---

## 5. Potential Blockers & Architectural Constraints

1. **Vercel Serverless Function Limits for ML Workloads**:
   - Vercel Serverless Functions have a 50MB (uncompressed) or 250MB limit on function bundle size.
   - PyTorch + Torchvision + Torchaudio + Librosa + OpenCV collectively exceed 1.2 GB, far beyond Vercel Serverless limits.
   - Serverless execution timeouts (10s free / 60s pro) are too tight for full video frame extraction and multi-second audio spectrogram calculations.
   - **Solution**: Deploy the React SPA frontend to **Vercel** with full CDN distribution, high-speed edge delivery, and configure API proxying/CORS pointing to the dedicated backend service.

2. **Frontend Hardcoded Paths**:
   - Currently, `frontend/src/App.jsx` and components use relative `/api/...` endpoints that rely on Vite's local dev server proxy.
   - For production Vercel deployment, we must:
     a) Support `VITE_API_BASE_URL` dynamically for direct cross-origin API calls with credentials.
     b) Configure `vercel.json` rewrites/routes so that relative `/api/*` and `/storage/*` requests can seamlessly proxy to the backend if desired.

3. **Git Status**:
   - The directory is not currently a Git repository (`fatal: not a git repository`).
   - Large directories like `datasets/` (384MB), `experiments/` (659MB), `storage/` (782MB), `forensic_reports/` (71MB), and `venv/` must be safely gitignored before creating the initial commit.

---

## 6. Proposed Deployment Plan

1. **Phase 2 & 3: Codebase Preparation**:
   - Create comprehensive `.gitignore` ensuring zero secret leaks, no venvs, no node_modules, and ignoring oversized checkpoint files while preserving the active 17.6MB models.
   - Add `.env.example` with documented environment configuration.
   - Update frontend API clients to use `VITE_API_BASE_URL` with automatic fallback to relative paths.
   - Add production `vercel.json` for frontend SPA routing and API proxy routing.
   - Validate frontend build (`npm run build`) and backend tests (`pytest`).

2. **Phase 4 & 5: GitHub Repository Creation & Push**:
   - Initialize git, set default branch to `main`.
   - Use authenticated GitHub access (`gh` / OAuth) to create `tanishqnandwana/veriface`.
   - Commit clean source code and push to `origin/main`.

3. **Phase 6 & 7: Vercel Frontend Deployment**:
   - Link project to Vercel under authenticated account `tanishqnandwana`.
   - Set framework preset to `vite`, root directory to `frontend`.
   - Deploy to Vercel production.
   - Verify live frontend URL.

4. **Phase 8 & 9: Backend Architecture & Connection**:
   - Provide complete production backend configuration (Docker container, Procfile, environment config).
   - Configure backend CORS and URL routing.
   - Connect frontend to backend.

5. **Phase 10 & 11: End-to-End Validation & Automation**:
   - Run production verification on live endpoints.
   - Ensure GitHub-to-Vercel CI/CD automatic deployment on push.
   - Finalize `README.md` and `DEPLOYMENT.md`.
