# VeriFace — Verify the Media. Reveal the Truth.

[![Vercel Deployment](https://img.shields.io/badge/Vercel-Live%20Frontend-black?logo=vercel)](https://veriface-tau.vercel.app)
[![GitHub Repository](https://img.shields.io/badge/GitHub-tanishqnandwana%2Fveriface-blue?logo=github)](https://github.com/tanishqnandwana/veriface)
[![FastAPI](https://img.shields.io/badge/Backend-FastAPI%200.141-009688?logo=fastapi)](https://fastapi.tiangolo.com)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.14%20(MPS%2FCUDA%2FCPU)-EE4C2C?logo=pytorch)](https://pytorch.org)
[![React](https://img.shields.io/badge/Frontend-React%2018%20%2B%20Vite-61DAFB?logo=react)](https://react.dev)

VeriFace is a multimodal AI deepfake detection and media forensics platform capable of analyzing manipulated **Images**, deepfake **Videos**, and synthetic/cloned **Audio** attacks. It generates fine-grained forensic explanations (Grad-CAM spatial heatmaps, temporal frame-by-frame manipulation timelines, and audio segment saliencies) along with downloadable PDF forensic reports.

---

## Live Links & Repositories

- **Live Production Application (Vercel)**: [https://veriface-tau.vercel.app](https://veriface-tau.vercel.app)
- **Official GitHub Repository**: [https://github.com/tanishqnandwana/veriface](https://github.com/tanishqnandwana/veriface)
- **Deployment Documentation**: [DEPLOYMENT.md](DEPLOYMENT.md)
- **Deployment Audit Report**: [reports/deployment_audit.md](reports/deployment_audit.md)

---

## Key Features

1. **Multimodal Detection Architecture**:
   - **Image Branch**: Forgery-Aware Adaptive Vision Transformer (**FA-ViT**) & **EfficientNet-B0** with Global Adaptive Module (GAM), Local Adaptive Module (LAM), Fine-grained Adaptive Learning (FAL) loss, and 2D FFT spectral forensic analysis.
   - **Video Branch**: Frame-CNN + **BiGRU Temporal Aggregator** and **Style Latent Flows** with pSp $W+$ latent space inversion, temporal style differences ($\Delta S_t$), 3D ResNet-50 visual representations, Style Attention Module (SAM), and Temporal Transformer Encoder (TTE).
   - **Audio Branch**: Contrastive Learning Audio Deepfake Detector (**CLAD**) combining **AASIST** spectro-temporal graph attention, MoCo-style contrastive representations, length-regularized loss, and downstream manipulation classification.
   - **Media-Aware Late Fusion**: Extracts audio tracks from submitted video clips using FFmpeg and performs late score fusion (60% Visual, 40% Audio).

2. **Full-Stack Media Forensics Platform**:
   - **Interactive Frontend**: Glassmorphism React SPA featuring drag-and-drop media uploaders, modality-specific visualizers (Grad-CAM heatmaps, temporal bar graphs, audio saliency), and real-time case history.
   - **FastAPI REST Server**: Async file ingestion (<100MB bound checking), SHA-256 chain-of-custody hashing, SQLite/PostgreSQL persistence, and automated PDF forensic report generation via ReportLab.
   - **Hardware Acceleration**: Automatic PyTorch device detection supporting Apple Silicon GPU (`mps`), NVIDIA CUDA, and CPU fallback.

---

## Technology Stack

| Layer | Technologies |
|---|---|
| **Frontend** | React 18, Vite 5, Lucide React, Vanilla CSS (Glassmorphism design system) |
| **Backend API** | FastAPI, Uvicorn (ASGI), Pydantic v2, Starlette |
| **Machine Learning** | PyTorch 2.x, Torchvision, Torchaudio, Timm, Facenet-PyTorch, Librosa, OpenCV |
| **Database** | SQLAlchemy ORM with SQLite (default) and PostgreSQL compatibility |
| **Report Generation** | ReportLab PDF engine |
| **Hosting & CI/CD** | Vercel (Edge CDN Frontend) & Containerized Python Host (Backend) |

---

## System Architecture

```text
┌────────────────────────────────────────────────────────────────────────┐
│                   VERIFACE CLIENT (REACT 18 SPA)                       │
│    • Drag & Drop Uploader      • Modality Selector (Img/Vid/Aud)       │
│    • Grad-CAM Heatmap Viewer   • Temporal Frame Scoring Visualizer     │
│    • Audio Saliency Display    • Filterable Case History & PDF Report  │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │ HTTP / REST (/api/*)
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│                   FASTAPI ASYNC BACKEND SERVER                         │
│    • Media Validation (<100MB) • SHA-256 Hash Hashing & Persistence    │
│    • Audio Extractor (FFmpeg)  • ReportLab PDF Forensic Generator      │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│               MASTER MULTIMODAL INFERENCE ENGINE                       │
│   ├── Image: EfficientNet-B0 / FA-ViT + FFT High-Frequency Spectrum    │
│   ├── Video: Frame-CNN + BiGRU / Style Latent Flows (pSp + SAM + TTE)  │
│   ├── Audio: CLAD (AASIST Spectro-Temporal Graph Attention)            │
│   └── Late Fusion: 60% Visual + 40% Audio Late Score Integration       │
└────────────────────────────────────────────────────────────────────────┘
```

---

## Local Setup & Development

### 1. Prerequisites
- **Python**: 3.10+ (tested on Python 3.12)
- **Node.js**: 18+ and npm
- **FFmpeg**: (Optional, required for audio extraction from video files)

### 2. Environment Setup
```bash
# Clone the repository
git clone https://github.com/tanishqnandwana/veriface.git
cd veriface

# Create and activate Python virtual environment
python3 -m venv venv
source venv/bin/activate

# Install Python backend and ML dependencies
pip install -r requirements.txt

# Install frontend dependencies
cd frontend
npm install
cd ..

# Verify environment and hardware acceleration
python scripts/check_environment.py
```

### 3. Environment Configuration
Copy `.env.example` to create local environment files:
```bash
cp .env.example .env
cp frontend/.env.example frontend/.env
```
Default configuration values:
- `PORT=8000`
- `DATABASE_URL=sqlite:///./veriface.db`
- `CORS_ORIGINS=*`
- `VITE_API_BASE_URL=` (leave blank for local development to use Vite's proxy)

### 4. Running the Application

#### Option A: Running Both via FastAPI (Single Server)
```bash
# Build frontend static bundle
npm --prefix frontend run build

# Launch FastAPI (serves API and React frontend on http://localhost:8000)
PYTHONPATH=. venv/bin/python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000
```

#### Option B: Decoupled Development Mode (Hot Reloading)
```bash
# Terminal 1: Start FastAPI backend
PYTHONPATH=. venv/bin/python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload

# Terminal 2: Start Vite frontend dev server (port 3000)
cd frontend
npm run dev
```
Open [http://localhost:3000](http://localhost:3000) in your web browser.

---

## Model Checkpoint Configuration

VeriFace includes lightweight, production-grade model checkpoints committed directly to the GitHub repository:

| Modality | Model Architecture | Checkpoint File | Size | Version Controlled? |
|---|---|---|---|---|
| **Image** | EfficientNet-B0 Deepfake Classifier | `checkpoints/image/efficientnet_b0_best.pth` | 16 MB | **Yes (in repo)** |
| **Video** | Frame-CNN + BiGRU Temporal Aggregator | `checkpoints/video/frame_cnn___bigru_temporal_aggregator_best.pth` | 906 KB | **Yes (in repo)** |
| **Audio** | CLAD AASIST Graph Attention | `checkpoints/audio/clad_full_model_aasist___contrastive___lengthloss_best.pth` | 702 KB | **Yes (in repo)** |

### Oversized Research Models
Heavy experimental models (>100MB) evaluated during research are excluded from Git to comply with repository limits:
- `checkpoints/image/fa_vit_gam_lam_fft_fal_best.pth` (333 MB)
- `checkpoints/image/vanilla_vit_b_16_best.pth` (328 MB)
- `checkpoints/video/style_latent_flow_video_model_full_best.pth` (325 MB)
- `checkpoints/image/resnet_50_best.pth` (91 MB)

These models can be trained or retrieved using `scripts/train_*_experiments.py`.

---

## Automated Testing Suite

Execute the complete automated PyTest test suite (unit tests, API endpoints, multimodal inference pipelines):
```bash
PYTHONPATH=. venv/bin/pytest tests/
```

All 13 integration and end-to-end tests validate:
1. `GET /api/health` device acceleration and model status.
2. `POST /api/analyze/image` end-to-end analysis, heatmap generation, and DB storage.
3. `POST /api/analyze/audio` CLAD spectro-temporal analysis.
4. `GET /api/cases` history retrieval.
5. `GET /api/reports/{case_id}` ReportLab forensic PDF creation.
6. Multimodal inference engine direct calls across Image, Video, and Audio.

---

## Deployment Architecture

- **Frontend**: Deployed to **Vercel** with global edge CDN distribution at [https://veriface-tau.vercel.app](https://veriface-tau.vercel.app).
- **Backend**: Containerized via `Dockerfile` and configured with `Procfile` for hosting on any Python-compatible cloud platform (Render, Hugging Face Spaces, Railway, or VPS).
- **Connection**: Set `VITE_API_BASE_URL` in Vercel to point the frontend to the deployed backend.

For step-by-step cloud deployment instructions, consult [DEPLOYMENT.md](DEPLOYMENT.md).

---

## Known Limitations

1. **Vercel Serverless Function Limits**: The Python ML backend requires ~1.5 GB of binary libraries (PyTorch, OpenCV, Librosa, FFmpeg) and dedicated RAM (~1 GB peak), which exceeds Vercel Serverless Function package limits (50 MB zipped). Therefore, the backend must be hosted on a dedicated Python service.
2. **Video Upload Limits**: Single media uploads are capped at 100 MB.
3. **FFmpeg Requirement**: Late audiovisual fusion on uploaded videos requires FFmpeg installed on the backend server. If FFmpeg is unavailable, video visual inference proceeds with a graceful fallback.

---

## Research References

1. **FA-ViT (Image)**: Luo et al., *"Forgery-Aware Adaptive Learning With Vision Transformer for Generalized Face Forgery Detection"*, IEEE TCSVT, 2025.
2. **Style Latent Flows (Video)**: Choi et al., *"Exploiting Style Latent Flows for Generalizing Deepfake Video Detection"*, IEEE/CVF CVPR, 2024.
3. **CLAD (Audio)**: Wu et al., *"CLAD: Robust Audio Deepfake Detection Against Manipulation Attacks with Contrastive Learning"*, arXiv:2404.15854, 2024.
4. **AASIST (Audio)**: Jung et al., *"AASIST: Audio Anti-Spoofing Using Integrated Spectro-Temporal Graph Attention Networks"*, IEEE ICASSP, 2022.

---

## License

Academic and research use only. All rights reserved.
