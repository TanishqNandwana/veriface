# VeriFace — Verify the Media. Reveal the Truth.

VeriFace is a state-of-the-art multimodal AI deepfake detection platform capable of analyzing manipulated **Images**, deepfake **Videos**, and synthetic/cloned **Audio** attacks. It presents fine-grained forensic explanations (Grad-CAM spatial heatmaps, temporal frame timelines, audio segment saliencies) and generates downloadable PDF forensic reports.

---

## Key Features

1. **Multimodal Detection Architecture**:
   - **Image Branch**: Forgery-Aware Adaptive Vision Transformer (**FA-ViT**) with Global Adaptive Module (GAM), Local Adaptive Module (LAM), Fine-grained Adaptive Learning (FAL) loss, and FFT high-frequency spectral forensic cues.
   - **Video Branch**: Style Latent Flow framework featuring pSp W+ latent space inversion, temporal style differences ($\Delta S_t$), StyleGRU, 3D ResNet-50 visual content representations, Style Attention Module (SAM), and Temporal Transformer Encoder (TTE).
   - **Audio Branch**: Contrastive Learning Audio Deepfake Detector (**CLAD**) combining AASIST spectro-temporal graph attention, MoCo-style contrastive pretraining under manipulation augmentations, length loss, and downstream classification.
   - **Media-Aware Late Fusion**: Automatically extracts video audio tracks and performs late fusion (60% Visual, 40% Audio) when evaluating video clips.

2. **Full Stack Platform**:
   - **Backend**: FastAPI server with async file processing, SQLite/PostgreSQL persistence, and automated PDF forensic report generation via ReportLab.
   - **Frontend**: Glassmorphism React SPA with live drag-and-drop uploader, real-time progress indicators, interactive forensic evidence visualizers, and filterable case history.
   - **Hardware Acceleration**: Automatic PyTorch device detection supporting Apple Silicon GPU (`mps`), NVIDIA CUDA, and CPU fallback.

---

## Quick Start (Single-Command Execution)

### 1. Prerequisites
- Python 3.8+
- Node.js 18+ and npm
- FFmpeg (optional, required for video audio track extraction)

### 2. Setup & Environment
```bash
# Clone or navigate to VeriFace root
cd Veriface

# Create virtual environment & install requirements
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Verify environment & hardware acceleration (MPS/CPU)
python scripts/check_environment.py
```

### 3. Run Automated Tests
```bash
# Execute unit & end-to-end integration tests
PYTHONPATH=. venv/bin/pytest tests/
```

### 4. Launch Application Server
```bash
# Start FastAPI backend (serves API and React frontend on http://localhost:8000)
PYTHONPATH=. venv/bin/python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000
```

Open [http://localhost:8000](http://localhost:8000) in your web browser.

---

## Project Structure

```text
veriface/
├── backend/                  # FastAPI Web Backend & Database
│   ├── api/router.py         # REST API routes (/api/analyze/image, video, audio, health)
│   ├── db/                   # Database schemas (models.py, database.py)
│   ├── services/             # ReportLab PDF generator (report_service.py)
│   └── main.py               # FastAPI main server entry point
├── frontend/                 # React Web Application
│   ├── src/                  # React components & UI (App.jsx, ResultView.jsx, Uploader.jsx)
│   └── dist/                 # Production compiled frontend bundle
├── ml/                       # Machine Learning Research Pipelines
│   ├── image/                # FA-ViT (favit.py, gam.py, lam.py, fal_loss.py, fft_features.py)
│   ├── video/                # Style Latent Flows (model.py, psp_encoder.py, style_gru.py, sam.py)
│   ├── audio/                # CLAD (clad.py, aasist_encoder.py, length_loss.py, augmentations.py)
│   ├── common/               # Device selection, dataset builder, research metrics
│   └── inference/            # Master Multimodal Inference Engine (engine.py)
├── experiments/              # Immutable experiment logs & checkpoints
├── reports/                  # Model selection & interim academic reports
│   ├── model_selection.md    # Empirical metrics comparative table
│   └── interim_report.md     # Interim academic report following institutional format
├── scripts/                  # Environment check & utility scripts
├── tests/                    # Backend & End-to-End PyTest test suite
├── forensic_reports/         # Generated PDF forensic reports
└── storage/                  # Uploaded files & explainability artifacts
```

---

## Research References

1. **FA-ViT (Image)**: Luo et al., *"Forgery-Aware Adaptive Learning With Vision Transformer for Generalized Face Forgery Detection"*, IEEE TCSVT, 2025.
2. **Style Latent Flows (Video)**: Choi et al., *"Exploiting Style Latent Flows for Generalizing Deepfake Video Detection"*, IEEE/CVF CVPR, 2024.
3. **CLAD (Audio)**: Wu et al., *"CLAD: Robust Audio Deepfake Detection Against Manipulation Attacks with Contrastive Learning"*, arXiv:2404.15854, 2024.
4. **AASIST (Audio)**: Jung et al., *"AASIST: Audio Anti-Spoofing Using Integrated Spectro-Temporal Graph Attention Networks"*, IEEE ICASSP, 2022.
