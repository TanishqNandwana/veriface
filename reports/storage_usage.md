# VeriFace — Storage Usage and Budget Allocation Report

**Date:** October 12, 2026  
**Hardware Host:** Apple Silicon M4 Mac (Darwin arm64)  
**Primary Storage Location:** `~/VeriFaceStorage/` (`/Users/tanishqnandwana/VeriFaceStorage/`)  
**Workspace Location:** `/Users/tanishqnandwana/Desktop/Veriface/`

---

## 1. Storage Baseline & Global System Capacity

- **Total Disk Volume:** 228.27 GB (`/System/Volumes/Data`)
- **Current Used Space:** 185.33 GB
- **Current Available Free Space:** **42.94 GB**
- **Safety Margin:** Minimum 20.0 GB reserved for macOS system operation, virtual memory paging, and application caching.

---

## 2. ML Asset Storage Directory Hierarchy

To prevent cluttering the project codebase and ensure strict disk management, external machine learning assets are centralized in `~/VeriFaceStorage/`:

```
~/VeriFaceStorage/
├── datasets/       # Raw archives, extracted data partitions, manifests
├── checkpoints/    # Trained PyTorch model weights (.pth)
├── experiments/    # Experiment JSON metrics, loss curves, confusion matrices
└── cache/          # Temporary downloaded chunks and preprocessed cache
```

---

## 3. Storage Budget Allocation Plan

| Storage Category | Target Budget Limit | Current Consumption | Planned Ingestion | Post-Ingestion Projected |
|---|---|---|---|---|
| **Datasets & Preprocessing** | 12.0 – 16.0 GB | ~0.24 GB (existing benchmarks) | ~1.5 – 3.0 GB (Zenodo Image + ASVspoof LA + FF++ subset) | ~2.5 – 3.5 GB |
| **Model Checkpoints** | 4.0 – 6.0 GB | ~1.15 GB (existing 10 models) | ~0.5 – 1.0 GB (new trained candidate weights) | ~1.8 – 2.2 GB |
| **Experiment Artifacts & Cache** | 1.0 – 2.0 GB | ~0.02 GB | ~0.50 GB (temporary downloads) | ~0.55 GB |
| **Reserved System Margin** | ≥ 20.0 GB | Active (42.94 GB free) | Safeguard threshold strictly preserved | ≥ 38.0 GB free |

---

## 4. Pre-Download Storage Estimation Model

Before downloading any dataset, storage requirements are calculated as follows:

$$\text{Peak Storage} = \text{Download Archive Size} + \text{Extracted Size} + \text{Preprocessed Size}$$

| Candidate Dataset | Modality | Compressed Size | Extracted Size | Preprocessed Size | Peak Storage | Status / Decision |
|---|---|---|---|---|---|---|
| **Real vs Fake Faces (Zenodo)** | Image | 1.07 GB | 1.15 GB | 0.85 GB | **2.22 GB** | **Approved:** High relevance, fully open, well within budget. |
| **ASVspoof 2019 (LA Subset)** | Audio | ~0.25 – 0.50 GB | ~0.50 GB | ~0.15 GB | **0.80 GB** | **Approved:** High relevance, compact audio benchmark. |
| **DFDC / FF++ Curated Subset** | Video/Image | ~0.50 – 1.00 GB | ~1.20 GB | ~0.40 GB | **1.70 GB** | **Approved:** Controlled frame volume prevents storage exhaustion. |
| **Full FaceForensics++ (Raw)** | Video | > 500 GB | > 1000 GB | N/A | > 1000 GB | **Rejected:** Exceeds total disk capacity. |
| **Full DFDC (Kaggle)** | Video | 470 GB | 550 GB | N/A | > 500 GB | **Rejected:** Exceeds total disk capacity. |

---

## 5. Active Storage Audit of Workspace Assets

```
/Users/tanishqnandwana/Desktop/Veriface/
├── datasets/processed/
│   ├── image_benchmark/       4.76 MB (400 face crops)
│   ├── video_benchmark/     218.21 MB (76 clips)
│   └── audio_benchmark/      12.21 MB (100 audio clips)
├── checkpoints/
│   ├── image/               822.5 MB (EfficientNet, ResNet, ViT, FA-ViT, Combined Multi-Dataset)
│   ├── video/               346.5 MB (Frame-BiGRU, 3D ResNet, StyleFlow)
│   └── audio/                 2.7 MB (CLAD, AASIST, Spectrogram CNN, Combined Multi-Dataset)
└── experiments/             ~15.0 MB (JSON metrics, cross-dataset evals, confusion matrices)
-------------------------------------------------------------------------
Total Workspace ML Footprint: ~1.40 GB
```

---

## 6. Actual Post-Ingestion Storage Footprint (`~/VeriFaceStorage/`)

| Asset Directory | Contents & Sample Count | Actual Storage Size |
|---|---|---|
| `~/VeriFaceStorage/datasets/asvspoof2019_la/` | `0.parquet` ASVspoof 2019 LA partition | 463.7 MB |
| `~/VeriFaceStorage/datasets/asvspoof_benchmark/` | 100 standardized 16kHz 4s evaluation clips | 6.4 MB |
| `~/VeriFaceStorage/datasets/dfdc_video_benchmark/` | 20 MTCNN-extracted 5-frame temporal tensor sequences | 24.1 MB |
| `~/VeriFaceStorage/datasets/ffhq_stylegan_benchmark/` | `val_0.parquet` + 200 face crops (100 Real FFHQ, 100 Fake StyleGAN) | 277.2 MB |
| `~/VeriFaceStorage/datasets/real_vs_fake_faces/` | Zenodo Record 14532968 extracted dataset (6040, 7030, 8020, 7525 splits) | 1.10 GB |
| **Total Ingested in `~/VeriFaceStorage/`** | **Multi-dataset ML storage (Temporary zip pruned)** | **~1.87 GB** |
| **Current Available Free Space on Mac** | **`/System/Volumes/Data`** | **43.0 GB free** |

**Budget Execution Assessment:**  
The multi-dataset ingestion consumed only **1.87 GB** of disk space (vastly below the 12–16 GB conservative ceiling). The system safety headroom was completely preserved (43.0 GB free $\gg$ 20.0 GB threshold). The downloaded 1.07 GB zip archive was automatically pruned after verifying extraction integrity.
