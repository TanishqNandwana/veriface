# VeriFace — Project Instructions for Claude Code

## Project

**VeriFace — Verify the Media. Reveal the Truth.**

VeriFace is a unified multimodal AI platform for detecting manipulated images, deepfake videos, and synthetic/cloned audio.

The user interacts with one web application and selects:

- Image
- Video
- Audio

The application routes the media to the appropriate detection branch and returns a verdict, confidence, and explainable forensic evidence.

---

# 1. Non-Negotiable Source of Truth

Before writing substantial code, inspect the three files in the project root:

```text
mix.pdf
veriface 1st ppt(2).pptx
Interim Report Format.doc
```

Read/review them completely enough to understand:

- the three base-paper methodologies;
- proposed system architecture;
- technologies;
- objectives;
- research gaps;
- project timeline;
- report structure.

Do not replace source-defined terminology with unrelated terminology.

The three principal methodologies are:

### Image
**Forgery-Aware Adaptive Vision Transformer (FA-ViT)**

- frozen pretrained ViT
- Global Adaptive Module (GAM)
- Local Adaptive Module (LAM)
- Fine-grained Adaptive Learning (FAL)
- project PPT additionally proposes FFT forensic cues
- face detection/alignment
- Grad-CAM-style explainability

### Video
**Style Latent Flow / StyleGRU framework**

- pSp style latent extraction
- temporal style differences
- StyleGRU
- 3D ResNet-50 content branch
- Style Attention Module
- Temporal Transformer Encoder
- frame/video-level output
- temporal/frame explanation

### Audio
**CLAD**

- AASIST encoder
- MoCo-style contrastive learning
- manipulation augmentations
- length loss
- downstream classifier
- segment-level explainability

---

# 2. Research Goals

The implementation must answer these project questions:

1. Can a unified architecture detect image, video and audio deepfakes?
2. Which implementation of each branch provides the strongest generalization?
3. How well do the models perform on unseen datasets?
4. How robust are they to compression and common manipulations?
5. Can the system provide meaningful forensic explanations?
6. Can all three branches be integrated into one reliable web application?

---

# 3. Expected System Architecture

```text
                    ┌──────────────────────┐
                    │      VeriFace UI     │
                    │ Image / Video /Audio │
                    └──────────┬───────────┘
                               │
                         Media Type Router
                               │
        ┌──────────────────────┼──────────────────────┐
        │                      │                      │
        ▼                      ▼                      ▼
 ┌─────────────┐        ┌─────────────┐        ┌─────────────┐
 │ IMAGE       │        │ VIDEO       │        │ AUDIO       │
 │ FA-ViT      │        │ Style Flow  │        │ CLAD        │
 │ GAM/LAM/FAL │        │ StyleGRU    │        │ AASIST      │
 │ FFT cue     │        │ 3D ResNet   │        │ MoCo + LL   │
 └──────┬──────┘        │ SAM + TTE   │        └──────┬──────┘
        │               └──────┬──────┘               │
        ▼                      ▼                      ▼
 Image score              Video score             Audio score
 Grad-CAM                  Frame scores            Segment scores
 Heatmap                   Temporal evidence      Saliency
        │                      │                      │
        └──────────────────────┼──────────────────────┘
                               ▼
                    ┌─────────────────────┐
                    │ Decision / Fusion   │
                    │ Real / Manipulated  │
                    │ Confidence          │
                    └──────────┬──────────┘
                               ▼
                 ┌──────────────────────────┐
                 │ Forensic Results         │
                 │ Heatmaps / Frames /      │
                 │ Audio segments / Report  │
                 └──────────┬───────────────┘
                            ▼
                 PostgreSQL Case Database
```

For video containing an audio track, the architecture may additionally run the audio branch and perform media-aware late fusion. Keep component scores separate so users can see whether visual or audio evidence drove the decision.

---

# 4. First Task: Repository Reconnaissance

Do not start by immediately writing hundreds of files.

First:

1. Inspect the complete repository.
2. Inspect all provided project documents.
3. Build a requirements matrix.
4. Build a research/experiment matrix.
5. Identify what already exists.
6. Identify missing components.
7. Create a phased implementation plan.
8. Start executing the plan immediately.

Create:

```text
docs/requirements_matrix.md
docs/architecture.md
docs/experiment_plan.md
docs/dataset_plan.md
docs/implementation_plan.md
```

Do not ask the user to approve this plan.

---

# 5. Phased Execution

## Phase 0 — Environment

Create an isolated Python environment.

Validate:

- Python
- PyTorch
- MPS availability
- OpenCV
- Pillow
- audio libraries
- FastAPI
- PostgreSQL connectivity
- Node/npm
- React build
- FFmpeg availability

Because the primary local environment is Apple Silicon/M4:

- prefer PyTorch MPS;
- provide CPU fallback;
- do not require NVIDIA CUDA for local development;
- isolate CUDA-only research code rather than breaking the whole project.

Create:

```text
scripts/check_environment.py
```

---

## Phase 1 — Data

Implement dataset preparation scripts.

Every dataset must have:

```text
raw/
processed/
manifests/
metadata/
```

Never mix source data with generated artifacts.

Implement:

- dataset scanners
- duplicate detection
- media validation
- face extraction
- frame sampling
- audio extraction
- train/val/test manifest generation
- class-balance reports

Prevent identity/video leakage wherever the dataset protocol requires subject-level separation.

---

# 6. Image Model Development

Implement the FA-ViT branch first as the initial vertical slice.

Required modules:

```text
ml/image/
├── datasets.py
├── preprocessing.py
├── vit_backbone.py
├── gam.py
├── lam.py
├── fal.py
├── fft_features.py
├── favit.py
├── losses.py
├── train.py
├── evaluate.py
├── explain.py
└── inference.py
```

Start with a tiny end-to-end dataset smoke test.

Then implement controlled experiments.

Do not fine-tune the pretrained ViT by default if reproducing FA-ViT. Preserve the frozen-backbone methodology.

Compare:

- ViT baseline
- ViT + GAM
- ViT + LAM
- ViT + GAM + LAM
- full FA-ViT + FAL
- full model + FFT cue

Use appropriate face detection and alignment.

Evaluate both clean and compressed conditions.

Prioritize cross-dataset tests.

---

# 7. Video Model Development

Implement:

```text
ml/video/
├── datasets.py
├── preprocessing.py
├── frame_sampler.py
├── face_cropper.py
├── psp_encoder.py
├── style_flow.py
├── style_gru.py
├── content_3d_resnet.py
├── style_attention.py
├── temporal_transformer.py
├── model.py
├── losses.py
├── train.py
├── evaluate.py
├── explain.py
└── inference.py
```

Use the base-paper strategy as the primary implementation.

Support configurable frame counts for smoke tests and full experiments.

Default research configuration should reflect the base-paper setup when computationally practical.

Benchmark against simpler baselines before claiming the proposed architecture is superior.

---

# 8. Audio Model Development

Implement:

```text
ml/audio/
├── datasets.py
├── preprocessing.py
├── augmentations.py
├── aasist_encoder.py
├── moco.py
├── length_loss.py
├── clad.py
├── train_pretrain.py
├── train_downstream.py
├── evaluate.py
├── explain.py
└── inference.py
```

Implement CLAD as a two-stage pipeline:

```text
Stage 1:
AASIST encoder
→ two manipulated views
→ query/key encoders
→ contrastive loss
→ negative queue
→ length loss

Stage 2:
pretrained encoder
→ linear classifier
→ downstream cross-entropy training
```

Use configurable manipulation strengths.

The paper's reported configuration may be used as the starting baseline, but all project results must be independently measured.

---

# 9. Benchmarking

Create a unified experiment runner:

```text
scripts/run_experiment.py
```

Every run must produce a unique experiment ID.

Support:

```text
--modality image|video|audio
--model ...
--dataset ...
--split ...
--seed ...
--epochs ...
--batch-size ...
--device ...
--config ...
```

Create:

```text
experiments/
reports/
```

with machine-readable metrics.

Never rely on manually copied numbers.

---

# 10. Model Selection

Create:

```text
reports/model_selection.md
```

For each modality show:

| Model | Dataset | Accuracy | Precision | Recall | F1 | AUC | EER/FAR/FRR | Cross-Dataset | Robustness | Latency |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|

Select the final model based on generalization + robustness + explainability + efficiency, not clean accuracy alone.

---

# 11. Explainability

Implement real evidence generation.

### Image

Output:

```text
original image
face crop
heatmap
overlay
prediction
confidence
```

### Video

Output:

```text
video score
frame scores
top suspicious frames
temporal confidence chart
frame heatmaps
```

### Audio

Output:

```text
audio score
segment scores
suspicious timestamps
spectrogram/waveform saliency
```

Store artifacts using the case ID.

---

# 12. Unified Backend

Use FastAPI.

Create a service layer so API routes do not contain model logic.

Recommended:

```text
backend/
├── main.py
├── api/
├── services/
├── schemas/
├── db/
├── models/
├── workers/
├── storage/
└── utils/
```

Long video/audio analysis should be handled asynchronously if required.

Never block the web server with unnecessary long-running inference.

---

# 13. Frontend

Use React.

The interface should include:

### Landing page
- VeriFace branding
- concise explanation
- three media options

### Upload page
- drag/drop
- file validation
- preview
- analyze button
- processing status

### Results page
- Real / Manipulated verdict
- confidence
- model used
- evidence summary
- heatmap/frame/audio visualization
- technical details
- download forensic report

### Case history
- previous analyses
- media type
- verdict
- confidence
- timestamp
- report access

The UI must communicate uncertainty honestly.

---

# 14. Forensic Report

Generate a PDF report for every completed analysis.

Report should contain:

1. VeriFace case information
2. Input media metadata
3. Hash
4. Model information
5. Verdict
6. Confidence
7. Component scores
8. Explainability evidence
9. Suspicious regions/frames/time ranges
10. Processing information
11. Limitations/disclaimer
12. Reproducibility/model version

Do not state that a confidence percentage is a probability of objective truth unless calibration validates that interpretation.

---

# 15. Interim Report

The supplied `Interim Report Format.doc` is mandatory.

Create a report draft following its structure:

```text
1 INTRODUCTION
1.1 Background
1.2 Motivation and Scope
1.3 Problem Statement
1.4 Proposed Solution
1.5 Organization of Report

2 LITERATURE SURVEY
2.1 Review of current work
2.2 Identifying gaps

3 METHODOLOGY
3.1 System architecture/Flowcharts
3.2 Hardware and Software description
3.3 Algorithm/Data structures

4 RESULT AND ANALYSIS

5 ADVANTAGE, LIMITATIONS AND APPLICATIONS
5.1 Advantages
5.2 Limitations
5.3 Applications

6 CONCLUSION AND FUTURE SCOPE
6.1 Conclusions
6.2 Future Scope

References
```

Maintain an interim report draft throughout development rather than writing it only at the end.

---

# 16. Research Writing Rules

Use academic language.

Never copy large passages from the papers.

When discussing published results:

- identify them as reported results;
- cite the relevant paper;
- do not present them as VeriFace results.

When discussing project experiments:

- identify the experiment;
- provide dataset;
- provide model;
- provide metric;
- provide checkpoint/config reference.

---

# 17. Testing

Create:

```text
tests/
├── test_image_preprocessing.py
├── test_video_preprocessing.py
├── test_audio_preprocessing.py
├── test_image_model.py
├── test_video_model.py
├── test_audio_model.py
├── test_api.py
├── test_database.py
└── test_end_to_end.py
```

Every model must have a tiny deterministic inference test.

---

# 18. Security

The application must safely process uploads.

Implement:

- extension validation
- MIME validation
- size limits
- filename sanitization
- generated unique storage names
- media decoding validation
- safe temporary-file cleanup
- path traversal protection
- no secret keys in source control

---

# 19. Definition of Done

Do not say "complete" until:

- image pipeline works;
- video pipeline works;
- audio pipeline works;
- models can load from saved checkpoints;
- web router selects correct branch;
- API works;
- frontend works;
- PostgreSQL records cases;
- explainability artifacts are generated;
- forensic PDF report is generated;
- tests pass;
- experiment records exist;
- interim report is updated;
- README contains reproducible setup instructions.

---

# 20. Autonomous Mode

Operate continuously through the entire implementation.

Do not ask routine "Should I continue?", "May I install this?", "Can I create this file?", or "Do you want me to run the tests?" questions.

Make reasonable engineering decisions within this specification and continue.

If a dependency fails, diagnose and fix it.

If a model cannot run on MPS, create the closest reproducible CPU/MPS-compatible implementation without changing the research objective.

If a dataset is unavailable, record it and continue with the available benchmark plan.

If a full training run is too large for the local machine, first run a smaller scientifically valid smoke/ablation experiment, then continue with the largest reproducible configuration supported by the hardware.

Never fabricate a result simply to complete a table.

---

# 21. Final Deliverables

The repository must ultimately contain:

```text
Working VeriFace web application
Three trained/evaluated modality models
Model comparison report
Experiment artifacts
Explainability artifacts
FastAPI backend
React frontend
PostgreSQL schema
Docker configuration
Automated tests
Forensic PDF generation
Interim report
Final technical documentation
README
Reproducibility instructions
```

The project should be demonstrable from one command sequence documented in the README.
