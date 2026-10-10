You are the autonomous lead engineer and research agent for my capstone project:

# VeriFace — Verify the Media. Reveal the Truth.

I want you to build the COMPLETE VeriFace system from scratch, not just a UI prototype.

## PROJECT DESCRIPTION

VeriFace is a unified AI-based deepfake detection platform.

The web application gives the user three choices:

1. Image
2. Video
3. Audio

After the user uploads media, the system automatically routes it to the correct detection model and returns:

- Real / Manipulated verdict
- Confidence score
- Model score/probability
- Explainability evidence
- Image heatmap / suspicious regions
- Suspicious video frames and temporal evidence
- Suspicious audio segments/time ranges
- Model and processing information
- Case history
- Downloadable forensic PDF report

For videos containing an audio track, the system may also use the audio branch and perform media-aware late fusion. Keep visual and audio component scores separately visible.

## FILES IN THIS PROJECT

The project folder contains:

- `mix.pdf` — combined base papers
- `veriface 1st ppt(2).pptx` — VeriFace first-phase project presentation
- `Interim Report Format.doc` — mandatory interim report format
- `AGENT.md`
- `HOOK.md`
- `CLAUDE.md`

READ ALL OF THESE BEFORE IMPLEMENTING THE SYSTEM.

The three main base-paper methodologies are:

### IMAGE — FA-ViT
Forgery-Aware Adaptive Vision Transformer:

- face detection/alignment
- 224×224 face crops
- frozen pretrained ViT-B/16
- Global Adaptive Module (GAM)
- Local Adaptive Module (LAM)
- Fine-grained Adaptive Learning (FAL)
- project PPT also specifies an FFT forensic cue
- classification
- Grad-CAM/explainability

### VIDEO — Style Latent Flow
Style-latent-flow-based deepfake detection:

- face preprocessing
- frame sampling
- pSp latent extraction
- temporal style differences
- StyleGRU
- 3D ResNet-50 content features
- Style Attention Module
- Temporal Transformer Encoder
- real/fake classification
- frame/temporal explanation

### AUDIO — CLAD
Contrastive Learning-based Audio Deepfake Detector:

- 16 kHz waveform
- AASIST encoder
- MoCo-style contrastive learning
- negative queue
- manipulation augmentations
- length loss
- downstream classifier
- segment-level explanation

## MAIN RESEARCH OBJECTIVE

Do not simply reproduce one paper.

I want you to:

1. Build the three branches.
2. Train/evaluate different reasonable model variants.
3. Benchmark the proposed architectures against suitable baselines.
4. Test intra-dataset and cross-dataset generalization.
5. Test robustness against realistic media perturbations.
6. Compare the models quantitatively.
7. Select the strongest final model for each modality using a documented selection criterion.
8. Integrate the selected models into one unified web application.
9. Add real explainability.
10. Generate forensic reports.
11. Generate and maintain the interim report.
12. Produce a reproducible final project.

## DATASETS

Use the datasets specified or referenced by the base papers/PPT where available.

Potential image/video datasets include:

- FaceForensics++
- Celeb-DF-v2
- WildDeepfake
- DFDC
- DFDC Preview
- DeepFakeDetection
- DeeperForensics-1.0
- FFIW-10K

Audio:

- ASVspoof 2019 LA
- other permitted audio datasets only if they improve generalization evaluation

Audio robustness may use ESC-50 environmental noise where permitted.

Do NOT fabricate dataset availability or results.

Do NOT silently use private/restricted datasets.

Create dataset manifests and prevent data leakage.

## EXPERIMENTS

For every modality, compare appropriate baselines and proposed variants.

Track:

- Accuracy
- Precision
- Recall
- F1
- ROC-AUC
- PR-AUC where useful
- EER where applicable
- FAR
- FRR
- cross-dataset performance
- robustness
- inference latency
- memory usage
- model size

Do not select the model based only on training accuracy.

Final model selection should prioritize:

1. cross-dataset generalization
2. F1/AUC
3. robustness
4. explainability
5. inference efficiency
6. reproducibility

Create `reports/model_selection.md`.

## IMAGE EXPERIMENT PLAN

At minimum investigate:

- ViT baseline
- ViT + GAM
- ViT + LAM
- ViT + GAM + LAM
- FA-ViT + FAL
- FFT cue ablation
- compression robustness
- cross-dataset generalization

## VIDEO EXPERIMENT PLAN

Investigate:

- Xception baseline
- CNN-GRU
- 3D CNN baseline
- StyleGRU
- StyleGRU + 3D ResNet
- StyleGRU + SAM
- full StyleGRU + SAM + TTE

Where practical, compare with the methods discussed in the paper such as LipForensics, FTCN and AltFreeze.

Prioritize cross-dataset generalization.

## AUDIO EXPERIMENT PLAN

Investigate:

- RawNet2
- AASIST
- Res-TSSDNet
- SAMO
- AASIST with augmentation
- CLAD without length loss
- CLAD without contrastive learning
- full CLAD

Test manipulation robustness including:

- noise
- environmental noise
- volume control
- fading
- time stretching
- resampling
- time shifting
- echo

## LOCAL HARDWARE

My primary local development machine is an Apple Silicon M4 Mac.

Therefore:

- use PyTorch MPS where supported;
- provide CPU fallback;
- do not assume NVIDIA CUDA exists;
- detect device automatically;
- avoid CUDA-only assumptions;
- use small smoke tests before long runs;
- reduce batch size automatically when necessary;
- keep the research methodology intact.

Do not blindly reproduce obsolete research environments. Adapt dependencies carefully inside an isolated environment.

## REQUIRED SOFTWARE STACK

Follow the project PPT:

- Python 3.x
- PyTorch
- OpenCV
- Pillow
- MTCNN/RetinaFace or the appropriate face detector required by the model
- Librosa/Torchaudio
- FastAPI
- React
- PostgreSQL
- Grad-CAM or architecture-compatible explanation method
- Docker

## APPLICATION

Build a polished but technically honest web application.

Pages/components:

### Home
- VeriFace branding
- project description
- Image / Video / Audio options

### Upload
- drag and drop
- file validation
- preview
- analyze button
- progress indicator

### Results
- verdict
- confidence
- model
- evidence
- heatmap
- suspicious frames
- temporal/audio evidence
- component scores
- technical metadata
- report download

### History
- case ID
- media type
- verdict
- confidence
- date/time
- report

## BACKEND

Use FastAPI with separate model/service layers.

Conceptual endpoints:

- POST `/api/analyze/image`
- POST `/api/analyze/video`
- POST `/api/analyze/audio`
- GET `/api/cases`
- GET `/api/cases/{case_id}`
- GET `/api/reports/{case_id}`
- GET `/api/health`

Use PostgreSQL for case logs.

Use asynchronous/background processing for long-running video/audio inference where appropriate.

## EXPLAINABILITY

Do not generate decorative or random heatmaps.

Image:
- real model-derived Grad-CAM/attribution
- suspicious region
- overlay

Video:
- frame scores
- suspicious frames
- temporal score curve
- valid frame heatmaps where supported

Audio:
- segment scores
- suspicious timestamps
- waveform/spectrogram saliency where supported

If an exact heatmap method is not technically valid for a model, use a valid alternative and document it.

Never claim pixel-level localization if the model only produces image-level evidence.

## FORENSIC REPORT

Generate a PDF containing:

1. Case information
2. Input metadata
3. File hash
4. Media type
5. Model/checkpoint
6. Verdict
7. Confidence
8. Component scores
9. Explainability evidence
10. Suspicious regions/frames/audio intervals
11. Processing information
12. Limitations
13. Reproducibility information

## INTERIM REPORT

Use the attached `Interim Report Format.doc` exactly as the structural reference.

Required sections:

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

Maintain the report as the project progresses.

If an experiment is not complete, explicitly mark it as pending rather than inventing a result.

## DOCUMENTATION

Create and maintain:

- `README.md`
- `docs/architecture.md`
- `docs/requirements_matrix.md`
- `docs/experiment_plan.md`
- `docs/dataset_plan.md`
- `docs/reproducibility.md`
- `reports/model_selection.md`
- `reports/experiments_summary.md`
- interim report
- final technical report

## EXPERIMENT STORAGE

Every experiment must have a unique ID and store:

- configuration
- seed
- dataset
- split
- device
- hyperparameters
- training log
- checkpoint
- metrics
- predictions
- plots
- notes

Never overwrite completed experiments.

## AUTONOMOUS EXECUTION

Work autonomously.

DO NOT repeatedly ask me:

- "Should I continue?"
- "Can I create this file?"
- "May I install this dependency?"
- "Should I run the tests?"
- "Would you like me to proceed?"
- "Do you want me to implement the next phase?"

The answer to routine implementation decisions is already YES.

Create files, directories, code, tests, configs, reports and experiment scripts without asking for routine permission.

When something fails:

1. diagnose it;
2. fix it;
3. retry;
4. record the issue;
5. continue.

Do not repeatedly retry the same command without changing the diagnosis.

Only stop when a genuinely blocking issue requires information/credentials/access that cannot be inferred from the supplied project files.

## IMPORTANT RESEARCH INTEGRITY RULE

Never fabricate:

- model accuracy
- AUC
- F1
- EER
- FAR
- FRR
- dataset counts
- completed experiments
- heatmaps
- training results

Clearly label:

- paper-reported results
- reproduced results
- new VeriFace results
- pending experiments

## IMPLEMENTATION ORDER

Follow this order unless a dependency requires a different sequence:

### STEP 1
Read all project files.

### STEP 2
Inspect repository and create requirements/architecture/experiment plans.

### STEP 3
Validate M4 development environment.

### STEP 4
Build image branch end-to-end.

### STEP 5
Run image baseline and FA-ViT experiments.

### STEP 6
Build video branch.

### STEP 7
Run video baseline and StyleGRU experiments.

### STEP 8
Build audio branch.

### STEP 9
Run AASIST/RawNet2/Res-TSSDNet/SAMO/CLAD experiments.

### STEP 10
Compare and select final models.

### STEP 11
Implement explainability for selected models.

### STEP 12
Implement FastAPI backend.

### STEP 13
Implement React frontend.

### STEP 14
Implement PostgreSQL case logging.

### STEP 15
Implement forensic PDF reports.

### STEP 16
Run full end-to-end tests.

### STEP 17
Update interim report with actual results.

### STEP 18
Create final documentation and reproducibility instructions.

## DEFINITION OF DONE

The project is complete only when:

- all three modalities can be uploaded;
- each modality routes to the correct model;
- trained checkpoints can be loaded;
- predictions are returned;
- confidence is displayed;
- explanations are generated;
- cases are stored;
- forensic reports are generated;
- API works;
- frontend builds;
- tests pass;
- experiment records exist;
- model selection is documented;
- interim report is updated;
- README provides reproducible setup and execution instructions.

START NOW.

First inspect the repository and the three supplied source documents, create the implementation/research plan, and then execute the plan continuously without asking for routine permission.
