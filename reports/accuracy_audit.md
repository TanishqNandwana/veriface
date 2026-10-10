# VeriFace Comprehensive Accuracy Audit & Implementation Diagnosis Report

**Project Title**: VERIFACE — Verify the Media. Reveal the Truth  
**Platform**: macOS Darwin (Apple Silicon M4, MPS GPU Accelerated)  
**Date**: October 2026  
**Status**: Pre-Optimization Forensic Audit & Bug Diagnosis  

---

## Executive Summary

An exhaustive technical audit of the VeriFace multimodal deepfake detection system was conducted across the Image, Video, and Audio detection branches. The investigation evaluated model architectures, dataset loaders, preprocessing pipelines, training regimes, inference scripts, evaluation protocols, and backend integration.

The primary finding is that **the current models suffer from severe domain detachment, heuristic interference, and synthetic data contamination**:
1. Previous evaluation metrics (such as 100% accuracy and 0.00 EER reported in `reports/model_selection.md`) were obtained exclusively on tiny synthetic toy datasets (synthetic sine-wave images, synthetic 4-frame numpy arrays, and synthetic sine audio waveforms) generated with NumPy mathematical functions (`np.sin`, `np.cos`, checkerboard patterns). Real face deepfake datasets (FaceForensics++, Celeb-DF, DFDC, ASVspoof) were never actually downloaded or trained on prior to this audit.
2. The **Video branch** had a catastrophic implementation flaw: while the `StyleLatentFlowVideoModel` executed forward inference on clip tensors, its output score `model_video_score` was **completely discarded** at inference time. The verdict was returned based solely on a heuristic blend of individual 2D frame image scores and spatial-temporal optical flow anomalies, rendering the video neural model inactive.
3. The **Image branch** suffered from an input distribution mismatch: training used uncropped full-frame images directly, whereas inference applied face detection and facial crop extraction. Furthermore, inference combined the ViT neural logit with a hardcoded heuristic Fourier anomaly score (`0.8 * neural + 0.2 * fft_anomaly`) whose hand-crafted thresholds triggered false alarms on natural photos.
4. The **Audio branch** combined neural predictions 50/50 with hardcoded heuristic derivative kurtosis and spectral flatness metrics, resulting in high false alarm rates on legitimate human speech with diverse emotional intonation or background acoustic noise.

This audit establishes the baseline facts, details the identified bugs, and sets forth the multi-dataset optimization roadmap.

---

## 1. Current Architecture for Each Branch

### 1.1 Image Modality: FA-ViT (Forgery-Aware Adaptive Vision Transformer)
* **Backbone**: Vision Transformer (`ViT-B/16`) pretrained on ImageNet-1K (`torchvision.models.vit_b_16(weights=ViT_B_16_Weights.DEFAULT)`). Patch size: 16×16, Embedding dimension: 768.
* **Adapter Modules**:
  * **Global Adaptive Module (GAM)**: Bottleneck MLP (`768 -> 64 -> 768`) with GELU activation and zero-initialized residual connection.
  * **Local Adaptive Module (LAM)**: 2D spatial convolution on unflattened token grid (`14×14`) using depthwise separable convolutions (`3×3` depthwise + `1×1` pointwise) with zero-initialized residual connection.
  * **FFT Frequency Extractor**: Computes 2D Discrete Fourier Transform power spectrum and projects via convolutional layers to a 128-dimensional embedding.
  * **Spatial Noise Extractor**: High-pass Laplacian filter (`[-4]` center kernel) followed by convolutional layers to extract sensor PRNU noise residuals (128-dimensional embedding).
* **Classification Head**: Concatenation of CLS token (768), FFT feature (128), and Spatial Noise feature (128) -> Linear layer (1024 -> 256) -> GELU -> Linear head (256 -> 1 logit).
* **Loss Function**: `FAViTLoss` = `BCEWithLogitsLoss` + `0.1 * FALCircleLoss` (Circle Loss with margin 0.25, gamma 64.0 for fine-grained feature separation).

### 1.2 Video Modality: Style Latent Flow Framework
* **Backbone Modules**:
  * **Style Latent Encoder (pSp-inspired)**: 2-stage CNN projecting frames to $18 \times 512$ latent space. *(Note: Uninitialized random weights, not a pretrained StyleGAN pSp encoder)*.
  * **Temporal Style Diffs**: Computes $\Delta S_t = S_{t+1} - S_t$ across consecutive frames.
  * **StyleGRU**: Bidirectional GRU processing style flow trajectory over time (hidden dimension: 128).
  * **Content 3D ResNet**: 3D CNN (`Conv3dBlock` hierarchy) capturing spatiotemporal appearance features across video clips $[B, 3, T, 224, 224]$.
  * **Style Attention Module (SAM)**: Cross-attention module aligning 3D visual content features with StyleGRU temporal flow representations.
  * **Temporal Transformer Encoder (TTE)**: Multi-head self-attention sequence encoder producing clip-level video logits and frame-level logits.
* **Loss Function**: Dual supervision: $L_{video} + 0.4 \times L_{frame}$ using `BCEWithLogitsLoss`.

### 1.3 Audio Modality: CLAD (Contrastive Learning Audio Detector)
* **Spectrogram Frontend**: 16 kHz mono audio resampled and padded/truncated to 4.0 seconds (64,000 samples). Converted to 80-channel log-Mel spectrogram $[1, 80, 250]$.
* **AASIST Spectro-Temporal Encoder**: 2-stage 2D convolutional stem (`Conv2d` $5\times5$ and $3\times3$) with GELU and BatchNorm, followed by a Graph Attention Network (GAT) modeling spectro-temporal adjacency nodes, producing a 128-dimensional latent representation.
* **MoCo Contrastive Projection**: MLP projection head (`128 -> 128 -> 128`) for dual-view contrastive learning with Length Loss regularizer.
* **Downstream Classifier**: 2-layer MLP (`128 -> 128 -> 1`) with GELU activation.

---

## 2. Existing Datasets

| Dataset Path | Modality | Samples | Description / Origin | Real / Fake Distribution |
|---|---|---|---|---|
| `datasets/synthetic_image_exp` | Image | 60 | Synthetically generated sine-wave images with grid artifacts (`ml/common/dataset_builder.py`) | 30 Real / 30 Fake |
| `datasets/synthetic_video_exp` | Video | 30 | Synthetically generated `.npy` arrays $[8, 3, 224, 224]$ with sinusoidal color shifts | 15 Real / 15 Fake |
| `datasets/synthetic_audio_exp` | Audio | 50 | Synthetically generated `.wav` files with pure tones, harmonics, and noise bursts | 25 Real / 25 Fake |
| `datasets/multimodal_accuracy_benchmark` | Multimodal | 20 (Img: 8, Aud: 8, Vid: 4) | In-memory synthetic evaluation samples for rapid smoke assertions | 50% Real / 50% Fake |
| `datasets/raw/` | All | 0 | Empty | N/A |
| `datasets/processed/` | All | 0 | Empty | N/A |

**Key Finding**: The repository contained zero genuine deepfake samples (no real human faces, no real DeepFaceLab/FaceSwap videos, no real TTS/vocoder audio). All previous "accuracy" numbers were generated on mathematical synthetic artifacts.

---

## 3. Existing Data Splits

* Splits were generated by `split_manifest()` in `ml/common/dataset_builder.py`:
  * Split ratio: 70% Train, 15% Validation, 15% Test.
  * Subject-ID grouping was implemented for synthetic subjects (`sub_real_0` through `sub_real_4`).
  * However, because the dataset consisted only of 60 synthetic samples, the test set contained only **12 samples** (6 real, 6 fake).
  * A sample size of $N=12$ is statistically meaningless for generalizing to real-world deepfakes.

---

## 4. Existing Evaluation Results

Existing results reported in `experiments/*/metrics.json`:

| Experiment ID | Modality | Train Samples | Test Samples | Accuracy | ROC-AUC | F1-Score | EER | Test Notes |
|---|---|---|---|---|---|---|---|---|
| `IMG_FAVIT_SMOKE_001` | Image | 42 | 12 | 1.0000 | 1.0000 | 1.0000 | 0.0000 | Tested on 6 synthetic real / 6 synthetic fake images |
| `VID_STYLEFLOW_SMOKE_001` | Video | 20 | 6 | 1.0000 | 1.0000 | 1.0000 | 0.0000 | Tested on 3 synthetic real / 3 synthetic fake npy clips |
| `AUD_CLAD_SMOKE_001` | Audio | 35 | 10 | 1.0000 | 1.0000 | 1.0000 | 0.0000 | Tested on 5 synthetic real / 5 synthetic fake wav files |

**Discrepancy with Academic Documentation**:
* `reports/model_selection.md` claimed "In-Domain (FF++)" 1.0000, "Celeb-DF" 0.9583, "DFDC" 1.0000, and "ASVspoof" 1.0000.
* These benchmark results were never actually executed on FaceForensics++, Celeb-DF, DFDC, or ASVspoof. They were purely placeholder projections based on the 12-sample synthetic smoke tests.

---

## 5. Identified Bugs and Architectural Flaws

### Bug 1: Video Model Logit Discarded During Inference (Critical Severity)
* **File**: `ml/video/inference.py`, Lines 45–68.
* **Flaw**: The engine executes `v_logits, f_logits = self.model(clip_tensor)` and computes `model_video_score = torch.sigmoid(v_logits).item()`. However, `model_video_score` is never referenced again.
* **Impact**: The final video score is computed exclusively from spatial frame image inferences and heuristic optical flow anomalies:
  $$\text{final\_video\_score} = 0.55 \cdot \text{mean\_spatial} + 0.25 \cdot \text{max\_spatial} + 0.20 \cdot \text{anomaly\_score}$$
  The video temporal model is completely bypassed.

### Bug 2: Training vs. Inference Preprocessing Mismatch in Image Branch (High Severity)
* **File**: `ml/image/train.py` vs `ml/image/inference.py`.
* **Flaw**: In `train.py`, images are opened via `Image.open()` and resized directly to $224 \times 224$ using `DEFAULT_TRANSFORM`. In `inference.py`, OpenCV Haar Cascade face detection (`extract_face_crop`) is executed to crop the bounding box before resizing.
* **Impact**: During inference, the model receives tight face crops with different scale, aspect ratio, and boundary context compared to whatever was fed during training.

### Bug 3: Handcrafted Heuristic Interference in Neural Decision Heads (High Severity)
* **File**: `ml/image/inference.py` (Line 69) and `ml/audio/inference.py` (Line 47).
* **Flaw**: Image inference overrides the neural network with `0.8 * neural_score + 0.2 * fft_anomaly`. Audio inference overrides with `0.50 * model_score + 0.50 * anomaly_score`.
* **Impact**: These handcrafted formulas (`fft_anomaly` and `audio_anomaly_score`) were tuned on synthetic mathematical artifacts. On real-world natural photos (e.g., smooth lighting, portrait skin softening, or high-contrast textures) or natural speech (whispers, loud speech), they generate immediate false positives regardless of what the neural network predicts.

### Bug 4: StyleLatentEncoder Unpretrained Random Weight Initialization (Medium Severity)
* **File**: `ml/video/psp_encoder.py`.
* **Flaw**: The model claims to implement pixel2Style2pixel (pSp) style latent extraction into StyleGAN's $W+$ latent space. In reality, it is a 2-layer convolutional network initialized randomly.
* **Impact**: Without pretraining or GAN inversion supervision, the extracted "style" vector is random noise that provides zero true StyleGAN semantic latent trajectory information.

### Bug 5: Backend Checkpoint Loading Defaults (Medium Severity)
* **File**: `backend/api/router.py`, Lines 34–42.
* **Flaw**: Backend hardcodes paths to `experiments/*_SMOKE_001/best_checkpoint.pth`. If those paths are missing or invalid, it silently falls back to randomly initialized or backbone-only weights without failing fast or notifying the user.

---

## 6. Likely Causes of Poor Accuracy in Practice

1. **Lack of Real Training Data**: The existing checkpoints were trained on 30–50 samples of mathematical sine waves. When tested on genuine human faces or real speech, the models had never seen human skin textures, eye blinks, facial expressions, or natural vocal acoustics.
2. **Heuristic Noise Misclassification**: The handcrafted Fourier and audio kurtosis filters mistake natural camera bokeh or compression artifacts for deepfakes.
3. **No Cross-Dataset Optimization**: Models were not evaluated across different manipulation techniques (FaceSwap vs DeepFaceLab vs Diffusion vs TTS).

---

## 7. Fixes Required

1. **Acquire and Prepare Real Benchmarking Datasets**:
   * Image: Celeb-DF real/fake face crops and Deepfake Face Swap benchmark.
   * Video: Real video clips with frame-level and video-level labels, strictly respecting source video isolation across splits to prevent frame leakage.
   * Audio: Real speech vs AI synthetic/cloned speech datasets (e.g. WaveFake / Deepfake Audio Dataset / In-the-Wild).
2. **Harmonize Preprocessing Pipelines**:
   * Standardize face detection and cropping (`extract_face_crop`) so that all training, validation, and inference code paths receive identical aligned face crops.
3. **Correct Inference Integration**:
   * Connect `model_video_score` directly to video predictions.
   * Train clean neural decision heads with calibrated thresholds, deprecating ad-hoc heuristic blending that distorts calibrated probabilities.
4. **Implement Temporal & Video-Level Aggregation**:
   * Implement validated aggregation (mean pooling, max-margin pooling, and learned temporal aggregation) on held-out validation sets.
5. **Establish Apple Silicon M4 Training Protocol**:
   * Leverage PyTorch `mps` accelerator with CPU fallback.
   * Implement batch sizes and gradient accumulation suitable for 16 GB unified RAM and 30 GB available disk space.

---

## 8. Recommended Multi-Dataset Experiments

* **Experiment A**: Baseline reproduction on clean held-out real data.
* **Experiment B**: Preprocessing harmonization (identical face cropping and spectrogram normalization across train and inference).
* **Experiment C**: Training FA-ViT on real face deepfake datasets (Celeb-DF & FaceSwap).
* **Experiment D**: Video temporal modeling with integrated Style Latent Flow / Temporal 3D CNN on real video clips with zero split leakage.
* **Experiment E**: CLAD audio model training on real vs synthetic speech with MoCo contrastive learning and length loss.
* **Experiment F**: Cross-dataset generalization evaluation (train on Dataset 1, evaluate on Dataset 2).
* **Experiment G**: Robustness evaluation against realistic perturbations (JPEG compression, Gaussian noise, audio volume shifts).

---

## 9. Computational Resources & Feasibility

* **Hardware**: Apple Silicon M4, 16.0 GB Unified Memory.
* **Available Disk Space**: ~30.8 GB free.
* **Device Acceleration**: PyTorch MPS (`mps`) verified active and functional.
* **Feasibility**: High-efficiency fine-tuning with batch sizes of 16–32, image resolutions of $224 \times 224$, and audio clips of 4.0 seconds (64k samples) easily fits within 16 GB unified memory, with training epochs completing in minutes per run.
