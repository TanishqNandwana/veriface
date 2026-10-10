# VeriFace Phase 2 — Empirical Baseline Evaluation Report

**Evaluation Date**: 2026-10-09 16:01:00  
**Hardware Platform**: Apple Silicon M4 GPU (`mps` backend)  
**Protocol**: Rigorous evaluation of existing initial model implementations on genuine held-out real deepfake benchmarks.

---

## 1. Summary of Baseline Results

| Modality | Model Architecture | Test Set | Samples (Real / Fake) | Accuracy | Precision | Recall | F1-Score | ROC-AUC | EER | Mean Latency |
|---|---|---|---|---|---|---|---|---|---|---|
| **Image** | FA-ViT (Pretrained ViT + Heuristic FFT) | Celeb-DF v2 Held-out | 71 (40 / 31) | **0.4930** | **0.4390** | **0.5806** | **0.5000** | **0.5371** | **0.4919** | 35.68 ms |
| **Video** | Style Latent Flow / Spatial Heuristic | Celeb-DF Video Clips | 12 (6 / 6) | **0.5000** | **0.5000** | **0.6667** | **0.5714** | **0.5000** | **0.5833** | 319.98 ms |
| **Audio** | CLAD (AASIST + Heuristic Ensemble) | Hemg Speech vs TTS | 20 (10 / 10) | **0.4500** | **0.4545** | **0.5000** | **0.4762** | **0.3900** | **0.6000** | 79.13 ms |

---

## 2. Modality Detailed Breakdown

### 2.1 Image Modality
* **Confusion Matrix** (Threshold = 0.50):
  * True Negatives (Real detected as Real): `17`
  * False Positives (Real detected as Fake): `23`
  * False Negatives (Fake detected as Real): `13`
  * True Positives (Fake detected as Fake): `18`
* **False Acceptance Rate (FAR)**: `0.5000`
* **False Rejection Rate (FRR)**: `0.4839`
* **Optimal Decision Threshold (by EER)**: `0.5505`

### 2.2 Video Modality
* **Confusion Matrix** (Threshold = 0.50):
  * True Negatives: `2`
  * False Positives: `4`
  * False Negatives: `2`
  * True Positives: `4`
* **False Acceptance Rate (FAR)**: `0.6667`
* **False Rejection Rate (FRR)**: `0.5000`
* **Optimal Decision Threshold**: `0.5710`

### 2.3 Audio Modality
* **Confusion Matrix** (Threshold = 0.50):
  * True Negatives: `4`
  * False Positives: `6`
  * False Negatives: `5`
  * True Positives: `5`
* **Equal Error Rate (EER)**: `0.6000`
* **FAR**: `0.6000` | **FRR**: `0.6000`
* **Optimal Decision Threshold**: `0.5296`

---

## 3. Analysis of Baseline Deficiencies
1. **Severe Classification Failures on Real Faces**: The original image model achieved poor recall or false positive rates because the classification head was trained on synthetic sinusoidal noise rather than real face artifacts.
2. **Video Model Decoupling**: The video pipeline completely failed to utilize temporal motion features due to Bug 1 (temporal logits discarded).
3. **Audio Heuristic Bias**: Audio detection without proper neural pretraining suffered significant degradation on varied vocal acoustics.
