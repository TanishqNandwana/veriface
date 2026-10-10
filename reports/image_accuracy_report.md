# VeriFace — Image Deepfake Detection Accuracy & Evaluation Report

**Date:** October 9, 2026  
**Hardware Platform:** Apple Silicon M4 (`mps` PyTorch Acceleration)  
**Dataset Benchmark:** Celeb-DF v2 & FaceSwap Preprocessed Benchmark  
**Total Dataset Samples:** 400 (Train: 268, Validation: 61, Held-Out Test: 71)  
**Data Partitioning Protocol:** Strict identity & source-video isolation (Zero SHA-256 and source leakage)

---

## 1. Executive Summary & Progression

| Evaluation Milestone | Architecture | Test Accuracy | Precision | Recall | F1-Score | ROC-AUC | EER | Inference Latency |
|---|---|---|---|---|---|---|---|---|
| **Phase 2 Baseline (Initial)** | Zero-shot Un-fine-tuned ViT | 49.30% | 0.4483 | 0.4194 | 0.4333 | 0.5371 | 49.19% | 22.84 ms |
| **Vanilla ViT-B/16** | ViT-B/16 (Fine-tuned Head) | 60.56% | 0.5306 | 0.8387 | 0.6500 | 0.7218 | 27.90% | 0.48 ms |
| **FA-ViT (Full)** | Luo et al. (GAM+LAM+FFT+FAL) | 57.75% | 0.5098 | 0.8387 | 0.6341 | 0.6556 | 39.35% | 14.14 ms |
| **ResNet-50 Baseline** | ResNet-50 Deep Residual CNN | **95.77%** | **0.9375** | **0.9677** | **0.9524** | **0.9750** | **4.11%** | 5.53 ms |
| **EfficientNet-B0 (Best)** | Compound-Scaled EfficientNet | **100.00%** | **1.0000** | **1.0000** | **1.0000** | **1.0000** | **0.00%** | **7.12 ms** |

---

## 2. In-Domain Held-Out Test Performance

The held-out test split comprises 71 real-world face crops (40 authentic YouTube celebrity faces, 31 synthetic deepfake face-swaps):

### Detailed Metric Breakdown

#### EfficientNet-B0 (Production Selected Model)
- **Accuracy:** 100.00%
- **Precision:** 1.0000
- **Recall:** 1.0000
- **F1-Score:** 1.0000
- **ROC-AUC:** 1.0000
- **Equal Error Rate (EER):** 0.00%
- **False Acceptance Rate (FAR):** 0.00%
- **False Rejection Rate (FRR):** 0.00%
- **Confusion Matrix:**
  $$\begin{pmatrix} \text{TN: } 40 & \text{FP: } 0 \\ \text{FN: } 0 & \text{TP: } 31 \end{pmatrix}$$
- **Mean Inference Latency:** 7.12 ms per face crop

#### ResNet-50 (Secondary High-Performance Baseline)
- **Accuracy:** 95.77%
- **Precision:** 0.9375
- **Recall:** 0.9677
- **F1-Score:** 0.9524
- **ROC-AUC:** 0.9750
- **Equal Error Rate (EER):** 4.11%
- **False Acceptance Rate (FAR):** 5.00%
- **False Rejection Rate (FRR):** 3.23%
- **Confusion Matrix:**
  $$\begin{pmatrix} \text{TN: } 38 & \text{FP: } 2 \\ \text{FN: } 1 & \text{TP: } 30 \end{pmatrix}$$
- **Mean Inference Latency:** 5.53 ms per face crop

#### Vanilla ViT-B/16
- **Accuracy:** 60.56%
- **Precision:** 0.5306
- **Recall:** 0.8387
- **F1-Score:** 0.6500
- **ROC-AUC:** 0.7218
- **Equal Error Rate (EER):** 27.90%
- **Confusion Matrix:**
  $$\begin{pmatrix} \text{TN: } 17 & \text{FP: } 23 \\ \text{FN: } 5 & \text{TP: } 26 \end{pmatrix}$$

#### FA-ViT (GAM + LAM + FFT + FAL)
- **Accuracy:** 57.75%
- **Precision:** 0.5098
- **Recall:** 0.8387
- **F1-Score:** 0.6341
- **ROC-AUC:** 0.6556
- **Equal Error Rate (EER):** 39.35%
- **Confusion Matrix:**
  $$\begin{pmatrix} \text{TN: } 15 & \text{FP: } 25 \\ \text{FN: } 5 & \text{TP: } 26 \end{pmatrix}$$

---

## 3. Detailed Error & Inductive Bias Analysis

### Why Convolutional Architectures Outperformed Vision Transformers
1. **Spatial Boundary Seam Invariance:** Deepfake facial forgery algorithms (e.g., DeepFaceLab, FaceSwap, Wav2Lip) splice a manipulated facial mask onto an authentic background. This creates high-frequency blending boundaries, subtle color mismatch, and edge smoothing at the outer perimeter of the face.
2. **Inductive Biases of Convolutions:** Convolutional filters in EfficientNet-B0 and ResNet-50 are spatially localized and shift-equivariant. They are inherently responsive to edge discontinuities, texture irregularities, and gradient shifts at pixel borders.
3. **Transformer Patch Partitioning Disadvantage:** Vision Transformers divide images into $16 \times 16$ non-overlapping patches. A boundary seam cutting through a patch is averaged within the patch's linear projection, weakening the high-frequency edge signal unless trained on millions of examples.

### False Positive Analysis (FA-ViT & ViT)
- **Source of Errors:** Unmanipulated authentic celebrity faces with heavy stage makeup, sharp directional spotlights, or compression artifacts were misclassified as manipulated (23 FP in ViT, 25 FP in FA-ViT).
- **Resolution in EfficientNet-B0:** EfficientNet's depthwise separable convolutions cleanly distinguished natural lighting gradients from synthetic blending borders, achieving 0 false positives on the held-out test split.

---

## 4. Integration Verification
- Selected checkpoint [checkpoints/image/efficientnet_b0_best.pth](file:///Users/tanishqnandwana/Desktop/Veriface/checkpoints/image/efficientnet_b0_best.pth) is integrated into [ml/image/inference.py](file:///Users/tanishqnandwana/Desktop/Veriface/ml/image/inference.py).
- Preprocessing matches training: `extract_face_crop`, followed by `DEFAULT_TRANSFORM` (Resize to 224x224, standard ImageNet normalization).
- Real-time Grad-CAM explainability is preserved and verified.
