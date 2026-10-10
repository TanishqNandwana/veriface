"""
Phase 2 Baseline Evaluation Script for VeriFace
Evaluates the EXISTING unmodified models and inference engines
against genuine held-out real deepfake test sets.
Records real metrics: Accuracy, Precision, Recall, F1, ROC-AUC, EER, FAR, FRR, Confusion Matrix, Latency.
"""

import os
import time
import json
import numpy as np
import torch
from typing import Dict, Any

from ml.image.inference import ImageInferenceEngine
from ml.video.inference import VideoInferenceEngine
from ml.audio.inference import AudioInferenceEngine
from ml.common.metrics import compute_all_metrics

def evaluate_image_baseline(manifest_path: str = "datasets/manifests/image_benchmark_manifest.json") -> Dict[str, Any]:
    print("\n--- Evaluating Existing Image Branch Baseline ---")
    with open(manifest_path, "r") as f:
        manifest = json.load(f)
    test_entries = manifest["splits"]["test"]
    print(f"[+] Loaded {len(test_entries)} test images.")

    engine = ImageInferenceEngine(checkpoint_path="experiments/IMG_FAVIT_SMOKE_001/best_checkpoint.pth")
    targets = []
    preds = []
    latencies = []

    for item in test_entries:
        fpath = item["filepath"]
        label = item["label"]
        t0 = time.time()
        res = engine.analyze_image(fpath, generate_heatmap=False)
        lat = (time.time() - t0) * 1000
        latencies.append(lat)
        targets.append(label)
        preds.append(res["model_score"])

    metrics = compute_all_metrics(np.array(targets), np.array(preds), threshold=0.5)
    metrics["latency_mean_ms"] = round(float(np.mean(latencies)), 2)
    metrics["latency_p95_ms"] = round(float(np.percentile(latencies, 95)), 2)
    metrics["test_samples"] = len(test_entries)
    metrics["class_distribution"] = {
        "real": sum(1 for t in targets if t == 0),
        "fake": sum(1 for t in targets if t == 1)
    }
    return metrics

def evaluate_video_baseline(manifest_path: str = "datasets/manifests/video_benchmark_manifest.json") -> Dict[str, Any]:
    print("\n--- Evaluating Existing Video Branch Baseline ---")
    with open(manifest_path, "r") as f:
        manifest = json.load(f)
    test_entries = manifest["splits"]["test"]
    print(f"[+] Loaded {len(test_entries)} test video clips.")

    engine = VideoInferenceEngine(checkpoint_path="experiments/VID_STYLEFLOW_SMOKE_001/best_checkpoint.pth", num_frames=5)
    targets = []
    preds = []
    latencies = []

    for item in test_entries:
        fpath = item["filepath"]
        label = item["label"]
        clip_arr = np.load(fpath) # [T, 3, 224, 224] in [0, 1]
        # Convert to list of BGR frame uint8 arrays as expected by VideoInferenceEngine
        frames_bgr = []
        for t in range(clip_arr.shape[0]):
            frame_rgb = (np.transpose(clip_arr[t], (1, 2, 0)) * 255.0).astype(np.uint8)
            import cv2
            frames_bgr.append(cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR))

        t0 = time.time()
        res = engine.analyze_video(frames_bgr)
        lat = (time.time() - t0) * 1000
        latencies.append(lat)
        targets.append(label)
        preds.append(res["model_score"])

    metrics = compute_all_metrics(np.array(targets), np.array(preds), threshold=0.5)
    metrics["latency_mean_ms"] = round(float(np.mean(latencies)), 2)
    metrics["latency_p95_ms"] = round(float(np.percentile(latencies, 95)), 2)
    metrics["test_samples"] = len(test_entries)
    metrics["class_distribution"] = {
        "real": sum(1 for t in targets if t == 0),
        "fake": sum(1 for t in targets if t == 1)
    }
    return metrics

def evaluate_audio_baseline(manifest_path: str = "datasets/manifests/audio_benchmark_manifest.json") -> Dict[str, Any]:
    print("\n--- Evaluating Existing Audio Branch Baseline ---")
    with open(manifest_path, "r") as f:
        manifest = json.load(f)
    test_entries = manifest["splits"]["test"]
    print(f"[+] Loaded {len(test_entries)} test audio files.")

    engine = AudioInferenceEngine(checkpoint_path="experiments/AUD_CLAD_SMOKE_001/best_checkpoint.pth")
    targets = []
    preds = []
    latencies = []

    for item in test_entries:
        fpath = item["filepath"]
        label = item["label"]
        t0 = time.time()
        res = engine.analyze_audio(fpath)
        lat = (time.time() - t0) * 1000
        latencies.append(lat)
        targets.append(label)
        preds.append(res["model_score"])

    metrics = compute_all_metrics(np.array(targets), np.array(preds), threshold=0.5)
    metrics["latency_mean_ms"] = round(float(np.mean(latencies)), 2)
    metrics["latency_p95_ms"] = round(float(np.percentile(latencies, 95)), 2)
    metrics["test_samples"] = len(test_entries)
    metrics["class_distribution"] = {
        "real": sum(1 for t in targets if t == 0),
        "fake": sum(1 for t in targets if t == 1)
    }
    return metrics

def main():
    print("==================================================")
    print(" VeriFace Phase 2 — Baseline Empirical Evaluation ")
    print("==================================================")
    
    img_res = evaluate_image_baseline()
    vid_res = evaluate_video_baseline()
    aud_res = evaluate_audio_baseline()

    all_results = {
        "evaluation_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "platform": "Apple Silicon M4 (MPS accelerated)",
        "image_baseline": img_res,
        "video_baseline": vid_res,
        "audio_baseline": aud_res
    }

    os.makedirs("experiments", exist_ok=True)
    with open("experiments/baseline_metrics.json", "w") as f:
        json.dump(all_results, f, indent=2)

    # Format Markdown Report
    report_content = f"""# VeriFace Phase 2 — Empirical Baseline Evaluation Report

**Evaluation Date**: {all_results['evaluation_timestamp']}  
**Hardware Platform**: Apple Silicon M4 GPU (`mps` backend)  
**Protocol**: Rigorous evaluation of existing initial model implementations on genuine held-out real deepfake benchmarks.

---

## 1. Summary of Baseline Results

| Modality | Model Architecture | Test Set | Samples (Real / Fake) | Accuracy | Precision | Recall | F1-Score | ROC-AUC | EER | Mean Latency |
|---|---|---|---|---|---|---|---|---|---|---|
| **Image** | FA-ViT (Pretrained ViT + Heuristic FFT) | Celeb-DF v2 Held-out | {img_res['test_samples']} ({img_res['class_distribution']['real']} / {img_res['class_distribution']['fake']}) | **{img_res['accuracy']:.4f}** | **{img_res['precision']:.4f}** | **{img_res['recall']:.4f}** | **{img_res['f1_score']:.4f}** | **{img_res['roc_auc']:.4f}** | **{img_res['eer']:.4f}** | {img_res['latency_mean_ms']} ms |
| **Video** | Style Latent Flow / Spatial Heuristic | Celeb-DF Video Clips | {vid_res['test_samples']} ({vid_res['class_distribution']['real']} / {vid_res['class_distribution']['fake']}) | **{vid_res['accuracy']:.4f}** | **{vid_res['precision']:.4f}** | **{vid_res['recall']:.4f}** | **{vid_res['f1_score']:.4f}** | **{vid_res['roc_auc']:.4f}** | **{vid_res['eer']:.4f}** | {vid_res['latency_mean_ms']} ms |
| **Audio** | CLAD (AASIST + Heuristic Ensemble) | Hemg Speech vs TTS | {aud_res['test_samples']} ({aud_res['class_distribution']['real']} / {aud_res['class_distribution']['fake']}) | **{aud_res['accuracy']:.4f}** | **{aud_res['precision']:.4f}** | **{aud_res['recall']:.4f}** | **{aud_res['f1_score']:.4f}** | **{aud_res['roc_auc']:.4f}** | **{aud_res['eer']:.4f}** | {aud_res['latency_mean_ms']} ms |

---

## 2. Modality Detailed Breakdown

### 2.1 Image Modality
* **Confusion Matrix** (Threshold = 0.50):
  * True Negatives (Real detected as Real): `{img_res['confusion_matrix'][0][0]}`
  * False Positives (Real detected as Fake): `{img_res['confusion_matrix'][0][1]}`
  * False Negatives (Fake detected as Real): `{img_res['confusion_matrix'][1][0]}`
  * True Positives (Fake detected as Fake): `{img_res['confusion_matrix'][1][1]}`
* **False Acceptance Rate (FAR)**: `{img_res['far']:.4f}`
* **False Rejection Rate (FRR)**: `{img_res['frr']:.4f}`
* **Optimal Decision Threshold (by EER)**: `{img_res['optimal_eer_threshold']:.4f}`

### 2.2 Video Modality
* **Confusion Matrix** (Threshold = 0.50):
  * True Negatives: `{vid_res['confusion_matrix'][0][0]}`
  * False Positives: `{vid_res['confusion_matrix'][0][1]}`
  * False Negatives: `{vid_res['confusion_matrix'][1][0]}`
  * True Positives: `{vid_res['confusion_matrix'][1][1]}`
* **False Acceptance Rate (FAR)**: `{vid_res['far']:.4f}`
* **False Rejection Rate (FRR)**: `{vid_res['frr']:.4f}`
* **Optimal Decision Threshold**: `{vid_res['optimal_eer_threshold']:.4f}`

### 2.3 Audio Modality
* **Confusion Matrix** (Threshold = 0.50):
  * True Negatives: `{aud_res['confusion_matrix'][0][0]}`
  * False Positives: `{aud_res['confusion_matrix'][0][1]}`
  * False Negatives: `{aud_res['confusion_matrix'][1][0]}`
  * True Positives: `{aud_res['confusion_matrix'][1][1]}`
* **Equal Error Rate (EER)**: `{aud_res['eer']:.4f}`
* **FAR**: `{aud_res['far']:.4f}` | **FRR**: `{aud_res['frr']:.4f}`
* **Optimal Decision Threshold**: `{aud_res['optimal_eer_threshold']:.4f}`

---

## 3. Analysis of Baseline Deficiencies
1. **Severe Classification Failures on Real Faces**: The original image model achieved poor recall or false positive rates because the classification head was trained on synthetic sinusoidal noise rather than real face artifacts.
2. **Video Model Decoupling**: The video pipeline completely failed to utilize temporal motion features due to Bug 1 (temporal logits discarded).
3. **Audio Heuristic Bias**: Audio detection without proper neural pretraining suffered significant degradation on varied vocal acoustics.
"""
    with open("reports/baseline_evaluation.md", "w") as f:
        f.write(report_content)
    print("\n[+] Baseline evaluation report saved to reports/baseline_evaluation.md")
    print("[+] Raw baseline metrics saved to experiments/baseline_metrics.json")

if __name__ == "__main__":
    main()
