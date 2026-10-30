"""
Cross-Dataset Generalization & Independent Benchmark Evaluation Suite
Evaluates trained production checkpoints on newly acquired independent datasets:
1. Video: DFDC Video Benchmark (20 independent clips)
2. Audio: ASVspoof 2019 Logical Access Benchmark (100 independent audio clips)
3. Image: Zenodo Real vs Fake Faces (1,000 independent StyleGAN/FFHQ face crops)
Records: Accuracy, Precision, Recall, F1, ROC-AUC, EER, Confusion Matrix.
Saves to experiments/cross_dataset_evaluations.json.
"""

import os
import json
import time
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image

from ml.common.device import get_device
from ml.common.metrics import compute_all_metrics
from ml.image.baselines import EfficientNetBaseline, ResNet50Baseline
from ml.image.favit import FAViTModel
from ml.image.preprocessing import DEFAULT_TRANSFORM
from ml.video.model import FrameBiGRUTemporalModel, Content3DResNetClassifier
from ml.audio.clad import CLADModel
from ml.audio.baselines import AASISTBaselineClassifier
from ml.audio.preprocessing import load_and_preprocess_audio, compute_log_mel_spectrogram

def evaluate_video_cross_dataset(device):
    print("\n=======================================================")
    print("[*] Evaluating Video Models on Independent DFDC Benchmark")
    print("=======================================================")
    manifest_path = "datasets/manifests/dfdc_video_benchmark_manifest.json"
    if not os.path.exists(manifest_path):
        print(f"[!] Manifest not found: {manifest_path}")
        return {}
        
    with open(manifest_path, "r") as f:
        data = json.load(f)
        
    clips_data = data["clips"]
    print(f"[+] Loaded {len(clips_data)} DFDC clips")
    
    # 1. Evaluate Old Frame-BiGRU
    model_path = "checkpoints/video/frame_cnn___bigru_temporal_aggregator_best.pth"
    model = FrameBiGRUTemporalModel(embed_dim=128).to(device)
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"[+] Loaded model from {model_path}")
    model.eval()
    
    # 2. Evaluate New EfficientNet-BiGRU Combined Model
    from ml.video.model import EfficientNetBiGRUVideoModel
    eff_video_path = "checkpoints/video/efficientnet_bigru_combined_best.pth"
    eff_video_model = EfficientNetBiGRUVideoModel(embed_dim=128).to(device)
    if os.path.exists(eff_video_path):
        eff_video_model.load_state_dict(torch.load(eff_video_path, map_location=device))
        print(f"[+] Loaded EfficientNet-BiGRU from {eff_video_path}")
    eff_video_model.eval()
    
    targets, preds, eff_preds = [], [], []
    latencies, eff_lats = [], []
    
    with torch.no_grad():
        for item in clips_data:
            clip_path = item["filepath"]
            label = item["label"]
            arr = np.load(clip_path) # [5, 3, 224, 224]
            t = torch.tensor(arr, dtype=torch.float32).unsqueeze(0).to(device)
            
            # Old Frame-BiGRU
            t0 = time.time()
            v_logits, _ = model(t)
            lat = (time.time() - t0) * 1000
            prob = torch.sigmoid(v_logits).item()
            targets.append(label)
            preds.append(prob)
            latencies.append(lat)
            
            # New EfficientNet-BiGRU
            t0 = time.time()
            v_logits_eff, _ = eff_video_model(t)
            eff_lats.append((time.time() - t0) * 1000)
            eff_preds.append(torch.sigmoid(v_logits_eff).item())
            
    metrics = compute_all_metrics(np.array(targets), np.array(preds))
    metrics["mean_latency_ms"] = round(float(np.mean(latencies)), 2)
    
    metrics_eff = compute_all_metrics(np.array(targets), np.array(eff_preds))
    metrics_eff["mean_latency_ms"] = round(float(np.mean(eff_lats)), 2)
    
    print(f"[✓] DFDC Frame-BiGRU (Old)         | Acc: {metrics['accuracy']:.4f} | AUC: {metrics['roc_auc']:.4f} | Recall: {metrics['recall']:.4f}")
    print(f"[✓] DFDC EfficientNet-BiGRU (New) | Acc: {metrics_eff['accuracy']:.4f} | AUC: {metrics_eff['roc_auc']:.4f} | Recall: {metrics_eff['recall']:.4f}")
    return {
        "dfdc_frame_bigru_old": metrics,
        "dfdc_efficientnet_bigru_production": metrics_eff
    }

def evaluate_audio_cross_dataset(device):
    print("\n=======================================================")
    print("[*] Evaluating Audio Models on Independent ASVspoof 2019 LA Benchmark")
    print("=======================================================")
    manifest_path = "datasets/manifests/asvspoof_2019_la_manifest.json"
    if not os.path.exists(manifest_path):
        print(f"[!] Manifest not found: {manifest_path}")
        return {}
        
    with open(manifest_path, "r") as f:
        data = json.load(f)
        
    samples = data["samples"]
    print(f"[+] Loaded {len(samples)} ASVspoof 2019 LA samples")
    
    # Evaluate CLAD
    clad_path = "checkpoints/audio/clad_full_model_aasist___contrastive___lengthloss_best.pth"
    clad_model = CLADModel(in_channels=1, embed_dim=128).to(device)
    if os.path.exists(clad_path):
        clad_model.load_state_dict(torch.load(clad_path, map_location=device))
        print(f"[+] Loaded CLAD model from {clad_path}")
    clad_model.eval()
    
    # Evaluate AASIST Baseline
    aasist_path = "checkpoints/audio/aasist_baseline_classifier_best.pth"
    aasist_model = AASISTBaselineClassifier(embed_dim=128).to(device)
    if os.path.exists(aasist_path):
        aasist_model.load_state_dict(torch.load(aasist_path, map_location=device))
        print(f"[+] Loaded AASIST model from {aasist_path}")
    aasist_model.eval()

    # Evaluate Combined AASIST Model (Trained across benchmarks)
    comb_path = "checkpoints/audio/aasist_combined_multidataset_best.pth"
    comb_model = AASISTBaselineClassifier(embed_dim=128).to(device)
    if os.path.exists(comb_path):
        comb_model.load_state_dict(torch.load(comb_path, map_location=device))
        print(f"[+] Loaded Combined AASIST model from {comb_path}")
    comb_model.eval()
    
    targets = []
    clad_preds, clad_lats = [], []
    aasist_preds, aasist_lats = [], []
    comb_preds, comb_lats = [], []
    
    with torch.no_grad():
        for item in samples:
            wav_path = item["filepath"]
            label = item["label"]
            y, sr = load_and_preprocess_audio(wav_path)
            mel = compute_log_mel_spectrogram(y, sr=sr).unsqueeze(0).to(device)
            
            # CLAD
            t0 = time.time()
            log_c, _ = clad_model(mel)
            clad_lats.append((time.time() - t0) * 1000)
            clad_preds.append(torch.sigmoid(log_c).item())
            
            # AASIST Baseline
            t0 = time.time()
            log_a, _ = aasist_model(mel)
            aasist_lats.append((time.time() - t0) * 1000)
            aasist_preds.append(torch.sigmoid(log_a).item())

            # Combined AASIST (Production)
            t0 = time.time()
            log_comb, _ = comb_model(mel)
            comb_lats.append((time.time() - t0) * 1000)
            comb_preds.append(torch.sigmoid(log_comb).item())
            
            targets.append(label)
            
    m_clad = compute_all_metrics(np.array(targets), np.array(clad_preds))
    m_clad["mean_latency_ms"] = round(float(np.mean(clad_lats)), 2)
    
    m_aasist = compute_all_metrics(np.array(targets), np.array(aasist_preds))
    m_aasist["mean_latency_ms"] = round(float(np.mean(aasist_lats)), 2)

    m_comb = compute_all_metrics(np.array(targets), np.array(comb_preds))
    m_comb["mean_latency_ms"] = round(float(np.mean(comb_lats)), 2)
    
    print(f"[✓] ASVspoof 2019 CLAD (Old)       | Acc: {m_clad['accuracy']:.4f} | AUC: {m_clad['roc_auc']:.4f} | Recall: {m_clad['recall']:.4f}")
    print(f"[✓] ASVspoof 2019 AASIST (Base)    | Acc: {m_aasist['accuracy']:.4f} | AUC: {m_aasist['roc_auc']:.4f} | Recall: {m_aasist['recall']:.4f}")
    print(f"[✓] ASVspoof 2019 AASIST (Comb)    | Acc: {m_comb['accuracy']:.4f} | AUC: {m_comb['roc_auc']:.4f} | Recall: {m_comb['recall']:.4f}")
    
    return {
        "asvspoof_2019_clad": m_clad,
        "asvspoof_2019_aasist": m_aasist,
        "asvspoof_2019_aasist_combined_production": m_comb
    }

def evaluate_image_cross_dataset(device):
    print("\n=======================================================")
    print("[*] Evaluating Image Models on Independent FFHQ vs StyleGAN Benchmark")
    print("=======================================================")
    manifest_path = "datasets/manifests/ffhq_stylegan_manifest.json"
    if not os.path.exists(manifest_path):
        print(f"[!] Manifest not found: {manifest_path}")
        return {}
        
    with open(manifest_path, "r") as f:
        data = json.load(f)
        
    samples = data["samples"]
    print(f"[+] Loaded {len(samples)} FFHQ vs StyleGAN face crops")
    
    # 1. FAViT Model
    favit_path = "checkpoints/image/fa_vit_gam_lam_fft_fal_best.pth"
    favit_model = FAViTModel(use_gam=True, use_lam=True, use_fft=True, use_spatial_noise=True, pretrained=False).to(device)
    if os.path.exists(favit_path):
        favit_model.load_state_dict(torch.load(favit_path, map_location=device))
        print(f"[+] Loaded FAViT from {favit_path}")
    favit_model.eval()
    
    # 2. EfficientNet-B0 (Old Celeb-DF Only)
    eff_path = "checkpoints/image/efficientnet_b0_best.pth"
    eff_model = EfficientNetBaseline(pretrained=False).to(device)
    if os.path.exists(eff_path):
        eff_model.load_state_dict(torch.load(eff_path, map_location=device))
        print(f"[+] Loaded EfficientNet-B0 from {eff_path}")
    eff_model.eval()

    # 3. EfficientNet-B0 Combined Multi-Dataset (Production)
    eff_comb_path = "checkpoints/image/efficientnet_b0_combined_multidataset_best.pth"
    eff_comb_model = EfficientNetBaseline(pretrained=False).to(device)
    if os.path.exists(eff_comb_path):
        eff_comb_model.load_state_dict(torch.load(eff_comb_path, map_location=device))
        print(f"[+] Loaded Combined Multi-Dataset EfficientNet from {eff_comb_path}")
    eff_comb_model.eval()

    # 4. ResNet-50
    res_path = "checkpoints/image/resnet_50_best.pth"
    res_model = ResNet50Baseline(pretrained=False).to(device)
    if os.path.exists(res_path):
        res_model.load_state_dict(torch.load(res_path, map_location=device))
        print(f"[+] Loaded ResNet-50 from {res_path}")
    res_model.eval()
    
    targets = []
    favit_preds, favit_lats = [], []
    eff_preds, eff_lats = [], []
    eff_comb_preds, eff_comb_lats = [], []
    res_preds, res_lats = [], []
    
    with torch.no_grad():
        for item in samples:
            img_path = item["filepath"]
            label = item["label"]
            targets.append(label)
            
            try:
                pil_img = Image.open(img_path).convert("RGB")
                tensor_img = DEFAULT_TRANSFORM(pil_img).unsqueeze(0).to(device)
            except Exception as e:
                tensor_img = torch.zeros((1, 3, 224, 224), device=device)
                
            # FAViT
            t0 = time.time()
            log_f, _ = favit_model(tensor_img)
            favit_lats.append((time.time() - t0) * 1000)
            favit_preds.append(torch.sigmoid(log_f).item())
            
            # EfficientNet-B0 (Old)
            t0 = time.time()
            log_e, _ = eff_model(tensor_img)
            eff_lats.append((time.time() - t0) * 1000)
            eff_preds.append(torch.sigmoid(log_e).item())

            # EfficientNet-B0 Combined (Production)
            t0 = time.time()
            log_ec, _ = eff_comb_model(tensor_img)
            eff_comb_lats.append((time.time() - t0) * 1000)
            eff_comb_preds.append(torch.sigmoid(log_ec).item())
            
            # ResNet-50
            t0 = time.time()
            log_r, _ = res_model(tensor_img)
            res_lats.append((time.time() - t0) * 1000)
            res_preds.append(torch.sigmoid(log_r).item())
            
    m_favit = compute_all_metrics(np.array(targets), np.array(favit_preds))
    m_favit["mean_latency_ms"] = round(float(np.mean(favit_lats)), 2)
    
    m_eff = compute_all_metrics(np.array(targets), np.array(eff_preds))
    m_eff["mean_latency_ms"] = round(float(np.mean(eff_lats)), 2)

    m_eff_comb = compute_all_metrics(np.array(targets), np.array(eff_comb_preds))
    m_eff_comb["mean_latency_ms"] = round(float(np.mean(eff_comb_lats)), 2)
    
    m_res = compute_all_metrics(np.array(targets), np.array(res_preds))
    m_res["mean_latency_ms"] = round(float(np.mean(res_lats)), 2)
    
    print(f"[✓] StyleGAN FAViT               | Acc: {m_favit['accuracy']:.4f} | AUC: {m_favit['roc_auc']:.4f} | Recall: {m_favit['recall']:.4f}")
    print(f"[✓] StyleGAN EfficientNet (Old)  | Acc: {m_eff['accuracy']:.4f} | AUC: {m_eff['roc_auc']:.4f} | Recall: {m_eff['recall']:.4f}")
    print(f"[✓] StyleGAN EfficientNet (Comb) | Acc: {m_eff_comb['accuracy']:.4f} | AUC: {m_eff_comb['roc_auc']:.4f} | Recall: {m_eff_comb['recall']:.4f}")
    print(f"[✓] StyleGAN ResNet-50          | Acc: {m_res['accuracy']:.4f} | AUC: {m_res['roc_auc']:.4f} | Recall: {m_res['recall']:.4f}")
    
    return {
        "ffhq_stylegan_favit": m_favit,
        "ffhq_stylegan_efficientnet_old": m_eff,
        "ffhq_stylegan_efficientnet_combined_production": m_eff_comb,
        "ffhq_stylegan_resnet50": m_res
    }

def main():
    device = get_device()
    print(f"[*] Starting Cross-Dataset Generalization Evaluation on {device}")
    
    res_image = evaluate_image_cross_dataset(device)
    res_video = evaluate_video_cross_dataset(device)
    res_audio = evaluate_audio_cross_dataset(device)
    
    combined = {
        "evaluation_timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "hardware_device": str(device),
        "image_cross_dataset": res_image,
        "video_cross_dataset": res_video,
        "audio_cross_dataset": res_audio
    }
    
    os.makedirs("experiments", exist_ok=True)
    out_file = "experiments/cross_dataset_evaluations.json"
    with open(out_file, "w") as f:
        json.dump(combined, f, indent=2)
    print(f"\n[✓] All Cross-Dataset evaluations saved to {out_file}")

if __name__ == "__main__":
    main()

