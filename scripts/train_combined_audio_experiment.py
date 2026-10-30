"""
Phase 11 Experiment D: Combined Multi-Dataset Audio Training (WaveFake + ASVspoof 2019 LA)
Evaluates whether combining multiple speech spoofing datasets improves model generalization.
Trains:
1. Combined AASIST Classifier
2. Combined CLAD Full Model
Evaluates on held-out test splits and saves to experiments/audio/.
"""

import os
import json
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from ml.common.device import get_device
from ml.common.metrics import compute_all_metrics
from ml.audio.preprocessing import load_and_preprocess_audio, compute_log_mel_spectrogram
from ml.audio.clad import CLADModel
from ml.audio.baselines import AASISTBaselineClassifier
from ml.audio.length_loss import LengthLoss

class CombinedAudioDataset(Dataset):
    def __init__(self, items):
        self.items = items

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        item = self.items[idx]
        wav_path = item["filepath"]
        label = item["label"]
        try:
            y, sr = load_and_preprocess_audio(wav_path)
            mel = compute_log_mel_spectrogram(y, sr=sr)
            return mel, torch.tensor(label, dtype=torch.float32)
        except Exception:
            return torch.zeros((1, 80, 250)), torch.tensor(label, dtype=torch.float32)

def train_and_eval(model_name, model, train_items, val_items, test_items, device, epochs=10, lr=2e-4):
    print(f"\n[*] Training {model_name} on Combined Multi-Dataset (Epochs: {epochs})...")
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.BCEWithLogitsLoss()
    
    train_loader = DataLoader(CombinedAudioDataset(train_items), batch_size=8, shuffle=True)
    val_loader = DataLoader(CombinedAudioDataset(val_items), batch_size=8, shuffle=False)
    test_loader = DataLoader(CombinedAudioDataset(test_items), batch_size=8, shuffle=False)
    
    best_val_auc = -1.0
    best_weights = None
    
    for ep in range(1, epochs + 1):
        model.train()
        total_loss, n = 0.0, 0
        for mels, labels in train_loader:
            mels, labels = mels.to(device), labels.to(device)
            optimizer.zero_grad()
            logits, _ = model(mels)
            loss = criterion(logits.squeeze(-1), labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n += 1
            
        # Validation
        model.eval()
        v_preds, v_targets = [], []
        with torch.no_grad():
            for mels, labels in val_loader:
                mels = mels.to(device)
                logits, _ = model(mels)
                probs = torch.sigmoid(logits).squeeze(-1).cpu().numpy().tolist()
                if isinstance(probs, float):
                    probs = [probs]
                v_preds.extend(probs)
                v_targets.extend(labels.numpy().tolist())
                
        val_m = compute_all_metrics(np.array(v_targets), np.array(v_preds))
        if val_m["roc_auc"] >= best_val_auc:
            best_val_auc = val_m["roc_auc"]
            best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            
    # Test evaluation
    model.load_state_dict(best_weights)
    model.eval()
    t_preds, t_targets = [], []
    with torch.no_grad():
        for mels, labels in test_loader:
            mels = mels.to(device)
            logits, _ = model(mels)
            probs = torch.sigmoid(logits).squeeze(-1).cpu().numpy().tolist()
            if isinstance(probs, float):
                probs = [probs]
            t_preds.extend(probs)
            t_targets.extend(labels.numpy().tolist())
            
    test_m = compute_all_metrics(np.array(t_targets), np.array(t_preds))
    print(f"[✓] {model_name} Combined Test Acc: {test_m['accuracy']:.4f} | AUC: {test_m['roc_auc']:.4f} | F1: {test_m['f1_score']:.4f}")
    return test_m, best_weights

def main():
    device = get_device()
    print(f"[*] Starting Combined Multi-Dataset Audio Experiment on {device}")
    
    # 1. Load WaveFake samples
    with open("datasets/manifests/audio_benchmark_manifest.json", "r") as f:
        wf_data = json.load(f)
    wf_train = wf_data["splits"]["train"]
    wf_val = wf_data["splits"]["validation"]
    wf_test = wf_data["splits"]["test"]
    
    # 2. Load ASVspoof 2019 samples
    with open("datasets/manifests/asvspoof_2019_la_manifest.json", "r") as f:
        asv_data = json.load(f)
    asv_samples = asv_data["samples"]
    
    # Split ASVspoof (70 train, 10 val, 20 test)
    asv_train = asv_samples[:70]
    asv_val = asv_samples[70:80]
    asv_test = asv_samples[80:]
    
    # Combine datasets
    comb_train = wf_train + asv_train
    comb_val = wf_val + asv_val
    comb_test = wf_test + asv_test
    print(f"[+] Combined Dataset: Train={len(comb_train)}, Val={len(comb_val)}, Held-Out Test={len(comb_test)}")
    
    # Train Combined AASIST
    m_aasist, w_aasist = train_and_eval(
        "AASIST (Combined Multi-Dataset)",
        AASISTBaselineClassifier(embed_dim=128),
        comb_train, comb_val, comb_test,
        device=device, epochs=10, lr=2e-4
    )
    
    # Train Combined CLAD
    m_clad, w_clad = train_and_eval(
        "CLAD (Combined Multi-Dataset)",
        CLADModel(in_channels=1, embed_dim=128),
        comb_train, comb_val, comb_test,
        device=device, epochs=10, lr=2e-4
    )
    
    results = {
        "dataset_composition": "WaveFake (Neural Vocoders) + ASVspoof 2019 LA (TTS/VC A01-A19)",
        "train_samples": len(comb_train),
        "val_samples": len(comb_val),
        "test_samples": len(comb_test),
        "combined_aasist_metrics": m_aasist,
        "combined_clad_metrics": m_clad
    }
    
    os.makedirs("checkpoints/audio", exist_ok=True)
    if w_aasist is not None:
        torch.save(w_aasist, "checkpoints/audio/aasist_combined_multidataset_best.pth")
        print("[+] Saved AASIST combined checkpoint to checkpoints/audio/aasist_combined_multidataset_best.pth")
    if w_clad is not None:
        torch.save(w_clad, "checkpoints/audio/clad_combined_multidataset_best.pth")
        print("[+] Saved CLAD combined checkpoint to checkpoints/audio/clad_combined_multidataset_best.pth")
        
    os.makedirs("experiments/audio", exist_ok=True)
    out_path = "experiments/audio/combined_multidataset_audio_metrics.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[✓] Saved combined multi-dataset metrics to {out_path}")

if __name__ == "__main__":
    main()
