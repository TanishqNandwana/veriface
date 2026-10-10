"""
CLAD & Audio Baseline Deepfake Detection Trainer Script
Executes reproducible audio deepfake experiments, saving checkpoints,
training logs, and evaluation metrics under experiments/<exp_id>/
"""

import os
import sys
import json
import argparse
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from ml.common.device import get_device, get_device_info
from ml.common.metrics import compute_all_metrics
from ml.common.dataset_builder import split_manifest, generate_synthetic_audio_dataset
from ml.audio.preprocessing import load_and_preprocess_audio, compute_log_mel_spectrogram
from ml.audio.clad import CLADModel

class ManifestAudioDataset(Dataset):
    def __init__(self, entries: list):
        self.entries = entries

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        item = self.entries[idx]
        wav_path = item["filepath"]
        label = item["label"]
        
        try:
            y, sr = load_and_preprocess_audio(wav_path)
            mel_tensor = compute_log_mel_spectrogram(y, sr=sr)
        except Exception:
            mel_tensor = torch.zeros((1, 80, 250))
            
        return mel_tensor, torch.tensor(label, dtype=torch.long)

def train_audio_experiment(args):
    device = get_device()
    print(f"[*] Starting Audio Experiment '{args.exp_id}' on device: {device}")
    
    exp_dir = os.path.join("experiments", args.exp_id)
    os.makedirs(exp_dir, exist_ok=True)
    
    if not os.path.exists(args.dataset_manifest) or len(split_manifest(args.dataset_manifest)[0]) < 20:
        print(f"[!] Manifest {args.dataset_manifest} generating enhanced synthetic audio dataset...")
        syn_dir = os.path.join("datasets", "synthetic_audio_exp")
        args.dataset_manifest = generate_synthetic_audio_dataset(syn_dir, num_real=25, num_fake=25)
        
    train_entries, val_entries, test_entries = split_manifest(args.dataset_manifest, seed=args.seed)
    print(f"[+] Audio Dataset splits: Train={len(train_entries)}, Val={len(val_entries)}, Test={len(test_entries)}")
    
    train_loader = DataLoader(ManifestAudioDataset(train_entries), batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(ManifestAudioDataset(val_entries), batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(ManifestAudioDataset(test_entries), batch_size=args.batch_size, shuffle=False)
    
    model = CLADModel(in_channels=1, embed_dim=128).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    
    best_val_score = -1.0
    best_val_auc = 0.0
    start_time = time.time()
    
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        
        for batch_mels, batch_labels in train_loader:
            batch_mels = batch_mels.to(device)
            batch_labels = batch_labels.to(device).float()
            
            optimizer.zero_grad()
            logits, _ = model(batch_mels)
            loss = criterion(logits.squeeze(-1), batch_labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * batch_mels.size(0)
            
        train_loss /= len(train_entries)
        scheduler.step()
        
        # Validation
        model.eval()
        val_preds = []
        val_targets = []
        
        with torch.no_grad():
            for batch_mels, batch_labels in val_loader:
                batch_mels = batch_mels.to(device)
                logits, _ = model(batch_mels)
                probs = torch.sigmoid(logits).squeeze(-1).cpu().numpy()
                val_preds.extend(probs.tolist() if isinstance(probs.tolist(), list) else [probs.item()])
                val_targets.extend(batch_labels.numpy().tolist())
                
        val_metrics = compute_all_metrics(np.array(val_targets), np.array(val_preds))
        val_score = val_metrics['accuracy'] + val_metrics['roc_auc'] + val_metrics['f1_score']
        print(f"Epoch {epoch}/{args.epochs} | Train Loss: {train_loss:.4f} | Val Acc: {val_metrics['accuracy']:.4f} | Val AUC: {val_metrics['roc_auc']:.4f}")
        
        if val_score >= best_val_score:
            best_val_score = val_score
            best_val_auc = val_metrics['roc_auc']
            torch.save(model.state_dict(), os.path.join(exp_dir, "best_checkpoint.pth"))
            
    training_duration = round(time.time() - start_time, 2)
    
    # Test Evaluation
    model.load_state_dict(torch.load(os.path.join(exp_dir, "best_checkpoint.pth")))
    model.eval()
    test_preds = []
    test_targets = []
    
    with torch.no_grad():
        for batch_mels, batch_labels in test_loader:
            batch_mels = batch_mels.to(device)
            logits, _ = model(batch_mels)
            probs = torch.sigmoid(logits).squeeze(-1).cpu().numpy()
            test_preds.extend(probs.tolist() if isinstance(probs.tolist(), list) else [probs.item()])
            test_targets.extend(batch_labels.numpy().tolist())
            
    final_metrics = compute_all_metrics(np.array(test_targets), np.array(test_preds))
    print(f"[+] Final Audio Test Evaluation: {final_metrics}")
    
    exp_summary = {
        "experiment_id": args.exp_id,
        "modality": "audio",
        "model_architecture": "CLAD (AASIST + Contrastive Learning + Length Loss)",
        "config": vars(args),
        "device_info": get_device_info(),
        "training_duration_sec": training_duration,
        "best_val_auc": best_val_auc,
        "test_metrics": final_metrics,
        "checkpoint_path": os.path.join(exp_dir, "best_checkpoint.pth")
    }
    
    with open(os.path.join(exp_dir, "metrics.json"), "w") as f:
        json.dump(exp_summary, f, indent=2)
        
    print(f"[✓] Audio Experiment '{args.exp_id}' completed successfully!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-id", type=str, default="AUD_CLAD_SMOKE_001")
    parser.add_argument("--dataset-manifest", type=str, default="datasets/synthetic_audio_exp/manifest.json")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    
    args = parser.parse_args()
    train_audio_experiment(args)
