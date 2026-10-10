"""
FA-ViT & Image Baseline Training & Ablation Script
Executes reproducible image deepfake detection experiments, saving checkpoints,
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
from PIL import Image

from ml.common.device import get_device, get_device_info
from ml.common.metrics import compute_all_metrics
from ml.common.dataset_builder import split_manifest, generate_synthetic_image_dataset
from ml.image.preprocessing import DEFAULT_TRANSFORM
from ml.image.favit import FAViTModel
from ml.image.fal_loss import FAViTLoss

class ManifestDataset(Dataset):
    def __init__(self, entries: list, transform=DEFAULT_TRANSFORM):
        self.entries = entries
        self.transform = transform

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        item = self.entries[idx]
        img_path = item["filepath"]
        label = item["label"]
        
        try:
            img = Image.open(img_path).convert("RGB")
            tensor_img = self.transform(img)
        except Exception:
            tensor_img = torch.zeros((3, 224, 224))
            
        return tensor_img, torch.tensor(label, dtype=torch.long)

def train_image_experiment(args):
    device = get_device()
    print(f"[*] Starting Image Experiment '{args.exp_id}' on device: {device}")
    
    exp_dir = os.path.join("experiments", args.exp_id)
    os.makedirs(exp_dir, exist_ok=True)
    
    # Dataset Preparation
    if not os.path.exists(args.dataset_manifest):
        print(f"[!] Manifest {args.dataset_manifest} not found. Generating synthetic image dataset...")
        syn_dir = os.path.join("datasets", "synthetic_image_exp")
        args.dataset_manifest = generate_synthetic_image_dataset(syn_dir, num_real=30, num_fake=30)
        
    train_entries, val_entries, test_entries = split_manifest(args.dataset_manifest, seed=args.seed)
    print(f"[+] Dataset splits: Train={len(train_entries)}, Val={len(val_entries)}, Test={len(test_entries)}")
    
    train_loader = DataLoader(ManifestDataset(train_entries), batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(ManifestDataset(val_entries), batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(ManifestDataset(test_entries), batch_size=args.batch_size, shuffle=False)
    
    # Model Initialization
    model = FAViTModel(
        img_size=224,
        embed_dim=768,
        bottleneck_dim=64,
        use_gam=args.use_gam,
        use_lam=args.use_lam,
        use_fft=args.use_fft,
        use_spatial_noise=True,
        pretrained=True
    ).to(device)
    
    criterion = FAViTLoss(alpha_fal=0.1 if args.use_fal else 0.0)
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    
    best_val_auc = 0.0
    start_time = time.time()
    
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        
        for batch_imgs, batch_labels in train_loader:
            batch_imgs = batch_imgs.to(device)
            batch_labels = batch_labels.to(device)
            
            optimizer.zero_grad()
            logits, feats = model(batch_imgs)
            loss = criterion(logits, feats, batch_labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * batch_imgs.size(0)
            
        scheduler.step()
        train_loss /= len(train_entries)
        
        # Validation
        model.eval()
        val_preds = []
        val_targets = []
        
        with torch.no_grad():
            for batch_imgs, batch_labels in val_loader:
                batch_imgs = batch_imgs.to(device)
                logits, _ = model(batch_imgs)
                probs = torch.sigmoid(logits).squeeze(-1).cpu().numpy()
                val_preds.extend(probs.tolist() if isinstance(probs.tolist(), list) else [probs.item()])
                val_targets.extend(batch_labels.numpy().tolist())
                
        val_metrics = compute_all_metrics(np.array(val_targets), np.array(val_preds))
        print(f"Epoch {epoch}/{args.epochs} | Train Loss: {train_loss:.4f} | Val Acc: {val_metrics['accuracy']} | Val AUC: {val_metrics['roc_auc']}")
        
        if val_metrics['roc_auc'] >= best_val_auc:
            best_val_auc = val_metrics['roc_auc']
            torch.save(model.state_dict(), os.path.join(exp_dir, "best_checkpoint.pth"))
            
    training_duration = round(time.time() - start_time, 2)
    
    # Test Evaluation on Best Checkpoint
    model.load_state_dict(torch.load(os.path.join(exp_dir, "best_checkpoint.pth")))
    model.eval()
    test_preds = []
    test_targets = []
    
    with torch.no_grad():
        for batch_imgs, batch_labels in test_loader:
            batch_imgs = batch_imgs.to(device)
            logits, _ = model(batch_imgs)
            probs = torch.sigmoid(logits).squeeze(-1).cpu().numpy()
            test_preds.extend(probs.tolist() if isinstance(probs.tolist(), list) else [probs.item()])
            test_targets.extend(batch_labels.numpy().tolist())
            
    final_metrics = compute_all_metrics(np.array(test_targets), np.array(test_preds))
    print(f"[+] Final Test Evaluation: {final_metrics}")
    
    # Save Experiment Manifest
    exp_summary = {
        "experiment_id": args.exp_id,
        "modality": "image",
        "model_architecture": "FA-ViT",
        "config": vars(args),
        "device_info": get_device_info(),
        "training_duration_sec": training_duration,
        "best_val_auc": best_val_auc,
        "test_metrics": final_metrics,
        "checkpoint_path": os.path.join(exp_dir, "best_checkpoint.pth")
    }
    
    with open(os.path.join(exp_dir, "metrics.json"), "w") as f:
        json.dump(exp_summary, f, indent=2)
        
    print(f"[✓] Experiment '{args.exp_id}' completed successfully!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-id", type=str, default="IMG_FAVIT_SMOKE_001")
    parser.add_argument("--dataset-manifest", type=str, default="datasets/synthetic_image_exp/manifest.json")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--use-gam", action="store_true", default=True)
    parser.add_argument("--use-lam", action="store_true", default=True)
    parser.add_argument("--use-fal", action="store_true", default=True)
    parser.add_argument("--use-fft", action="store_true", default=True)
    parser.add_argument("--unfreeze-backbone", action="store_true", default=False)
    parser.add_argument("--seed", type=int, default=42)
    
    args = parser.parse_args()
    train_image_experiment(args)
