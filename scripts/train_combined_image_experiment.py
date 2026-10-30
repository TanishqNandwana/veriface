"""
Phase 11 Experiment D: Combined Multi-Dataset Image Training (Celeb-DF + FFHQ vs StyleGAN)
Evaluates whether combining facial boundary manipulation (Celeb-DF) with
pure AI generative synthesis (FFHQ vs StyleGAN) improves cross-dataset generalization.

Trains:
1. Combined EfficientNet-B0 Classifier
2. Combined FA-ViT Model (GAM + LAM + FFT + Spatial Forensics + FAL Loss)

Evaluates on:
- In-domain Celeb-DF test set
- Generative StyleGAN test set
- Combined multi-domain test set

Saves results to experiments/image/combined_multidataset_image_metrics.json.
"""

import os
import json
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image

from ml.common.device import get_device
from ml.common.metrics import compute_all_metrics
from ml.image.preprocessing import DEFAULT_TRANSFORM
from ml.image.baselines import EfficientNetBaseline
from ml.image.favit import FAViTModel
from ml.image.fal_loss import FAViTLoss

class CombinedImageDataset(Dataset):
    def __init__(self, items, transform=DEFAULT_TRANSFORM):
        self.items = items
        self.transform = transform

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        item = self.items[idx]
        img_path = item["filepath"]
        label = item["label"]
        try:
            pil_img = Image.open(img_path).convert("RGB")
            tensor_img = self.transform(pil_img)
        except Exception:
            tensor_img = torch.zeros((3, 224, 224))
        return tensor_img, torch.tensor(label, dtype=torch.float32)

def train_image_model(model_name, model, train_items, val_items, test_celeb_items, test_stylegan_items, test_comb_items, device, epochs=6, lr=1e-4, is_favit=False):
    print(f"\n=======================================================")
    print(f"[*] Training {model_name} on Combined Image Multi-Dataset (Epochs: {epochs})...")
    print(f"=======================================================")
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    
    if is_favit:
        criterion = FAViTLoss(alpha_fal=0.1)
    else:
        bce = nn.BCEWithLogitsLoss()
        
    train_loader = DataLoader(CombinedImageDataset(train_items), batch_size=16, shuffle=True)
    val_loader = DataLoader(CombinedImageDataset(val_items), batch_size=16, shuffle=False)
    
    best_val_auc = -1.0
    best_weights = None
    
    for ep in range(1, epochs + 1):
        model.train()
        total_loss, n = 0.0, 0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            logits, feats = model(imgs)
            logits = logits.squeeze(-1)
            
            if is_favit:
                loss = criterion(logits, feats, labels)
            else:
                loss = bce(logits, labels)
                
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(labels)
            n += len(labels)
            
        train_loss = total_loss / max(n, 1)
        
        # Validation
        model.eval()
        val_targets, val_preds = [], []
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs = imgs.to(device)
                logits, _ = model(imgs)
                probs = torch.sigmoid(logits.squeeze(-1)).cpu().numpy()
                val_preds.extend(probs.tolist() if isinstance(probs.tolist(), list) else [probs.item()])
                val_targets.extend(labels.numpy().tolist())
                
        val_metrics = compute_all_metrics(np.array(val_targets), np.array(val_preds))
        val_auc = val_metrics["roc_auc"]
        print(f"  --> Epoch {ep:02d}/{epochs:02d} | Train Loss: {train_loss:.4f} | Val Acc: {val_metrics['accuracy']:.4f} | Val AUC: {val_auc:.4f}")
        
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            
    if best_weights is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_weights.items()})
    model.eval()
    
    def evaluate_set(items, set_name):
        loader = DataLoader(CombinedImageDataset(items), batch_size=16, shuffle=False)
        targets, preds, lats = [], [], []
        with torch.no_grad():
            for imgs, labels in loader:
                imgs = imgs.to(device)
                t0 = time.time()
                logits, _ = model(imgs)
                lat = (time.time() - t0) * 1000 / len(labels)
                probs = torch.sigmoid(logits.squeeze(-1)).cpu().numpy()
                preds.extend(probs.tolist() if isinstance(probs.tolist(), list) else [probs.item()])
                targets.extend(labels.numpy().tolist())
                lats.extend([lat] * len(labels))
        m = compute_all_metrics(np.array(targets), np.array(preds))
        m["mean_latency_ms"] = round(float(np.mean(lats)), 2)
        print(f"[✓] {model_name} on {set_name:20s} | Acc: {m['accuracy']:.4f} | AUC: {m['roc_auc']:.4f} | Recall: {m['recall']:.4f}")
        return m

    metrics_celeb = evaluate_set(test_celeb_items, "Celeb-DF Test")
    metrics_stylegan = evaluate_set(test_stylegan_items, "StyleGAN Test")
    metrics_combined = evaluate_set(test_comb_items, "Combined Test")
    
    return {
        "model_name": model_name,
        "best_val_auc": round(float(best_val_auc), 4),
        "test_celebdf": metrics_celeb,
        "test_stylegan": metrics_stylegan,
        "test_combined": metrics_combined,
        "weights": best_weights
    }

def main():
    device = get_device()
    print(f"[*] Starting Combined Multi-Dataset Image Training on {device}")
    
    # 1. Load Celeb-DF manifest
    celeb_manifest_path = "datasets/manifests/image_benchmark_manifest.json"
    with open(celeb_manifest_path, "r") as f:
        celeb_data = json.load(f)
    celeb_splits = celeb_data["splits"]
    celeb_train = celeb_splits["train"]
    celeb_val = celeb_splits.get("validation", celeb_splits.get("val", []))
    celeb_test = celeb_splits["test"]
    print(f"[+] Loaded Celeb-DF splits: Train={len(celeb_train)}, Val={len(celeb_val)}, Test={len(celeb_test)}")
    
    # 2. Load FFHQ vs StyleGAN manifest
    stylegan_manifest_path = "datasets/manifests/ffhq_stylegan_manifest.json"
    with open(stylegan_manifest_path, "r") as f:
        sg_data = json.load(f)
    sg_samples = sg_data["samples"]
    
    reals = [s for s in sg_samples if s["label"] == 0]
    fakes = [s for s in sg_samples if s["label"] == 1]
    
    # Split FFHQ/StyleGAN into 120 train (60r, 60f), 40 val (20r, 20f), 40 test (20r, 20f)
    sg_train = reals[:60] + fakes[:60]
    sg_val = reals[60:80] + fakes[60:80]
    sg_test = reals[80:100] + fakes[80:100]
    print(f"[+] Split FFHQ/StyleGAN: Train={len(sg_train)}, Val={len(sg_val)}, Test={len(sg_test)}")
    
    # Combine datasets
    comb_train = celeb_train + sg_train
    comb_val = celeb_val + sg_val
    comb_test = celeb_test + sg_test
    print(f"[+] Total Combined: Train={len(comb_train)}, Val={len(comb_val)}, Combined Test={len(comb_test)}")
    
    # Train 1: EfficientNet-B0
    eff_model = EfficientNetBaseline(pretrained=True)
    res_eff = train_image_model(
        "EfficientNet-B0 (Combined)",
        eff_model,
        comb_train, comb_val,
        celeb_test, sg_test, comb_test,
        device, epochs=6, lr=1e-4, is_favit=False
    )
    
    # Train 2: FA-ViT
    favit_model = FAViTModel(use_gam=True, use_lam=True, use_fft=True, use_spatial_noise=True, pretrained=True)
    res_favit = train_image_model(
        "FA-ViT (Combined)",
        favit_model,
        comb_train, comb_val,
        celeb_test, sg_test, comb_test,
        device, epochs=6, lr=8e-5, is_favit=True
    )
    
    # Save best checkpoints
    os.makedirs("checkpoints/image", exist_ok=True)
    if res_eff["weights"] is not None:
        eff_save_path = "checkpoints/image/efficientnet_b0_combined_multidataset_best.pth"
        torch.save(res_eff["weights"], eff_save_path)
        print(f"[+] Saved Combined EfficientNet-B0 weights to {eff_save_path}")
        
    if res_favit["weights"] is not None:
        favit_save_path = "checkpoints/image/fa_vit_combined_multidataset_best.pth"
        torch.save(res_favit["weights"], favit_save_path)
        print(f"[+] Saved Combined FA-ViT weights to {favit_save_path}")
        
    # Serialize results
    results_to_save = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "device": str(device),
        "dataset_composition": {
            "celeb_df_train": len(celeb_train),
            "celeb_df_val": len(celeb_val),
            "celeb_df_test": len(celeb_test),
            "ffhq_stylegan_train": len(sg_train),
            "ffhq_stylegan_val": len(sg_val),
            "ffhq_stylegan_test": len(sg_test),
            "total_combined_train": len(comb_train),
            "total_combined_val": len(comb_val),
            "total_combined_test": len(comb_test)
        },
        "models": {
            "efficientnet_b0_combined": {
                "best_val_auc": res_eff["best_val_auc"],
                "test_celebdf": res_eff["test_celebdf"],
                "test_stylegan": res_eff["test_stylegan"],
                "test_combined": res_eff["test_combined"]
            },
            "fa_vit_combined": {
                "best_val_auc": res_favit["best_val_auc"],
                "test_celebdf": res_favit["test_celebdf"],
                "test_stylegan": res_favit["test_stylegan"],
                "test_combined": res_favit["test_combined"]
            }
        }
    }
    
    os.makedirs("experiments/image", exist_ok=True)
    out_json = "experiments/image/combined_multidataset_image_metrics.json"
    with open(out_json, "w") as f:
        json.dump(results_to_save, f, indent=2)
    print(f"\n[✓] Combined Multi-Dataset Image results successfully saved to {out_json}")

if __name__ == "__main__":
    main()
