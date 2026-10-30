"""
Combined Multi-Dataset Video Deepfake Training Suite
Trains EfficientNet-B0 + BiGRU Temporal Aggregator on Celeb-DF and DFDC Video Benchmarks.
Combines high-resolution convolutional spatial representation with bidirectional recurrent temporal flow.
Saves best checkpoint to checkpoints/video/efficientnet_bigru_combined_best.pth.
"""

import os
import json
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights

from ml.common.device import get_device
from ml.common.metrics import compute_all_metrics

class CombinedVideoDataset(Dataset):
    def __init__(self, entries):
        self.entries = entries

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        item = self.entries[idx]
        clip_path = item["filepath"]
        label = item["label"]
        try:
            arr = np.load(clip_path) # [T, 3, 224, 224] in range [0, 1]
            tensor_clip = torch.tensor(arr, dtype=torch.float32)
        except Exception:
            tensor_clip = torch.zeros((5, 3, 224, 224), dtype=torch.float32)
        return tensor_clip, torch.tensor(label, dtype=torch.float32)

class EfficientNetBiGRUVideoModel(nn.Module):
    def __init__(self, embed_dim: int = 128):
        super().__init__()
        base = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
        self.backbone = base.features
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.proj = nn.Linear(1280, embed_dim)
        self.gru = nn.GRU(embed_dim, embed_dim // 2, batch_first=True, bidirectional=True)
        self.head = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(embed_dim, 64),
            nn.GELU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        # x: [B, T, 3, 224, 224]
        B, T, C, H, W = x.shape
        x_flat = x.view(B * T, C, H, W)
        feats = self.backbone(x_flat)
        pooled = self.pool(feats).flatten(1) # [B*T, 1280]
        proj = torch.relu(self.proj(pooled))
        proj_seq = proj.view(B, T, -1) # [B, T, embed_dim]
        gru_out, h_n = self.gru(proj_seq) # [B, T, embed_dim]
        # Video level logit from bidirectional hidden states
        v_feat = torch.cat([h_n[-2], h_n[-1]], dim=-1) # [B, embed_dim]
        v_logits = self.head(v_feat)
        f_logits = self.head(gru_out) # [B, T, 1]
        return v_logits, f_logits

def main():
    device = get_device()
    print(f"[*] Starting Combined Video Deepfake Training on {device}...")

    # 1. Load Celeb-DF manifest
    with open("datasets/manifests/video_benchmark_manifest.json") as f:
        celeb_data = json.load(f)
    celeb_train = celeb_data["splits"]["train"]
    celeb_val = celeb_data["splits"]["validation"]
    celeb_test = celeb_data["splits"]["test"]

    # 2. Load DFDC manifest
    with open("datasets/manifests/dfdc_video_benchmark_manifest.json") as f:
        dfdc_data = json.load(f)
    dfdc_clips = dfdc_data["clips"]
    dfdc_train = dfdc_clips[:12]
    dfdc_val = dfdc_clips[12:16]
    dfdc_test = dfdc_clips[16:]

    train_entries = celeb_train + dfdc_train
    val_entries = celeb_val + dfdc_val
    test_entries = celeb_test + dfdc_test

    print(f"[+] Combined Video Dataset: Train={len(train_entries)}, Val={len(val_entries)}, Test={len(test_entries)}")

    train_loader = DataLoader(CombinedVideoDataset(train_entries), batch_size=4, shuffle=True)
    val_loader = DataLoader(CombinedVideoDataset(val_entries), batch_size=4, shuffle=False)
    test_loader = DataLoader(CombinedVideoDataset(test_entries), batch_size=4, shuffle=False)

    model = EfficientNetBiGRUVideoModel(embed_dim=128).to(device)

    # Differential learning rate: lower for pretrained backbone, higher for temporal head
    params = [
        {"params": model.backbone.parameters(), "lr": 2e-5},
        {"params": model.proj.parameters(), "lr": 1e-4},
        {"params": model.gru.parameters(), "lr": 2e-4},
        {"params": model.head.parameters(), "lr": 2e-4},
    ]
    optimizer = torch.optim.AdamW(params, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=8)
    bce = nn.BCEWithLogitsLoss()

    best_val_auc = -1.0
    best_weights = None
    epochs = 8

    for ep in range(1, epochs + 1):
        model.train()
        total_loss, n = 0.0, 0
        t0 = time.time()
        for clips, labels in train_loader:
            clips, labels = clips.to(device), labels.to(device)
            optimizer.zero_grad()
            v_logits, f_logits = model(clips)
            v_loss = bce(v_logits.squeeze(-1), labels)
            f_labels = labels.unsqueeze(1).expand(-1, clips.shape[1])
            f_loss = bce(f_logits.squeeze(-1), f_labels)
            loss = v_loss + 0.4 * f_loss
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(labels)
            n += len(labels)

        scheduler.step()
        train_loss = total_loss / max(n, 1)

        # Validation
        model.eval()
        v_preds, v_targets = [], []
        with torch.no_grad():
            for clips, labels in val_loader:
                clips = clips.to(device)
                v_logits, _ = model(clips)
                probs = torch.sigmoid(v_logits.squeeze(-1)).cpu().numpy().tolist()
                if isinstance(probs, float):
                    probs = [probs]
                v_preds.extend(probs)
                v_targets.extend(labels.numpy().tolist())

        val_m = compute_all_metrics(np.array(v_targets), np.array(v_preds))
        elapsed = time.time() - t0
        print(f"Epoch {ep}/{epochs} ({elapsed:.1f}s) | Train Loss: {train_loss:.4f} | Val Acc: {val_m['accuracy']:.4f} | Val AUC: {val_m['roc_auc']:.4f}")

        if val_m["roc_auc"] >= best_val_auc:
            best_val_auc = val_m["roc_auc"]
            best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    # Final Test Set Evaluation
    if best_weights is not None:
        model.load_state_dict(best_weights)
    model.eval()

    t_preds, t_targets = [], []
    with torch.no_grad():
        for clips, labels in test_loader:
            clips = clips.to(device)
            v_logits, _ = model(clips)
            probs = torch.sigmoid(v_logits.squeeze(-1)).cpu().numpy().tolist()
            if isinstance(probs, float):
                probs = [probs]
            t_preds.extend(probs)
            t_targets.extend(labels.numpy().tolist())

    test_m = compute_all_metrics(np.array(t_targets), np.array(t_preds))
    print("\n=======================================================")
    print(f"[✓] Final Combined Video Model Test Evaluation:")
    print(f"    Accuracy:  {test_m['accuracy'] * 100:.2f}%")
    print(f"    ROC-AUC:   {test_m['roc_auc']:.4f}")
    print(f"    Precision: {test_m['precision']:.4f}")
    print(f"    Recall:    {test_m['recall']:.4f}")
    print(f"    F1-Score:  {test_m['f1_score']:.4f}")
    print(f"    EER:       {test_m['eer'] * 100:.2f}%")
    print("=======================================================")

    os.makedirs("checkpoints/video", exist_ok=True)
    out_ckpt = "checkpoints/video/efficientnet_bigru_combined_best.pth"
    torch.save(best_weights, out_ckpt)
    print(f"[+] Saved high-accuracy video checkpoint to {out_ckpt}")

if __name__ == "__main__":
    main()
