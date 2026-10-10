"""
Phase 5 Video Training & Benchmarking Suite for VeriFace
Trains and evaluates:
1. 3D ResNet Video Classifier Baseline
2. Frame Feature Extractor + BiGRU Temporal Aggregation Baseline
3. Full Style Latent Flow Video Model (StyleGRU + 3D ResNet + SAM + TTE)
Evaluates video-level and frame-level predictions, compares aggregation strategies (Mean, Median, Max-Margin, Learned),
and saves checkpoints and logs to checkpoints/video/ and experiments/video/.
"""

import os
import time
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from typing import Dict, Any, List

from ml.common.device import get_device
from ml.common.metrics import compute_all_metrics
from ml.video.model import StyleLatentFlowVideoModel
from ml.video.content_3d_resnet import Content3DResNet

class RealVideoClipDataset(Dataset):
    def __init__(self, entries: list):
        self.entries = entries

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        item = self.entries[idx]
        clip_path = item["filepath"]
        label = item["label"]
        try:
            arr = np.load(clip_path) # [T, 3, 224, 224] in [0, 1]
            tensor_clip = torch.tensor(arr, dtype=torch.float32)
        except Exception:
            tensor_clip = torch.zeros((5, 3, 224, 224), dtype=torch.float32)
        return tensor_clip, torch.tensor(label, dtype=torch.float32)

class Content3DResNetClassifier(nn.Module):
    def __init__(self, embed_dim: int = 128):
        super().__init__()
        self.encoder = Content3DResNet(in_channels=3, out_dim=embed_dim)
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(embed_dim, 64),
            nn.GELU(),
            nn.Linear(64, 1)
        )

    def forward(self, clip_tensor: torch.Tensor):
        # clip_tensor: [B, T, 3, 224, 224]
        feats = self.encoder(clip_tensor) # [B, T, embed_dim]
        pooled = feats.mean(dim=1) # [B, embed_dim]
        v_logits = self.classifier(pooled)
        f_logits = self.classifier(feats) # [B, T, 1]
        return v_logits, f_logits

class FrameBiGRUTemporalModel(nn.Module):
    def __init__(self, embed_dim: int = 128):
        super().__init__()
        # Lightweight 2D frame feature stem
        self.frame_stem = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
            nn.Linear(64 * 16, embed_dim),
            nn.GELU()
        )
        self.gru = nn.GRU(embed_dim, embed_dim // 2, batch_first=True, bidirectional=True)
        self.head = nn.Linear(embed_dim, 1)

    def forward(self, clip_tensor: torch.Tensor):
        B, T, C, H, W = clip_tensor.shape
        flat_frames = clip_tensor.view(B * T, C, H, W)
        feats = self.frame_stem(flat_frames) # [B*T, embed_dim]
        seq_feats = feats.view(B, T, -1)
        gru_out, _ = self.gru(seq_feats) # [B, T, embed_dim]
        f_logits = self.head(gru_out) # [B, T, 1]
        v_logits = self.head(gru_out.mean(dim=1)) # [B, 1]
        return v_logits, f_logits

def train_and_eval_video_model(
    model_name: str,
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    device: torch.device,
    epochs: int = 10,
    lr: float = 2e-4
) -> Dict[str, Any]:
    print(f"\n=======================================================")
    print(f"[*] Training Video Model: {model_name} on {device} (Epochs: {epochs})")
    print(f"=======================================================")

    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    criterion = nn.BCEWithLogitsLoss()

    best_val_auc = -1.0
    best_weights = None
    history = {"train_loss": [], "val_acc": [], "val_auc": []}
    t_start = time.time()

    for ep in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        n_batches = 0

        for clips, labels in train_loader:
            clips, labels = clips.to(device), labels.to(device)
            optimizer.zero_grad()
            v_logits, f_logits = model(clips)
            
            # Joint supervision on video and frame logits
            v_loss = criterion(v_logits.squeeze(-1), labels)
            f_loss = criterion(f_logits.mean(dim=1).squeeze(-1), labels)
            loss = v_loss + 0.4 * f_loss
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1

        scheduler.step()
        epoch_loss = total_loss / max(1, n_batches)
        history["train_loss"].append(round(epoch_loss, 4))

        # Validation
        model.eval()
        v_preds, v_targets = [], []
        with torch.no_grad():
            for clips, labels in val_loader:
                clips = clips.to(device)
                v_logits, _ = model(clips)
                probs = torch.sigmoid(v_logits).squeeze(-1).cpu().numpy().tolist()
                if isinstance(probs, float):
                    probs = [probs]
                v_preds.extend(probs)
                v_targets.extend(labels.numpy().tolist())

        val_m = compute_all_metrics(np.array(v_targets), np.array(v_preds))
        history["val_acc"].append(round(val_m["accuracy"], 4))
        history["val_auc"].append(round(val_m["roc_auc"], 4))

        print(f"  Epoch {ep:02d}/{epochs:02d} | Train Loss: {epoch_loss:.4f} | Val Acc: {val_m['accuracy']:.4f} | Val AUC: {val_m['roc_auc']:.4f}")

        if val_m["roc_auc"] >= best_val_auc:
            best_val_auc = val_m["roc_auc"]
            best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    train_duration = round(time.time() - t_start, 2)

    # Load best checkpoint
    model.load_state_dict(best_weights)
    model.eval()

    # In-Domain Held-out Test Evaluation
    test_preds, test_frame_preds, test_targets, latencies = [], [], [], []
    with torch.no_grad():
        for clips, labels in test_loader:
            clips = clips.to(device)
            t0 = time.time()
            v_logits, f_logits = model(clips)
            lat = (time.time() - t0) * 1000 / clips.size(0)
            latencies.append(lat)

            probs = torch.sigmoid(v_logits).squeeze(-1).cpu().numpy().tolist()
            if isinstance(probs, float):
                probs = [probs]
            f_probs = torch.sigmoid(f_logits).squeeze(-1).cpu().numpy().tolist()
            
            test_preds.extend(probs)
            test_frame_preds.extend(f_probs)
            test_targets.extend(labels.numpy().tolist())

    test_metrics = compute_all_metrics(np.array(test_targets), np.array(test_preds))
    test_metrics["mean_latency_ms"] = round(float(np.mean(latencies)), 2)

    # Aggregation Strategy Comparison on Test Set
    mean_agg_preds = [float(np.mean(f)) for f in test_frame_preds]
    median_agg_preds = [float(np.median(f)) for f in test_frame_preds]
    max_agg_preds = [float(np.max(f)) for f in test_frame_preds]

    agg_comparison = {
        "learned_video_pooling": test_metrics,
        "mean_frame_probability": compute_all_metrics(np.array(test_targets), np.array(mean_agg_preds)),
        "median_frame_probability": compute_all_metrics(np.array(test_targets), np.array(median_agg_preds)),
        "max_frame_probability": compute_all_metrics(np.array(test_targets), np.array(max_agg_preds))
    }

    return {
        "model_name": model_name,
        "train_duration_sec": train_duration,
        "best_val_auc": round(best_val_auc, 4),
        "history": history,
        "test_metrics": test_metrics,
        "aggregation_comparison": agg_comparison,
        "best_state_dict": best_weights
    }

def main():
    device = get_device()
    print(f"[*] Starting Video Multi-Dataset & Baseline Experimentation on {device}")
    
    manifest_path = "datasets/manifests/video_benchmark_manifest.json"
    with open(manifest_path, "r") as f:
        manifest = json.load(f)

    train_ds = RealVideoClipDataset(manifest["splits"]["train"])
    val_ds = RealVideoClipDataset(manifest["splits"]["validation"])
    test_ds = RealVideoClipDataset(manifest["splits"]["test"])

    print(f"[+] Splits: Train={len(train_ds)}, Val={len(val_ds)}, Test={len(test_ds)}")

    train_loader = DataLoader(train_ds, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=8, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=8, shuffle=False)

    os.makedirs("checkpoints/video", exist_ok=True)
    os.makedirs("experiments/video", exist_ok=True)

    models_to_test = [
        ("3D ResNet Video Classifier", Content3DResNetClassifier(embed_dim=128), 2e-4),
        ("Frame-CNN + BiGRU Temporal Aggregator", FrameBiGRUTemporalModel(embed_dim=128), 2e-4),
        ("Style Latent Flow Video Model (Full)", StyleLatentFlowVideoModel(num_frames=5, embed_dim=128), 1e-4)
    ]

    all_experiments = {}

    for name, m, lr in models_to_test:
        res = train_and_eval_video_model(
            model_name=name,
            model=m,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            device=device,
            epochs=10,
            lr=lr
        )
        
        safe_name = name.lower().replace(" ", "_").replace("+", "_").replace("(", "").replace(")", "").replace("-", "_")
        ckpt_path = f"checkpoints/video/{safe_name}_best.pth"
        torch.save(res["best_state_dict"], ckpt_path)
        del res["best_state_dict"]
        res["checkpoint_path"] = ckpt_path

        exp_file = f"experiments/video/{safe_name}_metrics.json"
        with open(exp_file, "w") as f:
            json.dump(res, f, indent=2)

        all_experiments[name] = res
        print(f"[✓] {name} Complete | Test Acc: {res['test_metrics']['accuracy']:.4f} | Test AUC: {res['test_metrics']['roc_auc']:.4f} | F1: {res['test_metrics']['f1_score']:.4f}")

    with open("experiments/video/comparative_summary.json", "w") as f:
        json.dump(all_experiments, f, indent=2)

    print("\n=======================================================")
    print(" Video Modality Benchmark Summary (Celeb-DF Held-out) ")
    print("=======================================================")
    print(f"{'Model Name':<38} | {'Accuracy':<8} | {'Precision':<9} | {'Recall':<8} | {'F1-Score':<8} | {'ROC-AUC':<8}")
    print("-" * 88)
    for name, r in all_experiments.items():
        m = r["test_metrics"]
        print(f"{name:<38} | {m['accuracy']:<8.4f} | {m['precision']:<9.4f} | {m['recall']:<8.4f} | {m['f1_score']:<8.4f} | {m['roc_auc']:<8.4f}")

if __name__ == "__main__":
    main()
