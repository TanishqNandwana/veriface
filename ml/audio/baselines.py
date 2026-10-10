"""
Audio Deepfake Detection Baseline Architectures
1. AASIST Baseline Classifier (Jung et al. 2022)
2. Multi-Scale Spectrogram CNN Baseline
"""

import torch
import torch.nn as nn
from ml.audio.aasist_encoder import AASISTEncoder

class AASISTBaselineClassifier(nn.Module):
    def __init__(self, embed_dim: int = 128):
        super().__init__()
        self.encoder = AASISTEncoder(in_channels=1, embed_dim=embed_dim)
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(embed_dim, 64),
            nn.GELU(),
            nn.Linear(64, 1)
        )

    def forward(self, x: torch.Tensor):
        emb = self.encoder(x)
        logits = self.classifier(emb)
        return logits, emb

class SpectrogramCNNBaseline(nn.Module):
    def __init__(self, in_channels: int = 1, embed_dim: int = 128):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.MaxPool2d(2)
        )
        self.fc = nn.Sequential(
            nn.Linear(64, embed_dim),
            nn.GELU()
        )
        self.head = nn.Linear(embed_dim, 1)

    def forward(self, x: torch.Tensor):
        feat_map = self.conv(x)
        pooled = feat_map.mean(dim=[-2, -1])
        feats = self.fc(pooled)
        logits = self.head(feats)
        return logits, feats
