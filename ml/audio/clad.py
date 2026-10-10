"""
CLAD Architecture (Contrastive Learning-based Audio Deepfake Detector)
Base paper reference: Wu et al. (2024)
Integrates AASIST spectro-temporal encoder, MoCo contrastive learning projection,
length loss, and downstream classification head.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

from ml.audio.aasist_encoder import AASISTEncoder
from ml.audio.length_loss import LengthLoss

class CLADModel(nn.Module):
    def __init__(self, in_channels: int = 1, embed_dim: int = 256, proj_dim: int = 128):
        super(CLADModel, self).__init__()
        self.encoder = AASISTEncoder(in_channels=in_channels, embed_dim=embed_dim)
        
        # Projection head for MoCo contrastive learning
        self.proj_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, proj_dim)
        )
        
        # Downstream binary classifier
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, 128),
            nn.GELU(),
            nn.Linear(128, 1)
        )

    def forward_encoder(self, mel_tensor: torch.Tensor) -> torch.Tensor:
        return self.encoder(mel_tensor)

    def forward_contrastive(self, mel_tensor: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        emb = self.encoder(mel_tensor)
        proj = self.proj_head(emb)
        return emb, proj

    def forward(self, mel_tensor: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        mel_tensor: [B, 1, n_mels, time_steps]
        returns: (logits [B, 1], embedding [B, embed_dim])
        """
        emb = self.encoder(mel_tensor)
        logits = self.classifier(emb)
        return logits, emb

if __name__ == "__main__":
    model = CLADModel()
    dummy_mel = torch.randn(2, 1, 80, 250)
    logits, emb = model(dummy_mel)
    print("CLAD Model Output Test: Logits shape =", logits.shape, "Embedding shape =", emb.shape)
