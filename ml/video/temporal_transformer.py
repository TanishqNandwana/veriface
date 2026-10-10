"""
Temporal Transformer Encoder (TTE) for Video Deepfake Detection
Base paper reference: Choi et al. (CVPR 2024) - Style Latent Flows
Aggregates temporal sequence features to produce frame-level & video-level predictions.
"""

import torch
import torch.nn as nn
from typing import Tuple

class TemporalTransformerEncoder(nn.Module):
    def __init__(self, embed_dim: int = 256, depth: int = 4, num_heads: int = 8, max_len: int = 64):
        super(TemporalTransformerEncoder, self).__init__()
        self.pos_embed = nn.Parameter(torch.zeros(1, max_len, embed_dim))
        nn.init.normal_(self.pos_embed, std=0.02)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            batch_first=True,
            activation="gelu"
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        self.norm = nn.LayerNorm(embed_dim)
        
        self.frame_head = nn.Linear(embed_dim, 1)
        self.video_head = nn.Sequential(
            nn.Linear(embed_dim, 128),
            nn.GELU(),
            nn.Linear(128, 1)
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        x: [B, T_seq, embed_dim]
        returns: (video_logits [B, 1], frame_logits [B, T_seq, 1])
        """
        B, T_seq, _ = x.shape
        x = x + self.pos_embed[:, :T_seq, :]
        
        out = self.transformer(x)
        out = self.norm(out) # [B, T_seq, embed_dim]
        
        frame_logits = self.frame_head(out) # [B, T_seq, 1]
        
        # Temporal pool for video-level score
        video_feat = torch.mean(out, dim=1) # [B, embed_dim]
        video_logits = self.video_head(video_feat) # [B, 1]
        
        return video_logits, frame_logits

if __name__ == "__main__":
    dummy_x = torch.randn(2, 15, 256)
    tte = TemporalTransformerEncoder()
    v_logits, f_logits = tte(dummy_x)
    print("TTE Video Logits shape:", v_logits.shape, "Frame Logits shape:", f_logits.shape)
