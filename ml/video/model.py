"""
Style Latent Flow Video Deepfake Detection Architecture
Base paper reference: Choi et al. (CVPR 2024) - Style Latent Flows
Integrates pSp style latent extraction, temporal style diffs \Delta S, StyleGRU,
3D ResNet-50 content branch, Style Attention Module (SAM), and Temporal Transformer Encoder (TTE).
"""

import torch
import torch.nn as nn
from typing import Tuple

from ml.video.psp_encoder import StyleLatentEncoder, compute_temporal_style_diffs
from ml.video.style_gru import StyleGRU
from ml.video.content_3d_resnet import Content3DResNet
from ml.video.style_attention import StyleAttentionModule
from ml.video.temporal_transformer import TemporalTransformerEncoder

class StyleLatentFlowVideoModel(nn.Module):
    def __init__(self, num_frames: int = 16, embed_dim: int = 256):
        super(StyleLatentFlowVideoModel, self).__init__()
        self.num_frames = num_frames
        
        self.psp_encoder = StyleLatentEncoder()
        self.style_gru = StyleGRU(hidden_dim=embed_dim)
        self.content_3d = Content3DResNet(out_dim=embed_dim)
        self.sam = StyleAttentionModule(content_dim=embed_dim, style_dim=embed_dim, embed_dim=embed_dim)
        self.tte = TemporalTransformerEncoder(embed_dim=embed_dim)

    def forward(self, clip_tensor: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        clip_tensor: Video frames clip [B, T, 3, 224, 224]
        returns: (video_logits [B, 1], frame_logits [B, T-1, 1])
        """
        B, T, C, H, W = clip_tensor.shape
        
        # 1. Extract pSp W+ style latent codes per frame
        frames_flat = clip_tensor.view(B * T, C, H, W)
        styles_flat = self.psp_encoder(frames_flat) # [B*T, 18, 512]
        styles_seq = styles_flat.view(B, T, 18, 512)
        
        # 2. Compute temporal style differences \Delta S_t
        style_diffs = compute_temporal_style_diffs(styles_seq) # [B, T-1, 18, 512]
        
        # 3. Model style flow trajectory with StyleGRU
        style_flow_feats = self.style_gru(style_diffs) # [B, T-1, embed_dim]
        
        # 4. Extract 3D visual content appearance features
        content_feats = self.content_3d(clip_tensor) # [B, T, embed_dim]
        
        # 5. Fuse style flows and 3D visual features using Style Attention Module (SAM)
        fused_feats = self.sam(content_feats, style_flow_feats) # [B, T-1, embed_dim]
        
        # 6. Temporal Transformer Encoder sequence aggregation
        video_logits, frame_logits = self.tte(fused_feats)
        
        return video_logits, frame_logits

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
        feats = self.encoder(clip_tensor) # [B, T, embed_dim]
        pooled = feats.mean(dim=1) # [B, embed_dim]
        v_logits = self.classifier(pooled)
        f_logits = self.classifier(feats) # [B, T, 1]
        return v_logits, f_logits

class FrameBiGRUTemporalModel(nn.Module):
    def __init__(self, embed_dim: int = 128):
        super().__init__()
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

if __name__ == "__main__":
    model = StyleLatentFlowVideoModel()
    dummy_video = torch.randn(2, 8, 3, 224, 224) # B=2, T=8
    v_logits, f_logits = model(dummy_video)
    print("Video Model Output Test: Video Logits =", v_logits.shape, "Frame Logits =", f_logits.shape)

