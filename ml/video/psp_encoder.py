"""
pSp Style Latent Encoder & Temporal Style Flow Extractor
Base paper reference: Choi et al. (CVPR 2024) - Style Latent Flows
Maps video frame facial crops to 18-layer W+ latent space [B, 18, 512]
and calculates temporal style differences \Delta S_t = S_{t+1} - S_t.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class StyleLatentEncoder(nn.Module):
    def __init__(self, num_style_layers: int = 18, style_dim: int = 512):
        super(StyleLatentEncoder, self).__init__()
        self.num_style_layers = num_style_layers
        self.style_dim = style_dim
        
        # Lightweight ResNet-style backbone for W+ latent extraction
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.GELU(),
            nn.AdaptiveAvgPool2d((8, 8))
        )
        self.style_proj = nn.Linear(128 * 8 * 8, num_style_layers * style_dim)

    def forward(self, frame_tensor: torch.Tensor) -> torch.Tensor:
        """
        frame_tensor: [B, 3, 224, 224]
        returns: W+ style latent code [B, 18, 512]
        """
        B = frame_tensor.shape[0]
        feat = self.backbone(frame_tensor)
        feat_flat = feat.view(B, -1)
        styles = self.style_proj(feat_flat)
        styles_w = styles.view(B, self.num_style_layers, self.style_dim)
        return styles_w

def compute_temporal_style_diffs(style_sequence: torch.Tensor) -> torch.Tensor:
    """
    style_sequence: [B, T, 18, 512] style latent codes across T frames
    returns: \Delta S temporal style differences [B, T-1, 18, 512]
    """
    # \Delta S_t = S_{t+1} - S_t
    style_diffs = style_sequence[:, 1:] - style_sequence[:, :-1]
    return style_diffs

if __name__ == "__main__":
    encoder = StyleLatentEncoder()
    dummy_frames = torch.randn(4, 3, 224, 224) # 4 frames batch
    styles = encoder(dummy_frames)
    print("pSp Style Latent Output shape:", styles.shape)
    
    seq_styles = styles.unsqueeze(0) # [1, 4, 18, 512]
    diffs = compute_temporal_style_diffs(seq_styles)
    print("Temporal Style Diffs shape:", diffs.shape)
