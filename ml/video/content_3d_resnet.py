"""
3D ResNet Visual Content Feature Extractor
Base paper reference: Choi et al. (CVPR 2024) - Style Latent Flows
Extracts 3D spatiotemporal appearance features from video frame clips [B, C, T, H, W].
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class Conv3dBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, stride=(1, 2, 2)):
        super().__init__()
        self.conv = nn.Conv3d(in_ch, out_ch, kernel_size=(3, 3, 3), padding=(1, 1, 1), stride=stride)
        self.bn = nn.BatchNorm3d(out_ch)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))

class Content3DResNet(nn.Module):
    def __init__(self, in_channels: int = 3, out_dim: int = 256):
        super(Content3DResNet, self).__init__()
        self.stem = Conv3dBlock(in_channels, 64, stride=(1, 2, 2))
        
        self.layer1 = Conv3dBlock(64, 128, stride=(1, 2, 2))
        self.layer2 = Conv3dBlock(128, 256, stride=(1, 2, 2))
        
        self.pool = nn.AdaptiveAvgPool3d((None, 1, 1)) # Keep temporal dimension T intact
        self.proj = nn.Linear(256, out_dim)

    def forward(self, clip_tensor: torch.Tensor) -> torch.Tensor:
        """
        clip_tensor: [B, C, T, H, W] or [B, T, C, H, W]
        returns: Temporal content feature sequence [B, T, out_dim]
        """
        if clip_tensor.dim() == 5 and clip_tensor.shape[1] != 3 and clip_tensor.shape[2] == 3:
            # If shape is [B, T, C, H, W], transpose to [B, C, T, H, W]
            clip_tensor = clip_tensor.permute(0, 2, 1, 3, 4)
            
        x = self.stem(clip_tensor)
        x = self.layer1(x)
        x = self.layer2(x) # [B, 256, T, H', W']
        
        x_pooled = x.mean(dim=(-2, -1)) # Spatial average pool [B, 256, T]
        x_temp = x_pooled.permute(0, 2, 1) # [B, T, 256]
        
        content_feats = self.proj(x_temp) # [B, T, out_dim]
        return content_feats

if __name__ == "__main__":
    net = Content3DResNet()
    dummy_clip = torch.randn(2, 3, 16, 224, 224) # B=2, C=3, T=16
    feats = net(dummy_clip)
    print("3D ResNet Content Features Output shape:", feats.shape)
