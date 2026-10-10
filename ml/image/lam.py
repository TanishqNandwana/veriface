"""
Local Adaptive Module (LAM) for FA-ViT
Captures patch-level spatial relationship & local forgery boundary artifacts.
Reshapes 1D token sequences into 2D spatial feature maps for convolutional processing.
"""

import math
import torch
import torch.nn as nn

class LocalAdaptiveModule(nn.Module):
    def __init__(self, embed_dim: int = 768, bottleneck_dim: int = 64):
        super(LocalAdaptiveModule, self).__init__()
        self.down_proj = nn.Linear(embed_dim, bottleneck_dim)
        
        # Depthwise separable convolution for local spatial interactions
        self.conv = nn.Sequential(
            nn.Conv2d(bottleneck_dim, bottleneck_dim, kernel_size=3, padding=1, groups=bottleneck_dim),
            nn.BatchNorm2d(bottleneck_dim),
            nn.GELU(),
            nn.Conv2d(bottleneck_dim, bottleneck_dim, kernel_size=1),
            nn.BatchNorm2d(bottleneck_dim),
            nn.GELU()
        )
        
        self.up_proj = nn.Linear(bottleneck_dim, embed_dim)
        
        # Zero-initialization for output projection
        nn.init.zeros_(self.up_proj.weight)
        nn.init.zeros_(self.up_proj.bias)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, N, C] where N = 1 + H*W (1 CLS token + patch tokens)
        """
        B, N, C = x.shape
        cls_token = x[:, :1, :]
        patch_tokens = x[:, 1:, :]
        
        num_patches = patch_tokens.shape[1]
        h = w = int(math.sqrt(num_patches))
        assert h * w == num_patches, f"Patch count {num_patches} must be a perfect square."
        
        # Project patches down
        p_down = self.down_proj(patch_tokens) # [B, H*W, bottleneck_dim]
        
        # Reshape to 2D grid [B, bottleneck_dim, H, W]
        p_grid = p_down.permute(0, 2, 1).view(B, -1, h, w)
        
        # 2D local spatial convolution
        conv_out = self.conv(p_grid)
        
        # Flatten back to [B, H*W, bottleneck_dim]
        p_flat = conv_out.view(B, conv_out.shape[1], -1).permute(0, 2, 1)
        
        # Project up
        p_up = self.up_proj(p_flat)
        
        # Combine CLS token and local spatial patches
        out_patch = patch_tokens + p_up
        out = torch.cat([cls_token, out_patch], dim=1)
        return out

if __name__ == "__main__":
    dummy_x = torch.randn(2, 197, 768) # 1 CLS + 14x14 patches = 197
    lam = LocalAdaptiveModule(768, 64)
    out = lam(dummy_x)
    print("LAM input shape:", dummy_x.shape, "output shape:", out.shape)
    assert torch.allclose(dummy_x, out), "LAM zero-initialization check failed!"
    print("LAM zero-initialization check passed!")
