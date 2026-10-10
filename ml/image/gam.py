"""
Global Adaptive Module (GAM) for FA-ViT
Injects lightweight bottleneck adaptation into ViT self-attention / feed-forward layers.
Uses zero-initialization on output projection to maintain initial ViT representation stability.
"""

import torch
import torch.nn as nn

class GlobalAdaptiveModule(nn.Module):
    def __init__(self, embed_dim: int = 768, bottleneck_dim: int = 64):
        super(GlobalAdaptiveModule, self).__init__()
        self.down_proj = nn.Linear(embed_dim, bottleneck_dim)
        self.act = nn.GELU()
        self.up_proj = nn.Linear(bottleneck_dim, embed_dim)
        
        # Zero-initialization for output projection
        nn.init.zeros_(self.up_proj.weight)
        nn.init.zeros_(self.up_proj.bias)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, N, C] where B is batch size, N is number of tokens, C is embed_dim
        """
        residual = x
        adapted = self.down_proj(x)
        adapted = self.act(adapted)
        adapted = self.up_proj(adapted)
        return residual + adapted

if __name__ == "__main__":
    dummy_x = torch.randn(2, 197, 768)
    gam = GlobalAdaptiveModule(768, 64)
    out = gam(dummy_x)
    print("GAM input shape:", dummy_x.shape, "output shape:", out.shape)
    assert torch.allclose(dummy_x, out), "Zero initialization failed!"
    print("GAM zero-initialization check passed!")
