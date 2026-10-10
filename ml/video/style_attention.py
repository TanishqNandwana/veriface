"""
Style Attention Module (SAM) for Deepfake Video Detection
Base paper reference: Choi et al. (CVPR 2024) - Style Latent Flows
Cross-attends StyleGRU temporal style flows with 3D ResNet visual content features.
"""

import torch
import torch.nn as nn

class StyleAttentionModule(nn.Module):
    def __init__(self, content_dim: int = 256, style_dim: int = 256, embed_dim: int = 256, num_heads: int = 8):
        super(StyleAttentionModule, self).__init__()
        self.proj_content = nn.Linear(content_dim, embed_dim)
        self.proj_style = nn.Linear(style_dim, embed_dim)
        
        self.cross_attn = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, batch_first=True)
        self.norm = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.GELU(),
            nn.Linear(embed_dim * 2, embed_dim)
        )

    def forward(self, content_feats: torch.Tensor, style_feats: torch.Tensor) -> torch.Tensor:
        """
        content_feats: [B, T, content_dim] (3D ResNet visual features across T frames)
        style_feats: [B, T-1, style_dim] (StyleGRU temporal style flows)
        returns: Style-attended temporal visual features [B, T-1, embed_dim]
        """
        B, T, _ = content_feats.shape
        # Align content length to T-1 by taking adjacent frame feature means
        c_paired = (content_feats[:, :-1] + content_feats[:, 1:]) / 2.0 # [B, T-1, content_dim]
        
        q = self.proj_content(c_paired)
        k = self.proj_style(style_feats)
        v = k
        
        attn_out, _ = self.cross_attn(query=q, key=k, value=v)
        fused = self.norm(q + attn_out)
        out = fused + self.mlp(fused)
        return out

if __name__ == "__main__":
    content = torch.randn(2, 16, 256)
    style = torch.randn(2, 15, 256)
    sam = StyleAttentionModule()
    out = sam(content, style)
    print("Style Attention Output shape:", out.shape)
