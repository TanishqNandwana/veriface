"""
AASIST Spectro-Temporal Feature Encoder
Base paper reference: Jung et al. (ICASSP 2022) / Wu et al. (2024) - CLAD
Extracts graph-based spectro-temporal artifact feature embeddings [B, embed_dim] from audio inputs.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class AASISTConvStem(nn.Module):
    def __init__(self, in_channels: int = 1, out_channels: int = 64):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=(5, 5), stride=(2, 2), padding=(2, 2))
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, out_channels, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.act(self.bn1(self.conv1(x)))
        x = self.act(self.bn2(self.conv2(x)))
        return x

class GraphAttentionLayer(nn.Module):
    def __init__(self, in_features: int = 64, out_features: int = 128):
        super().__init__()
        self.fc = nn.Linear(in_features, out_features)
        self.attn = nn.Linear(out_features * 2, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T_seq, in_features]
        h = self.fc(x) # [B, T_seq, out_features]
        B, T_seq, C = h.shape
        
        # Self-attention adjacency graph weights
        h_i = h.unsqueeze(2).repeat(1, 1, T_seq, 1)
        h_j = h.unsqueeze(1).repeat(1, T_seq, 1, 1)
        pair = torch.cat([h_i, h_j], dim=-1) # [B, T_seq, T_seq, 2*C]
        
        scores = F.leaky_relu(self.attn(pair).squeeze(-1))
        adj = F.softmax(scores, dim=-1)
        
        out = torch.matmul(adj, h)
        return out

class AASISTEncoder(nn.Module):
    def __init__(self, in_channels: int = 1, embed_dim: int = 256):
        super(AASISTEncoder, self).__init__()
        self.stem = AASISTConvStem(in_channels=in_channels, out_channels=64)
        self.proj_seq = nn.Linear(64 * 20, 64)
        self.gat = GraphAttentionLayer(in_features=64, out_features=128)
        self.fc = nn.Linear(128, embed_dim)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, mel_tensor: torch.Tensor) -> torch.Tensor:
        """
        mel_tensor: [B, 1, n_mels, time_steps]
        returns: Feature embedding vector [B, embed_dim]
        """
        if mel_tensor.dim() == 3:
            mel_tensor = mel_tensor.unsqueeze(1) # Add channel dim
            
        x_stem = self.stem(mel_tensor) # [B, 64, H', W']
        
        # Flatten spatial dims to sequence length [B, W', 64*H'] -> map to 64 dims
        B, C, H, W = x_stem.shape
        x_seq = x_stem.permute(0, 3, 1, 2).reshape(B, W, C * H)
        x_proj = self.proj_seq(x_seq)
        
        gat_out = self.gat(x_proj) # [B, W, 128]
        pooled = torch.mean(gat_out, dim=1) # [B, 128]
        
        embed = self.norm(self.fc(pooled))
        return embed

if __name__ == "__main__":
    dummy_mel = torch.randn(2, 1, 80, 250) # B=2, 80 mels, 250 frames
    encoder = AASISTEncoder()
    emb = encoder(dummy_mel)
    print("AASIST Encoder Output shape:", emb.shape)
