"""
StyleGRU Module for Temporal Style Flow Modeling
Processes temporal style differences \Delta S_t through a Gated Recurrent Unit (GRU)
to capture non-linear latent space trajectory anomalies across video frames.
"""

import torch
import torch.nn as nn

class StyleGRU(nn.Module):
    def __init__(self, style_dim: int = 18 * 512, hidden_dim: int = 256, num_layers: int = 2):
        super(StyleGRU, self).__init__()
        self.gru = nn.GRU(
            input_size=style_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True
        )
        self.fc = nn.Linear(hidden_dim * 2, hidden_dim)

    def forward(self, style_diffs: torch.Tensor) -> torch.Tensor:
        """
        style_diffs: [B, T-1, 18, 512] or [B, T-1, 18*512]
        returns: Temporal style flow representation [B, T-1, hidden_dim]
        """
        B, Tm1 = style_diffs.shape[:2]
        if style_diffs.dim() == 4:
            style_diffs = style_diffs.view(B, Tm1, -1) # Flatten style dimensions [B, T-1, 18*512]
            
        gru_out, _ = self.gru(style_diffs) # [B, T-1, hidden_dim * 2]
        style_flow_feats = self.fc(gru_out) # [B, T-1, hidden_dim]
        return style_flow_feats

if __name__ == "__main__":
    dummy_diffs = torch.randn(2, 15, 18, 512) # B=2, T-1=15
    style_gru = StyleGRU()
    flow_feats = style_gru(dummy_diffs)
    print("StyleGRU Output shape:", flow_feats.shape)
