"""
Length Loss for CLAD Audio Deepfake Detector
Base paper reference: Wu et al. (2024) - CLAD
Penalizes embedding magnitude discrepancies across dual augmented views (z_i, z_j)
to enforce scale invariance and robustness against volume/manipulation attacks.
"""

import torch
import torch.nn as nn

class LengthLoss(nn.Module):
    def __init__(self):
        super(LengthLoss, self).__init__()

    def forward(self, z_i: torch.Tensor, z_j: torch.Tensor) -> torch.Tensor:
        """
        z_i: Feature embeddings of view 1 [B, D]
        z_j: Feature embeddings of view 2 [B, D]
        """
        norm_i = torch.norm(z_i, p=2, dim=1)
        norm_j = torch.norm(z_j, p=2, dim=1)
        loss = torch.mean((norm_i - norm_j) ** 2)
        return loss

if __name__ == "__main__":
    z1 = torch.randn(4, 128)
    z2 = torch.randn(4, 128)
    criterion = LengthLoss()
    loss = criterion(z1, z2)
    print("Length Loss test:", loss.item())
