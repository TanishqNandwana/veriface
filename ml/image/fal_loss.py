"""
Fine-grained Adaptive Learning (FAL) Loss for FA-ViT
Incorporates Circle Loss / Pairwise Margin Adaptive Weighting on subtle forgery pairs.
Supervises feature representations to maximize distance between real and subtle fake samples.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class FALCircleLoss(nn.Module):
    def __init__(self, margin: float = 0.25, gamma: float = 64.0):
        super(FALCircleLoss, self).__init__()
        self.margin = margin
        self.gamma = gamma
        self.op = 1 + margin
        self.on = -margin
        self.delta_p = 1 - margin
        self.delta_n = margin

    def forward(self, feats: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        feats: Normalized feature embeddings [B, D]
        targets: Binary labels [B] (0 = real, 1 = fake)
        """
        feats = F.normalize(feats, p=2, dim=1)
        sim_matrix = torch.matmul(feats, feats.T) # [B, B]
        
        B = targets.size(0)
        mask_pos = (targets.unsqueeze(0) == targets.unsqueeze(1)) & (~torch.eye(B, dtype=torch.bool, device=targets.device))
        mask_neg = targets.unsqueeze(0) != targets.unsqueeze(1)
        
        if not mask_pos.any() or not mask_neg.any():
            return torch.tensor(0.0, device=targets.device, requires_grad=True)
            
        sp = sim_matrix[mask_pos]
        sn = sim_matrix[mask_neg]
        
        alpha_p = torch.clamp(self.op - sp.detach(), min=0.0)
        alpha_n = torch.clamp(sn.detach() - self.on, min=0.0)
        
        logit_p = -self.gamma * alpha_p * (sp - self.delta_p)
        logit_n = self.gamma * alpha_n * (sn - self.delta_n)
        
        # Numerically stable LogSumExp formulation avoiding overflow in torch.exp
        lse_n = torch.logsumexp(logit_n, dim=0)
        lse_p = torch.logsumexp(logit_p, dim=0)
        loss = F.softplus(lse_n + lse_p)
        return loss

class FAViTLoss(nn.Module):
    def __init__(self, alpha_fal: float = 0.1, margin: float = 0.25, gamma: float = 64.0):
        super(FAViTLoss, self).__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.fal = FALCircleLoss(margin=margin, gamma=gamma)
        self.alpha_fal = alpha_fal

    def forward(self, logits: torch.Tensor, feats: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        logits: [B, 1] raw classification outputs
        feats: [B, D] feature embeddings
        targets: [B] ground truth labels
        """
        bce_loss = self.bce(logits.squeeze(-1), targets.float())
        fal_loss = self.fal(feats, targets)
        return bce_loss + self.alpha_fal * fal_loss

if __name__ == "__main__":
    feats = torch.randn(4, 128)
    logits = torch.randn(4, 1)
    targets = torch.tensor([0, 0, 1, 1])
    criterion = FAViTLoss()
    loss = criterion(logits, feats, targets)
    print("FAViT Loss test:", loss.item())
