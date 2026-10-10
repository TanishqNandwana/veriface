"""
Image Deepfake Detection Baseline Architectures
1. ResNet-50 (Pretrained on ImageNet-1K with modified binary head)
2. EfficientNet-B0 (Pretrained on ImageNet-1K with modified binary head)
3. Vanilla ViT-B/16 (Pretrained Vision Transformer with standard linear head)
"""

import torch
import torch.nn as nn
import torchvision.models as models

class ResNet50Baseline(nn.Module):
    def __init__(self, pretrained: bool = True):
        super().__init__()
        weights = models.ResNet50_Weights.DEFAULT if pretrained else None
        self.backbone = models.resnet50(weights=weights)
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(in_features, 128),
            nn.GELU(),
            nn.Linear(128, 1)
        )

    def forward(self, x: torch.Tensor):
        logits = self.backbone(x)
        return logits, logits

class EfficientNetBaseline(nn.Module):
    def __init__(self, pretrained: bool = True):
        super().__init__()
        weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
        self.backbone = models.efficientnet_b0(weights=weights)
        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(in_features, 128),
            nn.GELU(),
            nn.Linear(128, 1)
        )

    def forward(self, x: torch.Tensor):
        logits = self.backbone(x)
        return logits, logits

class VanillaViTBaseline(nn.Module):
    def __init__(self, pretrained: bool = True):
        super().__init__()
        weights = models.ViT_B_16_Weights.DEFAULT if pretrained else None
        self.backbone = models.vit_b_16(weights=weights)
        in_features = self.backbone.heads.head.in_features
        self.backbone.heads.head = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(in_features, 128),
            nn.GELU(),
            nn.Linear(128, 1)
        )

    def forward(self, x: torch.Tensor):
        logits = self.backbone(x)
        return logits, logits
