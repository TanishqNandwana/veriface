"""
FA-ViT Architecture (Forgery-Aware Adaptive Vision Transformer)
Base paper reference: Luo et al. (2025)
Combines pretrained ViT backbone with Global Adaptive Modules (GAM),
Local Adaptive Modules (LAM), FFT Frequency Forensic Cues, and Spatial Noise Residuals
for generalized face forgery & AI-generated image detection.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from ml.image.gam import GlobalAdaptiveModule
from ml.image.lam import LocalAdaptiveModule
from ml.image.fft_features import FFTFeatureExtractor
from ml.image.spatial_forensics import SpatialNoiseExtractor

class FAViTModel(nn.Module):
    def __init__(
        self,
        img_size: int = 224,
        embed_dim: int = 768,
        bottleneck_dim: int = 64,
        use_gam: bool = True,
        use_lam: bool = True,
        use_fft: bool = True,
        use_spatial_noise: bool = True,
        pretrained: bool = True,
        freeze_backbone: bool = True
    ):
        super().__init__()
        self.use_gam = use_gam
        self.use_lam = use_lam
        self.use_fft = use_fft
        self.use_spatial_noise = use_spatial_noise
        self.freeze_backbone = freeze_backbone
        
        # Load torchvision ViT-B/16 pretrained backbone if requested
        try:
            import torchvision.models as models
            if pretrained:
                weights = models.ViT_B_16_Weights.DEFAULT
                vit_model = models.vit_b_16(weights=weights)
            else:
                vit_model = models.vit_b_16(weights=None)
                
            self.patch_embed = vit_model.conv_proj
            self.cls_token = vit_model.class_token
            self.pos_embed = vit_model.encoder.pos_embedding
            self.encoder_layers = vit_model.encoder.layers
            self.encoder_norm = vit_model.encoder.ln
        except Exception as e:
            print(f"[!] Warning: Pretrained ViT load fallback: {e}")
            # Fallback self-contained ViT encoder stem
            self.patch_embed = nn.Conv2d(3, embed_dim, kernel_size=16, stride=16)
            self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
            self.pos_embed = nn.Parameter(torch.zeros(1, 197, embed_dim))
            encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=12, batch_first=True)
            self.encoder_layers = nn.TransformerEncoder(encoder_layer, num_layers=6)
            self.encoder_norm = nn.LayerNorm(embed_dim)

        if self.freeze_backbone:
            for mod in [self.patch_embed, self.encoder_layers, self.encoder_norm]:
                if hasattr(mod, "parameters"):
                    for p in mod.parameters():
                        p.requires_grad = False
            if hasattr(self, "cls_token") and isinstance(self.cls_token, nn.Parameter):
                self.cls_token.requires_grad = False
            if hasattr(self, "pos_embed") and isinstance(self.pos_embed, nn.Parameter):
                self.pos_embed.requires_grad = False

        if self.use_gam:
            self.gam = GlobalAdaptiveModule(embed_dim, bottleneck_dim)
        if self.use_lam:
            self.lam = LocalAdaptiveModule(embed_dim, bottleneck_dim)
            
        if self.use_fft:
            self.fft_extractor = FFTFeatureExtractor(in_channels=3, out_dim=128)
            fft_dim = 128
        else:
            fft_dim = 0
            
        if self.use_spatial_noise:
            self.spatial_noise_extractor = SpatialNoiseExtractor(out_dim=128)
            spatial_dim = 128
        else:
            spatial_dim = 0
            
        classifier_in_dim = embed_dim + fft_dim + spatial_dim
        
        self.fc_feat = nn.Linear(classifier_in_dim, 256)
        self.head = nn.Linear(256, 1)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        x_img = x
        
        # Patch Embed: [B, C, H, W] -> [B, N, C]
        if hasattr(self.patch_embed, 'flatten'):
            x_tokens = self.patch_embed(x).flatten(2).transpose(1, 2)
        else:
            x_tokens = self.patch_embed(x).flatten(2).transpose(1, 2)
            
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x_seq = torch.cat((cls_tokens, x_tokens), dim=1)
        x_seq = x_seq + self.pos_embed
        
        # Transformer Layer Execution with GAM & LAM Adaptation
        if isinstance(self.encoder_layers, nn.ModuleList):
            for layer in self.encoder_layers:
                x_seq = layer(x_seq)
        else:
            x_seq = self.encoder_layers(x_seq)
            
        x_seq = self.encoder_norm(x_seq)
        
        if self.use_gam:
            x_seq = self.gam(x_seq)
        if self.use_lam:
            x_seq = self.lam(x_seq)
            
        cls_feat = x_seq[:, 0] # [B, embed_dim]
        
        extra_feats = [cls_feat]
        
        if self.use_fft:
            fft_feat = self.fft_extractor(x_img)
            extra_feats.append(fft_feat)
            
        if self.use_spatial_noise:
            spatial_feat = self.spatial_noise_extractor(x_img)
            extra_feats.append(spatial_feat)
            
        combined_feat = torch.cat(extra_feats, dim=-1)
        return combined_feat

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.forward_features(x)
        embed = F.gelu(self.fc_feat(feat))
        logits = self.head(embed)
        return logits, embed

if __name__ == "__main__":
    model = FAViTModel(use_gam=True, use_lam=True, use_fft=True, use_spatial_noise=True, pretrained=False)
    dummy = torch.randn(2, 3, 224, 224)
    logits, embed = model(dummy)
    print("FA-ViT Logits shape:", logits.shape, "Embedding shape:", embed.shape)
