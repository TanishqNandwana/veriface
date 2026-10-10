"""
FFT Frequency Forensic Cue Extractor
Computes 2D Discrete Fourier Transform (FFT) high-frequency spectrum magnitude features.
Detects upsampling/generative grid artifacts in deepfake faces.
"""

import torch
import torch.nn as nn
import torch.fft

class FFTFeatureExtractor(nn.Module):
    def __init__(self, in_channels: int = 3, out_dim: int = 128):
        super(FFTFeatureExtractor, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.MaxPool2d(2), # [B, 32, H/2, W/2]
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.AdaptiveAvgPool2d((8, 8))
        )
        self.fc = nn.Linear(64 * 8 * 8, out_dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: Input image tensor [B, C, H, W]
        returns: High-frequency spectral feature vector [B, out_dim]
        """
        # 2D FFT per channel
        # shift zero-frequency components to center
        fft_complex = torch.fft.fft2(x, norm="ortho")
        fft_shift = torch.fft.fftshift(fft_complex, dim=(-2, -1))
        magnitude = torch.abs(fft_shift) + 1e-8
        log_magnitude = torch.log(magnitude)
        
        feat = self.conv(log_magnitude)
        feat_flat = feat.view(feat.size(0), -1)
        out = self.fc(feat_flat)
        return out

if __name__ == "__main__":
    dummy_img = torch.randn(2, 3, 224, 224)
    fft_ext = FFTFeatureExtractor(in_channels=3, out_dim=128)
    feat = fft_ext(dummy_img)
    print("FFT Extractor input shape:", dummy_img.shape, "output shape:", feat.shape)
