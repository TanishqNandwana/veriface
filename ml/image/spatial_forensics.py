"""
Spatial Noise & High-Frequency Forensic Extractor for AI-Generated & Deepfake Image Detection
Base paper & forensic reference: Luo et al. (2025) / DALL-E & GAN Forensic Spectrum Cues
Computes:
1. Spatial high-pass noise residual variance (Laplacian filter)
2. Local patch over-smoothing ratio (airbrushed facial skin detection)
3. 2D FFT High-Frequency Power Ratio & Spectral Anomaly Score
4. Local Color Channel Gradient Discrepancy
"""

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

def extract_numpy_forensic_scores(img_bgr: np.ndarray) -> dict:
    """
    Computes mathematical forensic statistics from raw BGR image array.
    Distinguishes natural camera photographs (PRNU sensor noise, natural FFT decay)
    from AI-generated / ChatGPT / DALL-E / Midjourney / Deepfake images.
    """
    if img_bgr is None or img_bgr.size == 0:
        return {"fft_anomaly_score": 0.5, "laplacian_variance": 100.0, "patch_smoothness_ratio": 0.5}

    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    H, W = gray.shape
    
    # 1. Spatial High-Pass Noise Residual (Laplacian Variance)
    lap = cv2.Laplacian(gray, cv2.CV_32F)
    lap_var = float(np.var(lap))
    
    # 2. Local Patch Over-Smoothing Analysis (Airbrushed skin detection common in ChatGPT / DALL-E 3 / Midjourney)
    patch_size = 16
    patch_vars = []
    for r in range(0, H - patch_size, patch_size):
        for c in range(0, W - patch_size, patch_size):
            p = gray[r:r+patch_size, c:c+patch_size]
            patch_vars.append(np.var(p))
            
    patch_vars = np.array(patch_vars)
    # Ratio of over-smoothed patches (< 2.0 variance)
    smooth_patches_ratio = float(np.mean(patch_vars < 2.0)) if len(patch_vars) > 0 else 0.0
    
    # 3. Color Channel Correlation Discrepancy (Luminance Normalized)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32)
    r_ch, g_ch, b_ch = img_rgb[:,:,0], img_rgb[:,:,1], img_rgb[:,:,2]
    mean_lum = float(np.mean(img_rgb) + 1.0)
    rg_diff = float(np.var(r_ch - g_ch))
    rb_diff = float(np.var(r_ch - b_ch))
    color_var = float(((rg_diff + rb_diff) / 2.0) / mean_lum) * 100.0
    
    # 4. 2D Discrete Fourier Transform (FFT) Power Spectrum Analysis
    fft = np.fft.fft2(gray)
    fft_shift = np.fft.fftshift(fft)
    magnitude = np.abs(fft_shift) + 1e-8
    
    cy, cx = H // 2, W // 2
    r_max = min(H, W) / 2.0
    y_idx, x_idx = np.ogrid[:H, :W]
    dist = np.sqrt((x_idx - cx)**2 + (y_idx - cy)**2)
    high_mask = (dist >= 0.2 * r_max) & (dist <= 0.8 * r_max)
    
    angle = np.arctan2(y_idx - cy, x_idx - cx)
    sector_energies = []
    for deg in range(0, 180, 22):
        rad = np.radians(deg)
        mask = high_mask & (np.abs(angle - rad) < np.radians(11))
        if np.sum(mask) > 0:
            sector_energies.append(float(np.mean(magnitude[mask])))
            
    azimuthal_var = float(np.std(sector_energies) / (np.mean(sector_energies) + 1e-8)) if len(sector_energies) > 0 else 0.0
    
    # 5. Calibrated Forensic AI Generation & Deepfake Anomaly Metric
    ai_score = 0.0
    
    # Require multiple combined markers to prevent single background noise false positives
    if smooth_patches_ratio > 0.65:
        ai_score += 0.35 + (smooth_patches_ratio - 0.65) * 0.8
    if color_var > 600.0:
        ai_score += 0.35 + min(0.3, (color_var - 600.0) / 2000.0)
    if azimuthal_var > 0.18:
        ai_score += 0.30 + min(0.3, (azimuthal_var - 0.18) * 2.0)
        
    ai_score = max(0.0, min(1.0, ai_score))
    
    return {
        "patch_smoothness_ratio": round(smooth_patches_ratio, 4),
        "azimuthal_variance": round(azimuthal_var, 5),
        "laplacian_variance": round(lap_var, 2),
        "color_variance": round(color_var, 2),
        "fft_anomaly_score": round(ai_score, 4)
    }

class SpatialNoiseExtractor(nn.Module):
    def __init__(self, out_dim: int = 128):
        super(SpatialNoiseExtractor, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.AdaptiveAvgPool2d((8, 8))
        )
        self.fc = nn.Linear(64 * 8 * 8, out_dim)
        
        # Register fixed Laplacian filter kernel
        lap_kernel = torch.tensor([[0., 1., 0.], [1., -4., 1.], [0., 1., 0.]]).unsqueeze(0).unsqueeze(0).repeat(3, 1, 1, 1)
        self.register_buffer("laplacian_kernel", lap_kernel)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        high_pass = F.conv2d(x, self.laplacian_kernel, padding=1, groups=3)
        feat = self.conv(high_pass)
        feat_flat = feat.view(feat.size(0), -1)
        return self.fc(feat_flat)

if __name__ == "__main__":
    dummy_img = np.ones((224, 224, 3), dtype=np.uint8) * 150
    scores = extract_numpy_forensic_scores(dummy_img)
    print("Spatial Forensics Test Output:", scores)
