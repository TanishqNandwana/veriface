"""
Spatial-Temporal Video Deepfake Forensic Feature Extractor for VeriFace
Extracts temporal frame-to-frame difference variance, cross-frame PRNU noise coherence,
and spatial-temporal Fourier upsampling grid artifacts.
"""

import numpy as np
import cv2

def compute_temporal_flicker_variance(frames: list) -> float:
    """
    Computes inter-frame temporal difference variance across frame sequence.
    AI video generators and deepfake face swaps exhibit unnatural frame-to-frame boundary jitter and flickering.
    """
    if len(frames) < 2:
        return 0.0
        
    diffs = []
    for i in range(1, len(frames)):
        f1 = frames[i-1].astype(np.float32)
        f2 = frames[i].astype(np.float32)
        diff = np.mean((f2 - f1) ** 2)
        diffs.append(diff)
        
    return float(np.var(diffs)) if len(diffs) > 1 else float(diffs[0])

def compute_video_prnu_coherence(frames: list) -> float:
    """
    Computes cross-frame PRNU noise residual coherence.
    Natural video maintains consistent camera sensor noise across consecutive frames.
    Synthetic deepfake video exhibits temporal noise destruction and high Laplacian residual variance fluctuations.
    """
    if len(frames) < 2:
        return 1.0
        
    lap_vars = []
    for frame in frames:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
        lap = cv2.Laplacian(gray, cv2.CV_64F)
        lap_vars.append(np.var(lap))
        
    # Standard deviation of Laplacian noise variance across frames normalized by mean
    mean_lap = np.mean(lap_vars) + 1e-6
    std_lap = np.std(lap_vars)
    rel_fluctuation = float(std_lap / mean_lap)
    return rel_fluctuation

def compute_video_fft_grid_variance(frames: list) -> float:
    """
    Computes average 2D Fourier transform azimuthal directional variance across video frames.
    Upsampling artifacts in generative video models produce prominent high-frequency grid peaks.
    """
    if len(frames) == 0:
        return 0.0
        
    azimuthal_vars = []
    for frame in frames:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
        h, w = gray.shape
        f = np.fft.fft2(gray)
        fshift = np.fft.fftshift(f)
        mag = np.abs(fshift) + 1e-8
        
        cy, cx = h // 2, w // 2
        r_max = min(h, w) / 2.0
        y, x = np.ogrid[:h, :w]
        dist = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
        high_mask = (dist >= 0.2 * r_max) & (dist <= 0.8 * r_max)
        angle = np.arctan2(y - cy, x - cx)
        
        sector_energies = []
        for deg in range(0, 180, 20):
            rad = np.radians(deg)
            mask = high_mask & (np.abs(angle - rad) < np.radians(10))
            if np.sum(mask) > 0:
                sector_energies.append(float(np.mean(mag[mask])))
                
        if len(sector_energies) > 0:
            azimuthal_vars.append(float(np.std(sector_energies) / (np.mean(sector_energies) + 1e-8)))
        else:
            azimuthal_vars.append(0.0)
            
    return float(np.mean(azimuthal_vars))

def extract_video_forensic_stats(frames: list) -> dict:
    """
    Aggregates spatial-temporal forensic metrics across video frames.
    """
    flicker_var = compute_temporal_flicker_variance(frames)
    prnu_fluctuation = compute_video_prnu_coherence(frames)
    fft_grid_var = compute_video_fft_grid_variance(frames)
    
    # Calculate anomaly score (0.0 = Real Camera, 1.0 = Deepfake / Synthetic Video)
    # 1. PRNU noise fluctuation: natural camera < 0.04, synthetic deepfake > 0.10
    prnu_score = min(1.0, max(0.0, (prnu_fluctuation - 0.03) / 0.10))
    
    # 2. 2D FFT azimuthal grid variance: natural < 0.05, generative upsampling > 0.25
    fft_score = min(1.0, max(0.0, (fft_grid_var - 0.04) / 0.25))
    
    # 3. Inter-frame difference flicker: natural continuous motion has stable diff variance
    flicker_score = min(1.0, max(0.0, (flicker_var - 40.0) / 120.0)) if flicker_var > 40.0 else 0.0
    
    # Multi-cue forensic anomaly ensemble
    combined_anomaly_score = float(0.50 * prnu_score + 0.35 * fft_score + 0.15 * flicker_score)
    combined_anomaly_score = max(0.0, min(1.0, combined_anomaly_score))
    
    return {
        "temporal_flicker_variance": round(flicker_var, 4),
        "prnu_fluctuation_ratio": round(prnu_fluctuation, 4),
        "fft_grid_variance": round(fft_grid_var, 4),
        "video_anomaly_score": round(combined_anomaly_score, 4)
    }

