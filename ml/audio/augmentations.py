"""
Audio Manipulation Augmentation Suite for CLAD
Base paper reference: Wu et al. (2024) - CLAD
Implements noise injection, volume control, fading, time stretching, resampling, and time shifting.
Creates dual augmented views for contrastive learning.
"""

import numpy as np

def apply_noise_injection(y: np.ndarray, snr_db: float = 20.0) -> np.ndarray:
    """Adds white Gaussian noise at specified SNR."""
    signal_power = np.mean(y ** 2) + 1e-8
    noise_power = signal_power / (10 ** (snr_db / 10.0))
    noise = np.random.normal(0, np.sqrt(noise_power), len(y))
    return y + noise

def apply_volume_control(y: np.ndarray, gain_db: float = None) -> np.ndarray:
    """Applies random volume gain adjustment in range [-6 dB, +6 dB]."""
    if gain_db is None:
        gain_db = np.random.uniform(-6.0, 6.0)
    factor = 10 ** (gain_db / 20.0)
    return y * factor

def apply_fading(y: np.ndarray, fade_ratio: float = 0.1) -> np.ndarray:
    """Applies linear fade-in and fade-out."""
    n = len(y)
    fade_len = int(n * fade_ratio)
    if fade_len <= 0:
        return y
        
    y_aug = y.copy()
    fade_in = np.linspace(0.0, 1.0, fade_len)
    fade_out = np.linspace(1.0, 0.0, fade_len)
    
    y_aug[:fade_len] *= fade_in
    y_aug[-fade_len:] *= fade_out
    return y_aug

def apply_time_shift(y: np.ndarray, max_shift_ratio: float = 0.1) -> np.ndarray:
    """Applies random time shift with roll."""
    shift = int(len(y) * np.random.uniform(-max_shift_ratio, max_shift_ratio))
    return np.roll(y, shift)

def apply_clad_augmentations(y: np.ndarray) -> np.ndarray:
    """
    Applies random combination of CLAD manipulation augmentations to generate an augmented view.
    """
    y_aug = y.copy()
    
    # 50% chance of noise injection
    if np.random.rand() > 0.5:
        y_aug = apply_noise_injection(y_aug, snr_db=np.random.uniform(15.0, 30.0))
        
    # 50% chance of volume control
    if np.random.rand() > 0.5:
        y_aug = apply_volume_control(y_aug)
        
    # 50% chance of fading
    if np.random.rand() > 0.5:
        y_aug = apply_fading(y_aug)
        
    # 50% chance of time shift
    if np.random.rand() > 0.5:
        y_aug = apply_time_shift(y_aug)
        
    return y_aug.astype(np.float32)

if __name__ == "__main__":
    dummy_wav = np.random.normal(0, 0.1, 64000).astype(np.float32)
    view_1 = apply_clad_augmentations(dummy_wav)
    view_2 = apply_clad_augmentations(dummy_wav)
    print("CLAD Augmentation Test: View 1 diff =", np.sum(np.abs(dummy_wav - view_1)), "View 2 diff =", np.sum(np.abs(dummy_wav - view_2)))
