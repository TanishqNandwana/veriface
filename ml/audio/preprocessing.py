"""
Audio Preprocessing Utility for VeriFace
Resamples input audio files to 16 kHz mono, pads/crops to fixed target length (e.g. 4 seconds = 64,000 samples),
and computes log-Mel spectrograms or raw 1D waveform tensors.
"""

import os
import numpy as np
import torch
import librosa
import soundfile as sf
from typing import Tuple

TARGET_SR = 16000
TARGET_DURATION_SEC = 4.0
TARGET_SAMPLES = int(TARGET_SR * TARGET_DURATION_SEC) # 64,000 samples

def load_and_preprocess_audio(audio_path_or_array, sr: int = TARGET_SR, target_samples: int = TARGET_SAMPLES) -> Tuple[np.ndarray, int]:
    """
    Loads audio file or 1D array, resamples to 16 kHz mono, and pads/crops to target_samples length.
    Returns (processed_waveform [target_samples], sr).
    """
    if isinstance(audio_path_or_array, str):
        if not os.path.exists(audio_path_or_array):
            raise FileNotFoundError(f"Audio file not found: {audio_path_or_array}")
        y, orig_sr = librosa.load(audio_path_or_array, sr=sr, mono=True)
    else:
        y = np.array(audio_path_or_array, dtype=np.float32)
        if y.ndim > 1:
            y = np.mean(y, axis=0) # Convert to mono
            
    # Pad or crop waveform to target_samples
    if len(y) < target_samples:
        pad_len = target_samples - len(y)
        y = np.pad(y, (0, pad_len), mode="constant")
    else:
        y = y[:target_samples]
        
    # Peak normalization
    max_val = np.max(np.abs(y))
    if max_val > 0:
        y = y / max_val
        
    return y.astype(np.float32), sr

def compute_log_mel_spectrogram(y: np.ndarray, sr: int = TARGET_SR, n_mels: int = 80, n_fft: int = 1024, hop_length: int = 256) -> torch.Tensor:
    """
    Computes log-Mel spectrogram tensor [1, n_mels, time_steps] from 1D waveform.
    """
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels, n_fft=n_fft, hop_length=hop_length)
    log_mel = librosa.power_to_db(mel, ref=np.max)
    
    # Normalize log-mel to zero mean, unit variance
    mean = np.mean(log_mel)
    std = np.std(log_mel) + 1e-8
    norm_mel = (log_mel - mean) / std
    
    tensor = torch.tensor(norm_mel, dtype=torch.float32).unsqueeze(0) # [1, 80, T]
    return tensor

if __name__ == "__main__":
    dummy_wav = np.random.normal(0, 0.1, 80000)
    y, sr = load_and_preprocess_audio(dummy_wav)
    mel_tensor = compute_log_mel_spectrogram(y)
    print("Audio Preprocessing Test: Waveform shape =", y.shape, "Log-Mel tensor shape =", mel_tensor.shape)
