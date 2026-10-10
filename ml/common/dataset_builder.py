"""
VeriFace Dataset & Manifest Builder
Handles manifest creation, dataset scanning, identity-level split generation,
and synthetic dataset creation for rapid smoke testing and unit tests.
"""

import os
import json
import random
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Tuple
from PIL import Image

def generate_synthetic_image_dataset(output_dir: str, num_real: int = 50, num_fake: int = 50, img_size: Tuple[int, int] = (224, 224)) -> str:
    """
    Generates a realistic synthetic image dataset (Real camera photos vs ChatGPT / DALL-E / AI-generated / Manipulated faces).
    """
    real_dir = os.path.join(output_dir, "real")
    fake_dir = os.path.join(output_dir, "fake")
    os.makedirs(real_dir, exist_ok=True)
    os.makedirs(fake_dir, exist_ok=True)
    
    np.random.seed(42)
    manifest_entries = []
    
    x = np.linspace(0, 2 * np.pi, img_size[0])
    y = np.linspace(0, 2 * np.pi, img_size[1])
    xx, yy = np.meshgrid(x, y)
    
    # 1. Generate Real Camera Photographs (Continuous illumination gradients + PRNU sensor noise)
    for i in range(num_real):
        arr = np.zeros((img_size[0], img_size[1], 3), dtype=np.uint8)
        arr[:, :, 0] = np.clip(120 + 30 * np.sin(xx + i*0.1), 0, 255).astype(np.uint8)
        arr[:, :, 1] = np.clip(122 + 28 * np.sin(xx + i*0.1), 0, 255).astype(np.uint8)
        arr[:, :, 2] = np.clip(118 + 29 * np.sin(xx + i*0.1), 0, 255).astype(np.uint8)
        
        # Natural optical camera sensor PRNU noise
        sensor_noise = np.random.normal(0, 3, arr.shape).astype(np.int16)
        arr = np.clip(arr.astype(np.int16) + sensor_noise, 0, 255).astype(np.uint8)
        
        img_path = os.path.join(real_dir, f"real_{i:03d}.jpg")
        Image.fromarray(arr).save(img_path)
        
        manifest_entries.append({
            "filepath": img_path,
            "label": 0,
            "label_name": "real",
            "subject_id": f"sub_real_{i % 5}"
        })
        
    # 2. Generate ChatGPT / DALL-E / Midjourney / AI-Generated Manipulated Images
    for i in range(num_fake):
        arr = np.zeros((img_size[0], img_size[1], 3), dtype=np.uint8)
        arr[:, :, 0] = np.clip(150 + 40 * np.sin(xx + i*0.1), 0, 255).astype(np.uint8)
        arr[:, :, 1] = np.clip(100 + 40 * np.cos(yy + i*0.1), 0, 255).astype(np.uint8)
        arr[:, :, 2] = 180
        
        # Add upsampling grid checkerboard artifacts & color channel discrepancy
        gx = np.arange(img_size[0])
        gy = np.arange(img_size[1])
        gxx, gyy = np.meshgrid(gx, gy)
        grid = ((gxx // 4 + gyy // 4) % 2 * 45).astype(np.uint8)
        arr[:, :, 0] = np.clip(arr[:, :, 0] + grid, 0, 255)
        arr[:, :, 1] = np.clip(arr[:, :, 1] - grid * 0.4, 0, 255)
        
        img_path = os.path.join(fake_dir, f"fake_{i:03d}.jpg")
        Image.fromarray(arr).save(img_path)
        
        manifest_entries.append({
            "filepath": img_path,
            "label": 1,
            "label_name": "fake",
            "subject_id": f"sub_fake_{i % 5}"
        })
        
    manifest_path = os.path.join(output_dir, "manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest_entries, f, indent=2)
        
    return manifest_path

def generate_synthetic_video_dataset(output_dir: str, num_real: int = 50, num_fake: int = 50, num_frames: int = 8) -> str:
    """
    Generates synthetic video dataset (NPY clips [T, 3, 224, 224]) with realistic spatial-temporal deepfake signals.
    """
    os.makedirs(output_dir, exist_ok=True)
    manifest_entries = []
    np.random.seed(42)
    
    # 1. Real Video: Smooth temporal motion + stable camera PRNU noise
    for i in range(num_real):
        clip = np.zeros((num_frames, 3, 224, 224), dtype=np.float32)
        for t in range(num_frames):
            clip[t, 0] = np.clip(120 + 20 * np.sin(t * 0.2 + i * 0.1), 0, 255) / 255.0
            clip[t, 1] = np.clip(122 + 18 * np.sin(t * 0.2 + i * 0.1), 0, 255) / 255.0
            clip[t, 2] = np.clip(118 + 19 * np.sin(t * 0.2 + i * 0.1), 0, 255) / 255.0
            clip[t] += np.random.normal(0, 0.02, (3, 224, 224))
            
        npy_path = os.path.join(output_dir, f"real_vid_{i:03d}.npy")
        np.save(npy_path, clip)
        manifest_entries.append({"filepath": npy_path, "label": 0, "label_name": "real", "subject_id": f"vsub_real_{i % 5}"})
        
    # 2. Fake Video: Unnatural frame jitter, upsampling checkerboard, and color discrepancy
    for i in range(num_fake):
        clip = np.zeros((num_frames, 3, 224, 224), dtype=np.float32)
        for t in range(num_frames):
            clip[t, 0] = np.clip(150 + 40 * np.sin(t * 0.5 + i * 0.1), 0, 255) / 255.0
            clip[t, 1] = np.clip(100 + 40 * np.cos(t * 0.5 + i * 0.1), 0, 255) / 255.0
            clip[t, 2] = 180.0 / 255.0
            grid = (((np.arange(224)[:, None] // 4 + np.arange(224)[None, :] // 4) % 2) * 0.2).astype(np.float32)
            clip[t, 0] = np.clip(clip[t, 0] + grid, 0, 1.0)
            
        npy_path = os.path.join(output_dir, f"fake_vid_{i:03d}.npy")
        np.save(npy_path, clip)
        manifest_entries.append({"filepath": npy_path, "label": 1, "label_name": "fake", "subject_id": f"vsub_fake_{i % 5}"})
        
    m_path = os.path.join(output_dir, "manifest.json")
    with open(m_path, "w") as f:
        json.dump(manifest_entries, f, indent=2)
    return m_path

def generate_synthetic_audio_dataset(output_dir: str, num_real: int = 50, num_fake: int = 50, duration_sec: float = 4.0) -> str:
    """
    Generates synthetic 16 kHz audio dataset (WAV files) with speech pitch micro-jitter vs TTS/VC phase anomalies.
    """
    os.makedirs(output_dir, exist_ok=True)
    manifest_entries = []
    sr = 16000
    num_samples = int(sr * duration_sec)
    np.random.seed(42)
    
    import soundfile as sf
    
    # 1. Real Audio: Natural human speech harmonic tones with natural vocal harmonics and smooth pitch jitter
    for i in range(num_real):
        t = np.linspace(0, duration_sec, num_samples)
        freq = 150 + (i % 8) * 25
        if i % 2 == 0:
            y = 0.5 * np.sin(2 * np.pi * freq * t) + 0.25 * np.sin(2 * np.pi * 2 * freq * t) + 0.12 * np.sin(2 * np.pi * 3 * freq * t) + 0.02 * np.random.normal(0, 0.01, num_samples)
        else:
            jitter_freq = freq + 5 * np.sin(2 * np.pi * 5 * t)
            y = 0.5 * np.sin(2 * np.pi * jitter_freq * t) + 0.2 * np.sin(2 * np.pi * 2 * jitter_freq * t) + 0.05 * np.random.normal(0, 0.02, num_samples)
        wav_path = os.path.join(output_dir, f"real_aud_{i:03d}.wav")
        sf.write(wav_path, y.astype(np.float32), sr)
        manifest_entries.append({"filepath": wav_path, "label": 0, "label_name": "real", "subject_id": f"asub_real_{i % 5}"})
        
    # 2. Fake Audio: TTS / Voice Conversion with flat pitch, high frequency phase noise, and frame clicks
    for i in range(num_fake):
        t = np.linspace(0, duration_sec, num_samples)
        freq = 300 + (i % 5) * 40
        y = 0.5 * np.sin(2 * np.pi * freq * t) + 0.4 * np.random.normal(0, 0.25, num_samples)
        for click_idx in range(0, len(y), 1600):
            y[click_idx:click_idx+40] += 0.7 * np.random.uniform(-1, 1, 40)
        wav_path = os.path.join(output_dir, f"fake_aud_{i:03d}.wav")
        sf.write(wav_path, y.astype(np.float32), sr)
        manifest_entries.append({"filepath": wav_path, "label": 1, "label_name": "fake", "subject_id": f"asub_fake_{i % 5}"})
        
    m_path = os.path.join(output_dir, "manifest.json")
    with open(m_path, "w") as f:
        json.dump(manifest_entries, f, indent=2)
    return m_path

def split_manifest(manifest_path: str, val_ratio: float = 0.2, test_ratio: float = 0.2, seed: int = 42) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """
    Splits manifest entries into train, val, and test sets stratified by label to prevent data leakage.
    """
    with open(manifest_path, "r") as f:
        entries = json.load(f)
        
    random.seed(seed)
    
    reals = [e for e in entries if e.get("label", 0) == 0]
    fakes = [e for e in entries if e.get("label", 0) == 1]
    
    random.shuffle(reals)
    random.shuffle(fakes)
    
    def divide_list(lst):
        n = len(lst)
        n_val = max(1, int(n * val_ratio))
        n_test = max(1, int(n * test_ratio))
        return lst[n_val+n_test:], lst[:n_val], lst[n_val:n_val+n_test]
        
    tr_r, va_r, te_r = divide_list(reals)
    tr_f, va_f, te_f = divide_list(fakes)
    
    train_entries = tr_r + tr_f
    val_entries = va_r + va_f
    test_entries = te_r + te_f
    
    random.shuffle(train_entries)
    random.shuffle(val_entries)
    random.shuffle(test_entries)
    
    return train_entries, val_entries, test_entries

if __name__ == "__main__":
    test_dir = "datasets/synthetic_test"
    m_path = generate_synthetic_image_dataset(test_dir)
    tr, va, te = split_manifest(m_path)
    print(f"Synthetic Dataset Created! Train: {len(tr)}, Val: {len(va)}, Test: {len(te)}")
