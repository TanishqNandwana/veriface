"""
Download and Prepare ASVspoof 2019 Logical Access (LA) Benchmark Subset
Source: SpeechAntiSpoofingBenchmarks/ASVspoof2019_LA (HuggingFace Hub)
Downloads 0.parquet, parses audio waveforms, applies VeriFace standard 16kHz preprocessing,
and builds an independent held-out evaluation manifest.
"""

import os
import io
import json
import hashlib
import urllib.request
import soundfile as sf
import pandas as pd
import numpy as np

def calculate_sha256(filepath: str) -> str:
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()

def main():
    target_storage = os.path.expanduser("~/VeriFaceStorage/datasets/asvspoof2019_la")
    os.makedirs(target_storage, exist_ok=True)
    
    parquet_path = os.path.join(target_storage, "0.parquet")
    parquet_url = "https://huggingface.co/api/datasets/SpeechAntiSpoofingBenchmarks/ASVspoof2019_LA/parquet/default/test/0.parquet"
    
    # 1. Download parquet if not already present
    if not os.path.exists(parquet_path) or os.path.getsize(parquet_path) < 100000000:
        print(f"[*] Downloading ASVspoof 2019 LA parquet from: {parquet_url}")
        req = urllib.request.Request(parquet_url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req) as response, open(parquet_path, "wb") as out_file:
            # Handle redirect or stream
            shutil_copy = True
            while chunk := response.read(1024 * 1024):
                out_file.write(chunk)
        print(f"[✓] Downloaded ASVspoof parquet ({os.path.getsize(parquet_path) / (1024*1024):.1f} MB)")
    else:
        print(f"[+] Parquet already exists at {parquet_path}")

    # 2. Read parquet and extract balanced samples
    print("[*] Reading parquet...")
    df = pd.read_parquet(parquet_path)
    print(f"[+] Total rows in parquet: {len(df)}")
    print(f"[+] Columns: {list(df.columns)}")
    
    # Find label column (often 'key', 'label', 'spoof', etc.)
    label_col = None
    for cand in ['key', 'label', 'target', 'class']:
        if cand in df.columns:
            label_col = cand
            break
            
    print(f"[+] Using label column: '{label_col}'")
    if label_col:
        print(f"[+] Label counts:\n{df[label_col].value_counts()}")
        
    out_dir = os.path.expanduser("~/VeriFaceStorage/datasets/asvspoof_benchmark")
    real_dir = os.path.join(out_dir, "real")
    fake_dir = os.path.join(out_dir, "fake")
    os.makedirs(real_dir, exist_ok=True)
    os.makedirs(fake_dir, exist_ok=True)
    
    # Extract 50 bonafide and 50 spoof
    n_per_class = 50
    real_count, fake_count = 0, 0
    manifest_entries = []
    
    for idx, row in df.iterrows():
        raw_label = str(row[label_col]).strip()
        is_bonafide = raw_label in ["0", "bonafide", "real"]
        
        if is_bonafide and real_count >= n_per_class:
            continue
        if not is_bonafide and fake_count >= n_per_class:
            continue
            
        audio_bytes = row["audio.bytes"] if "audio.bytes" in row else (row["audio"]["bytes"] if "audio" in row and isinstance(row["audio"], dict) else None)
        if not audio_bytes:
            continue
            
        # Parse audio
        try:
            bio = io.BytesIO(audio_bytes)
            y, sr = sf.read(bio)
                
            # Convert to mono if needed
            if y.ndim > 1:
                y = np.mean(y, axis=1)
                
            # Standardize 4.0s @ 16kHz
            target_sr = 16000
            if sr != target_sr:
                import librosa
                y = librosa.resample(y, orig_sr=sr, target_sr=target_sr)
                sr = target_sr
                
            target_samples = 64000
            if len(y) < target_samples:
                y = np.pad(y, (0, target_samples - len(y)))
            else:
                y = y[:target_samples]
                
            # Peak normalize
            max_val = np.max(np.abs(y))
            if max_val > 0:
                y = y / max_val * 0.95
                
            sample_id = f"asv2019_{'real' if is_bonafide else 'spoof'}_{idx:05d}"
            save_path = os.path.join(real_dir if is_bonafide else fake_dir, f"{sample_id}.wav")
            sf.write(save_path, y, target_sr)
            
            manifest_entries.append({
                "sample_id": sample_id,
                "filepath": save_path,
                "label": 0 if is_bonafide else 1,
                "label_name": "real" if is_bonafide else "fake",
                "sha256": calculate_sha256(save_path),
                "dataset_origin": "ASVspoof_2019_LA",
                "speaker_id": str(row.get("speaker_id", "unknown")),
                "system_id": str(row.get("system_id", "unknown"))
            })
            
            if is_bonafide:
                real_count += 1
            else:
                fake_count += 1
                
            if real_count >= n_per_class and fake_count >= n_per_class:
                break
        except Exception as e:
            continue
            
    print(f"[✓] Extracted: {real_count} Real and {fake_count} Spoof audio clips")
    
    # Save manifest
    manifest_data = {
        "dataset_name": "ASVspoof 2019 Logical Access Benchmark Subset",
        "total_samples": len(manifest_entries),
        "class_distribution": {"real": real_count, "fake": fake_count},
        "samples": manifest_entries
    }
    
    manifest_path = "datasets/manifests/asvspoof_2019_la_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest_data, f, indent=2)
    print(f"[✓] Saved manifest to {manifest_path}")

if __name__ == "__main__":
    main()
