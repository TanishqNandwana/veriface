"""
Download and Prepare FFHQ vs StyleGAN Benchmark Subset
Source: TheKernel01/140k-Real-and-Fake-Faces (HuggingFace Hub)
Downloads validation/0.parquet (286 MB), extracts balanced Real (FFHQ) and Fake (StyleGAN)
face crops into ~/VeriFaceStorage/datasets/ffhq_stylegan_benchmark/,
and generates datasets/manifests/ffhq_stylegan_manifest.json.
"""

import os
import io
import json
import hashlib
import urllib.request
import pandas as pd
from PIL import Image

def calculate_sha256(filepath: str) -> str:
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()

def main():
    target_storage = os.path.expanduser("~/VeriFaceStorage/datasets/ffhq_stylegan_benchmark")
    os.makedirs(target_storage, exist_ok=True)
    
    parquet_path = os.path.join(target_storage, "val_0.parquet")
    parquet_url = "https://huggingface.co/api/datasets/TheKernel01/140k-Real-and-Fake-Faces/parquet/default/validation/0.parquet"
    
    if not os.path.exists(parquet_path) or os.path.getsize(parquet_path) < 100000000:
        print(f"[*] Downloading validation parquet from {parquet_url}...")
        req = urllib.request.Request(parquet_url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req) as resp, open(parquet_path, "wb") as f:
            while chunk := resp.read(1024 * 1024):
                f.write(chunk)
        print(f"[✓] Downloaded {os.path.getsize(parquet_path) / (1024*1024):.1f} MB")
        
    print("[*] Reading parquet...")
    df = pd.read_parquet(parquet_path)
    print(f"[+] Total rows in parquet: {len(df)}")
    print(f"[+] Columns: {list(df.columns)}")
    
    label_col = 'label' if 'label' in df.columns else df.columns[1]
    print(f"[+] Label counts:\n{df[label_col].value_counts()}")
    
    real_dir = os.path.join(target_storage, "real")
    fake_dir = os.path.join(target_storage, "fake")
    os.makedirs(real_dir, exist_ok=True)
    os.makedirs(fake_dir, exist_ok=True)
    
    n_per_class = 100
    real_count, fake_count = 0, 0
    manifest_entries = []
    
    for idx, row in df.iterrows():
        veriface_label = int(row[label_col]) # 0=Real (FFHQ), 1=Fake (StyleGAN)
        
        if veriface_label == 0 and real_count >= n_per_class:
            continue
        if veriface_label == 1 and fake_count >= n_per_class:
            continue
            
        img_bytes = row["image.bytes"] if "image.bytes" in row else (row["image"]["bytes"] if "image" in row and isinstance(row["image"], dict) else None)
        if not img_bytes:
            continue
            
        try:
            img = Image.open(io.BytesIO(img_bytes)).convert("RGB").resize((224, 224))
                
            fname = f"face_{'real' if veriface_label == 0 else 'fake'}_{idx:05d}.jpg"
            dest_dir = real_dir if veriface_label == 0 else fake_dir
            dest_path = os.path.join(dest_dir, fname)
            img.save(dest_path, "JPEG", quality=95)
            
            manifest_entries.append({
                "sample_id": fname,
                "filepath": dest_path,
                "label": veriface_label,
                "label_name": "real" if veriface_label == 0 else "fake",
                "sha256": calculate_sha256(dest_path),
                "dataset_origin": "140k_FFHQ_StyleGAN"
            })
            
            if veriface_label == 0:
                real_count += 1
            else:
                fake_count += 1
                
            if real_count >= n_per_class and fake_count >= n_per_class:
                break
        except Exception:
            continue
            
    print(f"[✓] Extracted: {real_count} Real and {fake_count} Fake faces")
    
    manifest_data = {
        "dataset_name": "FFHQ vs StyleGAN Independent Evaluation Benchmark",
        "total_samples": len(manifest_entries),
        "class_distribution": {"real": real_count, "fake": fake_count},
        "samples": manifest_entries
    }
    
    manifest_path = "datasets/manifests/ffhq_stylegan_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest_data, f, indent=2)
    print(f"[✓] Saved manifest to {manifest_path}")

if __name__ == "__main__":
    main()
