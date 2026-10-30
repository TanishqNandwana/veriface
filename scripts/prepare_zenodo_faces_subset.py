"""
Process and Manifest Zenodo Real vs Fake Faces Dataset (FFHQ vs StyleGAN)
Extracts a clean, balanced independent evaluation set from ~/VeriFaceStorage/datasets/real_vs_fake_faces
and generates datasets/manifests/zenodo_faces_benchmark_manifest.json.
"""

import os
import json
import hashlib
from PIL import Image

def calculate_sha256(filepath: str) -> str:
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()

def main():
    base_dir = os.path.expanduser("~/VeriFaceStorage/datasets/real_vs_fake_faces")
    if not os.path.exists(base_dir):
        print(f"[!] Directory not found: {base_dir}")
        return

    print(f"[*] Scanning {base_dir} for face images...")
    
    real_images, fake_images = [], []
    
    for root, dirs, files in os.walk(base_dir):
        for f in files:
            if f.lower().endswith(('.jpg', '.jpeg', '.png')):
                full_path = os.path.join(root, f)
                p_lower = full_path.lower()
                if "real" in p_lower and "fake" not in os.path.basename(root).lower():
                    real_images.append(full_path)
                elif "fake" in p_lower:
                    fake_images.append(full_path)
                    
    print(f"[+] Found {len(real_images)} Real and {len(fake_images)} Fake images in extracted directory")
    
    # Select 250 Real and 250 Fake for a focused, ultra-fast 500-sample test benchmark
    n_sample = min(250, min(len(real_images), len(fake_images)))
    selected_real = real_images[:n_sample]
    selected_fake = fake_images[:n_sample]
    
    manifest_entries = []
    
    for path in selected_real:
        manifest_entries.append({
            "filepath": path,
            "label": 0,
            "label_name": "real",
            "sha256": calculate_sha256(path),
            "origin_dataset": "Zenodo_FFHQ_Real"
        })
        
    for path in selected_fake:
        manifest_entries.append({
            "filepath": path,
            "label": 1,
            "label_name": "fake",
            "sha256": calculate_sha256(path),
            "origin_dataset": "Zenodo_StyleGAN_Fake"
        })
        
    manifest_data = {
        "dataset_name": "Zenodo Real vs Fake Faces (FFHQ vs StyleGAN) Benchmark",
        "total_samples": len(manifest_entries),
        "class_distribution": {
            "real": len(selected_real),
            "fake": len(selected_fake)
        },
        "samples": manifest_entries
    }
    
    manifest_path = "datasets/manifests/zenodo_faces_benchmark_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest_data, f, indent=2)
        
    print(f"[✓] Created Zenodo faces benchmark manifest: {manifest_path} ({len(manifest_entries)} samples)")

if __name__ == "__main__":
    main()
