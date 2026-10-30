"""
Download and Prepare DFDC Video Benchmark Subset (oussema3002/dfdc-faces-mtcnn)
Source: HuggingFace Hub oussema3002/dfdc-faces-mtcnn
Curated 5-frame face sequences grouped strictly by source video ID.
Generates ~/VeriFaceStorage/datasets/dfdc_video_benchmark/ and manifest.
"""

import os
import json
import urllib.request
import pandas as pd
import numpy as np
from PIL import Image

def main():
    target_storage = os.path.expanduser("~/VeriFaceStorage/datasets/dfdc_video_benchmark")
    os.makedirs(target_storage, exist_ok=True)
    
    # 1. Download frames_index.csv
    csv_url = "https://huggingface.co/datasets/oussema3002/dfdc-faces-mtcnn/raw/main/frames_index.csv"
    csv_path = os.path.join(target_storage, "frames_index.csv")
    if not os.path.exists(csv_path):
        print(f"[*] Downloading DFDC index from {csv_url}...")
        req = urllib.request.Request(csv_url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req) as resp, open(csv_path, "wb") as f:
            f.write(resp.read())
            
    print(f"[+] Loaded frames index from {csv_path}")
    df = pd.read_csv(csv_path)
    print(f"[+] Total indexed frames: {len(df)}")
    print(f"[+] Unique videos: {df['video_name'].nunique() if 'video_name' in df.columns else 'N/A'}")
    
    # Identify video column and label column
    vid_col = 'video_id' if 'video_id' in df.columns else df.columns[2]
    label_col = 'label' if 'label' in df.columns else df.columns[1]
    path_col = 'frame_path' if 'frame_path' in df.columns else [c for c in df.columns if 'path' in c.lower()][0]
    
    print(f"[+] Using columns: vid={vid_col}, label={label_col}, path={path_col}")
    
    # Select 20 unique videos (10 real, 10 fake) for a fast, representative video benchmark
    unique_vids = df.groupby([vid_col, label_col]).size().reset_index()
    real_vids = unique_vids[unique_vids[label_col].str.lower().isin(['real', '0'])][vid_col].tolist()[:10]
    fake_vids = unique_vids[unique_vids[label_col].str.lower().isin(['fake', '1'])][vid_col].tolist()[:10]
    
    selected_vids = set(real_vids + fake_vids)
    print(f"[+] Selected {len(selected_vids)} unique videos ({len(real_vids)} Real, {len(fake_vids)} Fake)")
    
    clips_dir = os.path.join(target_storage, "clips")
    os.makedirs(clips_dir, exist_ok=True)
    manifest_entries = []
    
    base_hf_url = "https://huggingface.co/datasets/oussema3002/dfdc-faces-mtcnn/resolve/main"
    
    for vid_id in selected_vids:
        vid_rows = df[df[vid_col] == vid_id].sort_values(by=path_col)
        label_val = vid_rows.iloc[0][label_col]
        is_fake = str(label_val).lower() in ['fake', '1']
        
        frames_tensors = []
        for _, row in vid_rows.iterrows():
            rel_path = row[path_col]
            img_url = f"{base_hf_url}/{rel_path}"
            try:
                req = urllib.request.Request(img_url, headers={"User-Agent": "Mozilla/5.0"})
                with urllib.request.urlopen(req, timeout=10) as resp:
                    img = Image.open(resp).convert("RGB").resize((224, 224))
                    arr = np.array(img, dtype=np.float32) / 255.0 # [224, 224, 3]
                    arr = np.transpose(arr, (2, 0, 1)) # [3, 224, 224]
                    frames_tensors.append(arr)
            except Exception as e:
                continue
                
        if len(frames_tensors) >= 3:
            # Pad or truncate to 5 frames
            while len(frames_tensors) < 5:
                frames_tensors.append(frames_tensors[-1])
            frames_tensors = frames_tensors[:5]
            
            clip_arr = np.stack(frames_tensors, axis=0) # [5, 3, 224, 224]
            clip_filename = f"dfdc_{'fake' if is_fake else 'real'}_{vid_id}.npy"
            clip_path = os.path.join(clips_dir, clip_filename)
            np.save(clip_path, clip_arr)
            
            manifest_entries.append({
                "video_id": vid_id,
                "filepath": clip_path,
                "label": 1 if is_fake else 0,
                "label_name": "fake" if is_fake else "real",
                "num_frames": 5,
                "dataset_origin": "DFDC_MTCNN"
            })
            print(f"  [✓] Saved clip for {vid_id} ({'fake' if is_fake else 'real'}) -> {clip_filename}")
            
    manifest_data = {
        "dataset_name": "DFDC Video Benchmark Subset",
        "total_clips": len(manifest_entries),
        "class_distribution": {
            "real": sum(1 for x in manifest_entries if x["label"] == 0),
            "fake": sum(1 for x in manifest_entries if x["label"] == 1)
        },
        "clips": manifest_entries
    }
    
    manifest_path = "datasets/manifests/dfdc_video_benchmark_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest_data, f, indent=2)
    print(f"[✓] Saved DFDC video manifest to {manifest_path} ({len(manifest_entries)} clips)")

if __name__ == "__main__":
    main()
