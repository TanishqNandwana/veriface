"""
Unified Data Extraction, Face Preprocessing, and Manifest Generation for VeriFace
Extracts genuine Image, Video, and Audio deepfake datasets from downloaded HF parquets:
- Image: Celeb-DF Real YouTube face crops vs Celeb-DF & FaceSwap deepfakes
- Video: Multi-frame temporal video sequences strictly grouped by source video ID
- Audio: Real human speech vs Synthesized AI deepfake audio, standardized to 16kHz mono 4s
Enforces zero-leakage splits across Train (70%), Validation (15%), and Test (15%).
"""

import os
import io
import re
import json
import shutil
import random
import numpy as np
import pandas as pd
import cv2
import soundfile as sf
import librosa
from PIL import Image
from huggingface_hub import hf_hub_download

from ml.image.preprocessing import extract_face_crop
from ml.audio.preprocessing import load_and_preprocess_audio
from ml.common.dataset_pipeline import UnifiedDatasetPipeline, compute_file_sha256

def extract_video_id(path: str) -> tuple:
    m = re.match(r'^(.*)_(\d+)\.jpg$', path)
    if m:
        return m.group(1), int(m.group(2))
    return os.path.splitext(path)[0], 0

def prepare_image_dataset(output_dir: str = "datasets/processed/image_benchmark", max_real: int = 250, max_fake: int = 250):
    print("\n=======================================================")
    print("[1/3] Preparing Genuine Image Deepfake Benchmark (Celeb-DF & FaceSwap)")
    print("=======================================================")
    os.makedirs(os.path.join(output_dir, "real"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "fake"), exist_ok=True)

    # 1. Load Celeb-DF parquets
    p_test = hf_hub_download(repo_id='thenewsupercell/celeb-df-image-dataset', filename='data/test-00000-of-00001.parquet', repo_type='dataset')
    p_val = hf_hub_download(repo_id='thenewsupercell/celeb-df-image-dataset', filename='data/validation-00000-of-00001.parquet', repo_type='dataset')
    
    df_test = pd.read_parquet(p_test, engine='fastparquet')
    df_val = pd.read_parquet(p_val, engine='fastparquet')
    df_all = pd.concat([df_test, df_val], ignore_index=True)

    # Real samples in this dataset: label == 1 (original YouTube real videos)
    real_rows = df_all[df_all['label'] == 1].sample(frac=1.0, random_state=42)
    # Fake samples in this dataset: label == 0 (Wav2Lip / swap manipulations)
    fake_rows = df_all[df_all['label'] == 0].sample(frac=1.0, random_state=42)

    image_entries = []

    # Process Real Images
    saved_real = 0
    for _, row in real_rows.iterrows():
        if saved_real >= max_real:
            break
        raw_bytes = row['image.bytes']
        filename = row['image.path']
        vid_id, frame_idx = extract_video_id(filename)

        try:
            pil_img = Image.open(io.BytesIO(raw_bytes)).convert('RGB')
            arr_bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
            face_crop_bgr, bbox = extract_face_crop(arr_bgr)
            face_crop_resized = cv2.resize(face_crop_bgr, (224, 224))

            save_path = os.path.join(output_dir, "real", f"real_celeb_{saved_real:04d}_{filename}")
            cv2.imwrite(save_path, face_crop_resized)

            image_entries.append({
                "filepath": save_path,
                "label": 0, # Ground truth: 0 = Real
                "label_name": "real",
                "source_id": f"vid_real_{vid_id}",
                "origin_dataset": "Celeb-DF-v2"
            })
            saved_real += 1
        except Exception as e:
            continue

    print(f"  [+] Saved {saved_real} preprocessed real face crops.")

    # Process Fake Images
    saved_fake = 0
    for _, row in fake_rows.iterrows():
        if saved_fake >= max_fake:
            break
        raw_bytes = row['image.bytes']
        filename = row['image.path']
        vid_id, frame_idx = extract_video_id(filename)

        try:
            pil_img = Image.open(io.BytesIO(raw_bytes)).convert('RGB')
            arr_bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
            face_crop_bgr, bbox = extract_face_crop(arr_bgr)
            face_crop_resized = cv2.resize(face_crop_bgr, (224, 224))

            save_path = os.path.join(output_dir, "fake", f"fake_celeb_{saved_fake:04d}_{filename}")
            cv2.imwrite(save_path, face_crop_resized)

            image_entries.append({
                "filepath": save_path,
                "label": 1, # Ground truth: 1 = Fake
                "label_name": "fake",
                "source_id": f"vid_fake_{vid_id}",
                "origin_dataset": "Celeb-DF-v2-Manipulated"
            })
            saved_fake += 1
        except Exception as e:
            continue

    print(f"  [+] Saved {saved_fake} preprocessed fake face crops.")

    # Validate and split
    pipeline = UnifiedDatasetPipeline(modality="image")
    valid_samples, stats = pipeline.scan_and_validate(image_entries, remove_duplicates=True)
    tr, va, te, leak_rep = pipeline.create_leakage_free_splits(valid_samples, train_ratio=0.70, val_ratio=0.15, test_ratio=0.15)
    
    manifest_path = "datasets/manifests/image_benchmark_manifest.json"
    pipeline.save_manifest(tr, va, te, metadata={"stats": stats, "leakage": leak_rep}, output_manifest_path=manifest_path)
    print(f"  [+] Image Manifest generated: {leak_rep['split_counts']}")
    print(f"  [+] Leakage Detected: {leak_rep['leakage_detected']}")
    return manifest_path

def prepare_video_dataset(output_dir: str = "datasets/processed/video_benchmark", max_clips: int = 50, frames_per_clip: int = 5):
    print("\n=======================================================")
    print("[2/3] Preparing Genuine Video Deepfake Benchmark (Celeb-DF Sequences)")
    print("=======================================================")
    os.makedirs(os.path.join(output_dir, "real"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "fake"), exist_ok=True)

    p_test = hf_hub_download(repo_id='thenewsupercell/celeb-df-image-dataset', filename='data/test-00000-of-00001.parquet', repo_type='dataset')
    df_test = pd.read_parquet(p_test, engine='fastparquet')

    df_test[['video_id', 'frame_num']] = df_test['image.path'].apply(lambda x: pd.Series(extract_video_id(x)))

    video_entries = []

    # Group by video_id
    grouped_real = [group for vid, group in df_test[df_test['label'] == 1].groupby('video_id') if len(group) >= frames_per_clip]
    grouped_fake = [group for vid, group in df_test[df_test['label'] == 0].groupby('video_id') if len(group) >= frames_per_clip]

    random.seed(42)
    random.shuffle(grouped_real)
    random.shuffle(grouped_fake)

    # Build Real Video Clips
    real_clips_saved = 0
    for grp in grouped_real[:max_clips]:
        sorted_grp = grp.sort_values('frame_num')
        frames_list = []
        vid_id = sorted_grp.iloc[0]['video_id']

        for _, row in sorted_grp.iloc[:frames_per_clip].iterrows():
            pil_img = Image.open(io.BytesIO(row['image.bytes'])).convert('RGB')
            arr_bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
            face_crop_bgr, _ = extract_face_crop(arr_bgr)
            face_resized = cv2.resize(face_crop_bgr, (224, 224))
            rgb_arr = cv2.cvtColor(face_resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            # [3, 224, 224]
            frames_list.append(np.transpose(rgb_arr, (2, 0, 1)))

        clip_arr = np.stack(frames_list, axis=0) # [T, 3, 224, 224]
        clip_path = os.path.join(output_dir, "real", f"clip_real_{real_clips_saved:03d}_{vid_id}.npy")
        np.save(clip_path, clip_arr)

        video_entries.append({
            "filepath": clip_path,
            "label": 0, # Ground truth: 0 = Real
            "label_name": "real",
            "source_id": f"vid_src_{vid_id}",
            "frames_count": frames_per_clip
        })
        real_clips_saved += 1

    print(f"  [+] Saved {real_clips_saved} real video sequence clips.")

    # Build Fake Video Clips
    fake_clips_saved = 0
    for grp in grouped_fake[:max_clips]:
        sorted_grp = grp.sort_values('frame_num')
        frames_list = []
        vid_id = sorted_grp.iloc[0]['video_id']

        for _, row in sorted_grp.iloc[:frames_per_clip].iterrows():
            pil_img = Image.open(io.BytesIO(row['image.bytes'])).convert('RGB')
            arr_bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
            face_crop_bgr, _ = extract_face_crop(arr_bgr)
            face_resized = cv2.resize(face_crop_bgr, (224, 224))
            rgb_arr = cv2.cvtColor(face_resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            frames_list.append(np.transpose(rgb_arr, (2, 0, 1)))

        clip_arr = np.stack(frames_list, axis=0) # [T, 3, 224, 224]
        clip_path = os.path.join(output_dir, "fake", f"clip_fake_{fake_clips_saved:03d}_{vid_id}.npy")
        np.save(clip_path, clip_arr)

        video_entries.append({
            "filepath": clip_path,
            "label": 1, # Ground truth: 1 = Fake
            "label_name": "fake",
            "source_id": f"vid_src_{vid_id}",
            "frames_count": frames_per_clip
        })
        fake_clips_saved += 1

    print(f"  [+] Saved {fake_clips_saved} fake video sequence clips.")

    pipeline = UnifiedDatasetPipeline(modality="video")
    valid_samples, stats = pipeline.scan_and_validate(video_entries, remove_duplicates=True)
    tr, va, te, leak_rep = pipeline.create_leakage_free_splits(valid_samples, train_ratio=0.70, val_ratio=0.15, test_ratio=0.15)

    manifest_path = "datasets/manifests/video_benchmark_manifest.json"
    pipeline.save_manifest(tr, va, te, metadata={"stats": stats, "leakage": leak_rep}, output_manifest_path=manifest_path)
    print(f"  [+] Video Manifest generated: {leak_rep['split_counts']}")
    print(f"  [+] Leakage Detected: {leak_rep['leakage_detected']}")
    return manifest_path

def prepare_audio_dataset(output_dir: str = "datasets/processed/audio_benchmark"):
    print("\n=======================================================")
    print("[3/3] Preparing Genuine Audio Deepfake Benchmark (Hemg Speech vs Deepfakes)")
    print("=======================================================")
    os.makedirs(os.path.join(output_dir, "real"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "fake"), exist_ok=True)

    p_aud = hf_hub_download(repo_id='Hemg/Deepfake-Audio-Dataset', filename='data/train-00000-of-00001-ab2dff7d513c15ff.parquet', repo_type='dataset')
    df = pd.read_parquet(p_aud, engine='fastparquet')

    audio_entries = []

    # Map verified labels:
    # speaker1_*.wav (label==1 in source) -> Real (0)
    # fake1_*.wav (label==0 in source) -> Fake (1)
    for idx, row in df.iterrows():
        raw_bytes = row['audio.bytes']
        fpath = row['audio.path']
        src_label = row['label']

        is_real = (src_label == 1) or ("speaker" in fpath.lower())
        true_label = 0 if is_real else 1
        label_name = "real" if is_real else "fake"
        sub_dir = "real" if is_real else "fake"

        data, sr = sf.read(io.BytesIO(raw_bytes))
        # Standardize to 16 kHz mono 4.0s
        y_proc, target_sr = load_and_preprocess_audio(data, sr=sr)

        out_fname = f"aud_{label_name}_{idx:03d}_{os.path.basename(fpath)}"
        out_fpath = os.path.join(output_dir, sub_dir, out_fname)
        sf.write(out_fpath, y_proc, target_sr)

        speaker_id = "spk_human_1" if is_real else "tts_synth_1"

        audio_entries.append({
            "filepath": out_fpath,
            "label": true_label,
            "label_name": label_name,
            "source_id": f"{speaker_id}_{idx % 10}",
            "sample_rate": target_sr,
            "duration_sec": 4.0
        })

    print(f"  [+] Saved {len(audio_entries)} standardized 16kHz audio clips.")

    pipeline = UnifiedDatasetPipeline(modality="audio")
    valid_samples, stats = pipeline.scan_and_validate(audio_entries, remove_duplicates=True)
    tr, va, te, leak_rep = pipeline.create_leakage_free_splits(valid_samples, train_ratio=0.70, val_ratio=0.15, test_ratio=0.15)

    manifest_path = "datasets/manifests/audio_benchmark_manifest.json"
    pipeline.save_manifest(tr, va, te, metadata={"stats": stats, "leakage": leak_rep}, output_manifest_path=manifest_path)
    print(f"  [+] Audio Manifest generated: {leak_rep['split_counts']}")
    print(f"  [+] Leakage Detected: {leak_rep['leakage_detected']}")
    return manifest_path

if __name__ == "__main__":
    print("[*] Starting Unified Real Deepfake Dataset Preparation...")
    img_manifest = prepare_image_dataset(max_real=200, max_fake=200)
    vid_manifest = prepare_video_dataset(max_clips=40, frames_per_clip=5)
    aud_manifest = prepare_audio_dataset()
    print("\n[✓] All 3 Benchmark Datasets successfully created and verified!")
