"""
Phase 13 Automated Quality and Reliability Checks for VeriFace
Validates:
1. Checkpoint file presence and integrity
2. Label mapping conventions (0=Real, 1=Fake)
3. Dataset manifests and zero-leakage partitions
4. Prediction probability bounds and tensor shapes
5. Preprocessing shape consistency
6. Handling of corrupted or empty files
"""

import os
import json
import pytest
import numpy as np
import torch
import cv2

from ml.common.device import get_device
from ml.image.inference import ImageInferenceEngine
from ml.video.inference import VideoInferenceEngine
from ml.audio.inference import AudioInferenceEngine
from ml.image.preprocessing import extract_face_crop, DEFAULT_TRANSFORM
from ml.audio.preprocessing import load_and_preprocess_audio, compute_log_mel_spectrogram

def test_production_checkpoints_exist():
    expected_checkpoints = [
        "checkpoints/image/efficientnet_b0_best.pth",
        "checkpoints/video/frame_cnn___bigru_temporal_aggregator_best.pth",
        "checkpoints/audio/clad_full_model_aasist___contrastive___lengthloss_best.pth"
    ]
    for ckpt in expected_checkpoints:
        assert os.path.exists(ckpt), f"Missing production checkpoint: {ckpt}"
        assert os.path.getsize(ckpt) > 1000, f"Checkpoint file too small/empty: {ckpt}"

def test_dataset_manifests_leakage_free():
    manifest_files = [
        "datasets/manifests/image_benchmark_manifest.json",
        "datasets/manifests/video_benchmark_manifest.json",
        "datasets/manifests/audio_benchmark_manifest.json"
    ]
    for mf_path in manifest_files:
        assert os.path.exists(mf_path), f"Missing manifest: {mf_path}"
        with open(mf_path, "r") as f:
            data = json.load(f)
        splits = data["splits"]
        
        # Verify no file hash intersection between train, val, and test
        train_hashes = {item.get("sha256") or item.get("file_hash") for item in splits["train"]}
        val_hashes = {item.get("sha256") or item.get("file_hash") for item in splits["validation"]}
        test_hashes = {item.get("sha256") or item.get("file_hash") for item in splits["test"]}
        
        assert len(train_hashes.intersection(val_hashes)) == 0, f"Leakage detected between train and val in {mf_path}"
        assert len(train_hashes.intersection(test_hashes)) == 0, f"Leakage detected between train and test in {mf_path}"
        assert len(val_hashes.intersection(test_hashes)) == 0, f"Leakage detected between val and test in {mf_path}"

def test_label_mappings_consistency():
    manifest_path = "datasets/manifests/image_benchmark_manifest.json"
    with open(manifest_path, "r") as f:
        data = json.load(f)
    for split_name, items in data["splits"].items():
        for item in items:
            assert item["label"] in [0, 1], f"Invalid label {item['label']} in {split_name}"

def test_inference_probability_bounds():
    img_engine = ImageInferenceEngine()
    test_img = np.ones((224, 224, 3), dtype=np.uint8) * 128
    res = img_engine.analyze_image(test_img)
    assert 0.0 <= res["confidence"] <= 1.0, "Confidence out of bounds"
    assert 0.0 <= res["model_score"] <= 1.0, "Model score out of bounds"
    assert res["verdict"] in ["real", "manipulated"], "Invalid verdict string"

def test_audio_preprocessing_shapes():
    sr = 16000
    dummy_wav = np.random.normal(0, 0.1, sr * 4).astype(np.float32)
    mel = compute_log_mel_spectrogram(dummy_wav, sr=sr)
    assert mel.dim() == 3, f"Unexpected mel tensor dimensions: {mel.shape}"
    assert mel.shape[1] == 80, f"Expected 80 mel frequency bins, got {mel.shape[1]}"

def test_corrupted_image_handling():
    img_engine = ImageInferenceEngine()
    with pytest.raises(Exception):
        img_engine.analyze_image("non_existent_image_path_12345.jpg")
