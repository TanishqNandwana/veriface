"""
Unified Dataset Preparation, Validation & Manifest Pipeline for VeriFace
Supports Image, Video, and Audio modalities.
Enforces:
1. File integrity & corruption checks
2. Exact duplicate detection (SHA-256)
3. Train / Validation / Test leakage prevention (Subject & Source ID grouping)
4. Class balance reporting
5. Manifest serialization with metadata and preprocessing specs
"""

import os
import json
import hashlib
import random
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
import cv2
import numpy as np
import soundfile as sf
from PIL import Image

def compute_file_sha256(filepath: str) -> str:
    hasher = hashlib.sha256()
    with open(filepath, 'rb') as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()

def validate_image_file(filepath: str) -> bool:
    try:
        with Image.open(filepath) as img:
            img.verify()
        arr = cv2.imread(filepath)
        return arr is not None and arr.size > 0
    except Exception:
        return False

def validate_audio_file(filepath: str) -> bool:
    try:
        data, sr = sf.read(filepath)
        return len(data) > 0 and sr > 0
    except Exception:
        return False

def validate_video_file_or_npy(filepath: str) -> bool:
    try:
        if filepath.endswith('.npy'):
            arr = np.load(filepath)
            return arr.ndim in (4, 5) and arr.size > 0
        cap = cv2.VideoCapture(filepath)
        if not cap.isOpened():
            return False
        ret, frame = cap.read()
        cap.release()
        return ret and frame is not None and frame.size > 0
    except Exception:
        return False

class UnifiedDatasetPipeline:
    def __init__(self, modality: str, config_path: Optional[str] = None):
        self.modality = modality.lower()
        if self.modality not in ("image", "video", "audio"):
            raise ValueError(f"Unsupported modality: {self.modality}")
        self.config = {}
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r') as f:
                self.config = json.load(f)

    def scan_and_validate(
        self,
        samples: List[Dict[str, Any]],
        remove_duplicates: bool = True
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """
        Validates sample paths, detects corruption, removes duplicates by hash.
        Sample dict expected keys: 'filepath', 'label', 'label_name', optional 'source_id' / 'subject_id'
        """
        valid_samples = []
        seen_hashes = {}
        duplicates_count = 0
        corrupted_count = 0

        for item in samples:
            fpath = item["filepath"]
            if not os.path.exists(fpath):
                corrupted_count += 1
                continue

            # Modality specific corruption check
            is_valid = False
            if self.modality == "image":
                is_valid = validate_image_file(fpath)
            elif self.modality == "audio":
                is_valid = validate_audio_file(fpath)
            elif self.modality == "video":
                is_valid = validate_video_file_or_npy(fpath)

            if not is_valid:
                corrupted_count += 1
                continue

            file_hash = compute_file_sha256(fpath)
            if file_hash in seen_hashes:
                duplicates_count += 1
                if remove_duplicates:
                    continue
            seen_hashes[file_hash] = fpath

            cleaned_item = dict(item)
            cleaned_item["sha256"] = file_hash
            cleaned_item["modality"] = self.modality
            if "source_id" not in cleaned_item:
                cleaned_item["source_id"] = cleaned_item.get("subject_id", os.path.basename(fpath).split(".")[0])
            valid_samples.append(cleaned_item)

        stats = {
            "total_submitted": len(samples),
            "valid_samples": len(valid_samples),
            "corrupted_or_missing": corrupted_count,
            "exact_duplicates": duplicates_count,
            "modality": self.modality
        }
        return valid_samples, stats

    def create_leakage_free_splits(
        self,
        samples: List[Dict[str, Any]],
        train_ratio: float = 0.70,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        seed: int = 42
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
        """
        Groups by source_id / subject_id so clips or frames from the same source
        never leak across train, val, and test splits.
        """
        random.seed(seed)
        np.random.seed(seed)

        # Group by source_id
        grouped = {}
        for s in samples:
            src = s["source_id"]
            if src not in grouped:
                grouped[src] = []
            grouped[src].append(s)

        # Partition source IDs separately by dominant class to maintain balance
        source_real = []
        source_fake = []
        for src, items in grouped.items():
            fake_ratio = sum(1 for it in items if it["label"] == 1) / len(items)
            if fake_ratio >= 0.5:
                source_fake.append(src)
            else:
                source_real.append(src)

        random.shuffle(source_real)
        random.shuffle(source_fake)

        def partition_list(lst):
            n = len(lst)
            n_tr = int(n * train_ratio)
            n_va = int(n * val_ratio)
            return lst[:n_tr], lst[n_tr:n_tr + n_va], lst[n_tr + n_va:]

        tr_r, va_r, te_r = partition_list(source_real)
        tr_f, va_f, te_f = partition_list(source_fake)

        train_srcs = set(tr_r + tr_f)
        val_srcs = set(va_r + va_f)
        test_srcs = set(te_r + te_f)

        train_samples = [s for s in samples if s["source_id"] in train_srcs]
        val_samples = [s for s in samples if s["source_id"] in val_srcs]
        test_samples = [s for s in samples if s["source_id"] in test_srcs]

        # Verify zero leakage
        train_hashes = {s["sha256"] for s in train_samples}
        val_hashes = {s["sha256"] for s in val_samples}
        test_hashes = {s["sha256"] for s in test_samples}

        train_val_leak = len(train_hashes.intersection(val_hashes))
        train_test_leak = len(train_hashes.intersection(test_hashes))
        val_test_leak = len(val_hashes.intersection(test_hashes))

        leakage_report = {
            "train_val_hash_overlap": train_val_leak,
            "train_test_hash_overlap": train_test_leak,
            "val_test_hash_overlap": val_test_leak,
            "leakage_detected": (train_val_leak + train_test_leak + val_test_leak) > 0,
            "split_counts": {
                "train": len(train_samples),
                "val": len(val_samples),
                "test": len(test_samples)
            },
            "class_distribution": {
                "train": {"real": sum(1 for s in train_samples if s["label"] == 0), "fake": sum(1 for s in train_samples if s["label"] == 1)},
                "val": {"real": sum(1 for s in val_samples if s["label"] == 0), "fake": sum(1 for s in val_samples if s["label"] == 1)},
                "test": {"real": sum(1 for s in test_samples if s["label"] == 0), "fake": sum(1 for s in test_samples if s["label"] == 1)}
            }
        }

        return train_samples, val_samples, test_samples, leakage_report

    def save_manifest(
        self,
        train_samples: List[Dict[str, Any]],
        val_samples: List[Dict[str, Any]],
        test_samples: List[Dict[str, Any]],
        metadata: Dict[str, Any],
        output_manifest_path: str
    ):
        os.makedirs(os.path.dirname(output_manifest_path), exist_ok=True)
        manifest_data = {
            "manifest_version": "2.0",
            "modality": self.modality,
            "metadata": metadata,
            "splits": {
                "train": train_samples,
                "validation": val_samples,
                "test": test_samples
            }
        }
        with open(output_manifest_path, "w") as f:
            json.dump(manifest_data, f, indent=2)
        print(f"[+] Saved manifest to {output_manifest_path}")
        return output_manifest_path
