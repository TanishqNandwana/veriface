"""
Video Frame Sampler & Preprocessing Utility
Extracts uniform or keyframe video clips (8 to 32 frames), detects face crops across frames,
and resizes crops to 224x224 for style latent flow and 3D visual analysis.
"""

import os
import cv2
import numpy as np
import torch
from typing import List, Tuple
from ml.image.preprocessing import extract_face_crop, DEFAULT_TRANSFORM
from PIL import Image

def sample_video_frames(video_path: str, num_frames: int = 16) -> List[np.ndarray]:
    """
    Uniformly samples N frames from a video file.
    Returns list of BGR frame numpy arrays.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Unable to open video file at {video_path}")
        
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        total_frames = num_frames
        
    indices = np.linspace(0, max(0, total_frames - 1), num_frames, dtype=int)
    frames = []
    
    current_idx = 0
    frame_set = set(indices)
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if current_idx in frame_set:
            frames.append(frame)
            if len(frames) >= num_frames:
                break
        current_idx += 1
        
    cap.release()
    
    # If video had fewer frames than requested, repeat last frame
    while len(frames) < num_frames:
        if len(frames) > 0:
            frames.append(frames[-1].copy())
        else:
            frames.append(np.zeros((224, 224, 3), dtype=np.uint8))
            
    return frames

def preprocess_video_clip(video_path_or_frames, num_frames: int = 16) -> Tuple[torch.Tensor, List[np.ndarray], List[Tuple[int, int, int, int]]]:
    """
    Samples frames, crops faces across consecutive frames, normalizes to tensor [1, T, 3, 224, 224].
    Returns (clip_tensor, face_crops_bgr, bboxes).
    """
    if isinstance(video_path_or_frames, str):
        frames = sample_video_frames(video_path_or_frames, num_frames=num_frames)
    else:
        frames = video_path_or_frames
        
    face_crops = []
    bboxes = []
    tensors = []
    
    for f in frames:
        crop, bbox = extract_face_crop(f)
        face_crops.append(crop)
        bboxes.append(bbox)
        
        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb)
        tensors.append(DEFAULT_TRANSFORM(pil_img))
        
    clip_tensor = torch.stack(tensors, dim=0).unsqueeze(0) # [1, T, 3, 224, 224]
    return clip_tensor, face_crops, bboxes

if __name__ == "__main__":
    dummy_frames = [np.ones((200, 200, 3), dtype=np.uint8) * 100 for _ in range(8)]
    tensor_5d, crops, bboxes = preprocess_video_clip(dummy_frames, num_frames=8)
    print("Video Preprocessing Test: Clip Tensor Shape =", tensor_5d.shape, "Crops =", len(crops))
