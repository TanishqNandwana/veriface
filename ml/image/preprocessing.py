"""
Image Preprocessing & Face Crop Utility for VeriFace
Extracts face bounding boxes using OpenCV Haar Cascade / MTCNN, resizes to 224x224,
and applies ImageNet / FA-ViT normalization.
"""

import os
import cv2
import numpy as np
import torch
from typing import Tuple
from PIL import Image
from torchvision import transforms

# Default ImageNet normalization
DEFAULT_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def extract_face_crop(img_path_or_array, margin: float = 0.2) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
    """
    Detects the primary face in an image using OpenCV Haar Cascade fallback.
    Returns (cropped_face_bgr_array, (x, y, w, h)).
    """
    if isinstance(img_path_or_array, str):
        image = cv2.imread(img_path_or_array)
        if image is None:
            raise ValueError(f"Unable to read image at {img_path_or_array}")
    else:
        image = img_path_or_array.copy()

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = []
    
    try:
        if hasattr(cv2, "data") and hasattr(cv2, "CascadeClassifier"):
            cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            face_cascade = cv2.CascadeClassifier(cascade_path)
            if not face_cascade.empty():
                faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))
    except Exception:
        faces = []
        
    H, W, _ = image.shape
    if len(faces) == 0:
        # Fallback: center crop 80% of image if no face detected
        x = int(W * 0.1)
        y = int(H * 0.1)
        w = int(W * 0.8)
        h = int(H * 0.8)
    else:
        # Select largest face bounding box
        faces = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)
        x, y, w, h = faces[0]
        
        # Apply margin expansion
        mx = int(w * margin)
        my = int(h * margin)
        x = max(0, x - mx)
        y = max(0, y - my)
        w = min(W - x, w + 2 * mx)
        h = min(H - y, h + 2 * my)
        
    face_crop = image[y:y+h, x:x+w]
    return face_crop, (x, y, w, h)

def preprocess_image_tensor(img_path_or_array) -> Tuple[torch.Tensor, np.ndarray, Tuple[int, int, int, int]]:
    """
    Extracts face, crops, applies 224x224 resize and normalization.
    Returns (tensor_4d [1, 3, 224, 224], face_crop_bgr, bbox).
    """
    face_crop, bbox = extract_face_crop(img_path_or_array)
    rgb_face = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(rgb_face)
    
    tensor_3d = DEFAULT_TRANSFORM(pil_img)
    tensor_4d = tensor_3d.unsqueeze(0) # [1, 3, 224, 224]
    
    return tensor_4d, face_crop, bbox

if __name__ == "__main__":
    dummy_arr = np.zeros((300, 300, 3), dtype=np.uint8)
    cv2.rectangle(dummy_arr, (50, 50), (200, 200), (255, 255, 255), -1)
    tensor_4d, crop, bbox = preprocess_image_tensor(dummy_arr)
    print("Preprocessing Test: Tensor Shape =", tensor_4d.shape, "BBox =", bbox)
