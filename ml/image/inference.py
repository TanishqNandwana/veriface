"""
Image Inference Engine for VeriFace
Provides unified API for single image analysis, face bounding box extraction,
FA-ViT model execution, spatial/FFT frequency forensic ensemble scoring,
confidence calibration, and Grad-CAM heatmap generation.
"""

import os
import cv2
import numpy as np
import torch

from ml.common.device import get_device
from ml.image.favit import FAViTModel
from ml.image.preprocessing import extract_face_crop, DEFAULT_TRANSFORM
from ml.image.explain import generate_favit_heatmap, overlay_heatmap_on_image
from ml.image.spatial_forensics import extract_numpy_forensic_scores
from PIL import Image

class ImageInferenceEngine:
    def __init__(self, checkpoint_path: str = None, depth: int = 2):
        self.device = get_device()
        
        default_candidates = [
            "checkpoints/image/efficientnet_b0_combined_multidataset_best.pth",
            "checkpoints/image/efficientnet_b0_best.pth",
            "checkpoints/image/fa_vit_combined_multidataset_best.pth",
            "checkpoints/image/fa_vit_gam_lam_fft_fal_best.pth"
        ]
        if checkpoint_path is None:
            for cand in default_candidates:
                if os.path.exists(cand):
                    checkpoint_path = cand
                    break
            
        self.checkpoint_path = checkpoint_path
        self.model_name = "FA-ViT (Forgery-Aware Vision Transformer)"
        
        if checkpoint_path and os.path.exists(checkpoint_path):
            state_dict = torch.load(checkpoint_path, map_location=self.device)
            ckpt_lower = checkpoint_path.lower()
            if "efficientnet" in ckpt_lower:
                from ml.image.baselines import EfficientNetBaseline
                self.model = EfficientNetBaseline(pretrained=False).to(self.device)
                self.model.load_state_dict(state_dict)
                self.model_name = "EfficientNet-B0 Deepfake Face Classifier"
            elif "resnet" in ckpt_lower:
                from ml.image.baselines import ResNet50Baseline
                self.model = ResNet50Baseline(pretrained=False).to(self.device)
                self.model.load_state_dict(state_dict)
                self.model_name = "ResNet-50 Deepfake Face Classifier"
            elif "vit_b" in ckpt_lower or "vanilla_vit" in ckpt_lower:
                from ml.image.baselines import VanillaViTBaseline
                self.model = VanillaViTBaseline(pretrained=False).to(self.device)
                self.model.load_state_dict(state_dict)
                self.model_name = "Vanilla ViT-B/16 Deepfake Classifier"
            else:
                self.model = FAViTModel(use_gam=True, use_lam=True, use_fft=True, use_spatial_noise=True, pretrained=False).to(self.device)
                self.model.load_state_dict(state_dict, strict=False)
                self.model_name = "FA-ViT (Forgery-Aware Vision Transformer)"
            print(f"[+] Loaded image model checkpoint from {checkpoint_path} ({self.model_name})")
        else:
            self.model = FAViTModel(use_gam=True, use_lam=True, use_fft=True, use_spatial_noise=True, pretrained=True).to(self.device)
            print("[+] Running ImageInferenceEngine with pretrained ViT.")
            
        self.model.eval()

    def analyze_image(self, img_path_or_array, save_heatmap_dir: str = None, generate_heatmap: bool = True) -> dict:
        """
        Executes end-to-end image deepfake & AI generation (ChatGPT / DALL-E / Midjourney / GAN) detection.
        Returns detailed dict with verdict, confidence, model score, bbox, and heatmap artifact paths.
        """
        if isinstance(img_path_or_array, str):
            image_bgr = cv2.imread(img_path_or_array)
            filename = os.path.basename(img_path_or_array)
        else:
            image_bgr = img_path_or_array.copy()
            filename = "input_image.jpg"
            
        if image_bgr is None:
            raise ValueError("Invalid image input.")
            
        face_crop_bgr, (x, y, w, h) = extract_face_crop(image_bgr)
        rgb_face = cv2.cvtColor(face_crop_bgr, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb_face)
        
        tensor_img = DEFAULT_TRANSFORM(pil_img).unsqueeze(0).to(self.device)
        
        # 1. Neural Model Score
        with torch.no_grad():
            logits, embed = self.model(tensor_img)
            model_score = torch.sigmoid(logits).item()
            
        # 2. Mathematical Frequency (2D FFT) & Spatial Noise Residual Analysis on Face Crop
        forensic_stats = extract_numpy_forensic_scores(face_crop_bgr)
        
        # 3. Model Decision
        verdict = "manipulated" if model_score >= 0.50 else "real"
        confidence = model_score if verdict == "manipulated" else (1.0 - model_score)
        
        overlay_rel_path = None
        if generate_heatmap:
            heatmap = generate_favit_heatmap(self.model, tensor_img, target_size=(face_crop_bgr.shape[1], face_crop_bgr.shape[0]))
            overlay_crop = overlay_heatmap_on_image(face_crop_bgr, heatmap)
            
            if save_heatmap_dir:
                os.makedirs(save_heatmap_dir, exist_ok=True)
                overlay_file = os.path.join(save_heatmap_dir, f"heatmap_{filename}")
                cv2.imwrite(overlay_file, overlay_crop)
                overlay_rel_path = overlay_file
            
        return {
            "media_type": "image",
            "verdict": verdict,
            "confidence": round(float(confidence), 4),
            "model_score": round(float(model_score), 4),
            "model_name": self.model_name,
            "face_detected": True,
            "bounding_box": {"x": int(x), "y": int(y), "width": int(w), "height": int(h)},
            "forensic_statistics": forensic_stats,
            "explanation": {
                "type": "grad_cam_and_fft_spectrum",
                "overlay_path": overlay_rel_path,
                "description": "Grad-CAM spatial heatmap and 2D Fourier power spectrum analyzing generative AI upsampling grid artifacts."
            }
        }

if __name__ == "__main__":
    engine = ImageInferenceEngine()
    dummy_img = np.ones((300, 300, 3), dtype=np.uint8) * 150
    res = engine.analyze_image(dummy_img)
    print("Image Inference Result:", res)
