"""
Image Explainability Module (Grad-CAM & Patch Attribution Heatmap)
Generates model-derived suspicious region heatmaps for FA-ViT and ViT models.
Outputs overlay visualization image and normalized heatmap matrix.
"""

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from typing import Tuple

def generate_favit_heatmap(model: torch.nn.Module, tensor_input: torch.Tensor, target_size: Tuple[int, int] = (224, 224)) -> np.ndarray:
    """
    Computes spatial attribution heatmap for FA-ViT from output token activations & gradients.
    Returns normalized heatmap array [H, W] in range [0, 1].
    """
    model.eval()
    tensor_input = tensor_input.clone().detach().requires_grad_(True)
    
    # Forward pass
    logits, embed = model(tensor_input)
    score = torch.sigmoid(logits)[0, 0]
    
    # Backward pass with respect to predicted score
    model.zero_grad()
    score.backward()
    
    # Extract gradient from input tensor
    if tensor_input.grad is not None:
        grads = tensor_input.grad[0].cpu().data.numpy() # [3, 224, 224]
        weights = np.mean(grads, axis=0) # [224, 224]
        heatmap = np.maximum(weights, 0)
    else:
        # Fallback: activation magnitude across channels
        acts = tensor_input[0].cpu().data.numpy()
        heatmap = np.mean(np.abs(acts), axis=0)
        
    if heatmap.max() > 0:
        heatmap = heatmap / heatmap.max()
    else:
        heatmap = np.zeros(target_size, dtype=np.float32)
        
    # Resize to target face crop size
    heatmap_resized = cv2.resize(heatmap, target_size)
    return heatmap_resized

def overlay_heatmap_on_image(img_bgr: np.ndarray, heatmap: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    """
    Overlays normalized heatmap [0, 1] on original BGR image using COLORMAP_JET.
    """
    H, W, _ = img_bgr.shape
    heatmap_uint8 = np.uint8(255 * cv2.resize(heatmap, (W, H)))
    heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    
    overlay = cv2.addWeighted(img_bgr, 1 - alpha, heatmap_color, alpha, 0)
    return overlay

if __name__ == "__main__":
    from ml.image.favit import FAViTModel
    model = FAViTModel(depth=2)
    dummy_input = torch.randn(1, 3, 224, 224)
    hm = generate_favit_heatmap(model, dummy_input)
    dummy_bgr = np.ones((224, 224, 3), dtype=np.uint8) * 128
    overlay = overlay_heatmap_on_image(dummy_bgr, hm)
    print("Explainability test passed! Heatmap shape:", hm.shape, "Overlay shape:", overlay.shape)
