"""
Verification Script for Testing ChatGPT / AI-Generated Image Forensic Detection
"""

import cv2
import numpy as np
from ml.image.inference import ImageInferenceEngine

def run_ai_detection_test():
    print("==================================================")
    print(" VeriFace Image AI Generation Forensic Test ")
    print("==================================================")
    
    engine = ImageInferenceEngine(checkpoint_path="experiments/IMG_FAVIT_SMOKE_001/best_checkpoint.pth")
    
    # 1. Real Optical Camera Photograph Simulation
    x = np.linspace(0, 2 * np.pi, 400)
    y = np.linspace(0, 2 * np.pi, 400)
    xx, yy = np.meshgrid(x, y)
    real_img = np.zeros((400, 400, 3), dtype=np.uint8)
    real_img[:, :, 0] = np.clip(120 + 30 * np.sin(xx), 0, 255).astype(np.uint8)
    real_img[:, :, 1] = np.clip(122 + 28 * np.sin(xx), 0, 255).astype(np.uint8)
    real_img[:, :, 2] = np.clip(118 + 29 * np.sin(xx), 0, 255).astype(np.uint8)
    # Add natural camera sensor PRNU noise
    sensor_noise = np.random.normal(0, 3, real_img.shape).astype(np.int16)
    real_img = np.clip(real_img.astype(np.int16) + sensor_noise, 0, 255).astype(np.uint8)
    
    res_real = engine.analyze_image(real_img, save_heatmap_dir="storage/artifacts/TEST_REAL")
    print("\n[+] Natural Camera Image Analysis:")
    print(f"    -> Verdict: {res_real['verdict'].upper()}")
    print(f"    -> Confidence: {res_real['confidence'] * 100:.2f}%")
    print(f"    -> Model Score: {res_real['model_score']}")
    print(f"    -> Forensic Stats: {res_real['forensic_statistics']}")
    
    # 2. ChatGPT / DALL-E 3 / Midjourney / AI-Generated Image Simulation
    ai_img = np.zeros((400, 400, 3), dtype=np.uint8)
    ai_img[:, :, 0] = np.clip(150 + 40 * np.sin(xx), 0, 255).astype(np.uint8)
    ai_img[:, :, 1] = np.clip(100 + 40 * np.cos(yy), 0, 255).astype(np.uint8)
    ai_img[:, :, 2] = 180
    
    # Add generative upsampling checkerboard grid artifacts & color channel discrepancy
    gx = np.arange(400)
    gy = np.arange(400)
    gxx, gyy = np.meshgrid(gx, gy)
    grid = ((gxx // 4 + gyy // 4) % 2 * 50).astype(np.uint8)
    ai_img[:, :, 0] = np.clip(ai_img[:, :, 0] + grid, 0, 255)
    ai_img[:, :, 1] = np.clip(ai_img[:, :, 1] - grid * 0.4, 0, 255)
    
    res_ai = engine.analyze_image(ai_img, save_heatmap_dir="storage/artifacts/TEST_AI")
    print("\n[+] ChatGPT / AI-Generated Image Analysis:")
    print(f"    -> Verdict: {res_ai['verdict'].upper()}")
    print(f"    -> Confidence: {res_ai['confidence'] * 100:.2f}%")
    print(f"    -> Model Score: {res_ai['model_score']}")
    print(f"    -> Forensic Stats: {res_ai['forensic_statistics']}")
    
    print("==================================================")

if __name__ == "__main__":
    run_ai_detection_test()
