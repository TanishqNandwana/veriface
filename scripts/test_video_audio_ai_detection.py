"""
Script to test Video and Audio deepfake detection engines on real vs synthetic media samples.
"""

import numpy as np
import os
import cv2
import soundfile as sf

from ml.video.inference import VideoInferenceEngine
from ml.audio.inference import AudioInferenceEngine

def main():
    print("==================================================")
    print(" VeriFace Video & Audio Forensic Diagnostic Test ")
    print("==================================================")
    
    # --- 1. VIDEO DIAGNOSTIC ---
    # Create Real Video Sample: natural video frames with continuous motion & camera PRNU noise
    real_video_frames = []
    for i in range(16):
        frame = np.zeros((224, 224, 3), dtype=np.uint8)
        frame[:, :, 0] = np.clip(120 + 20 * np.sin(i * 0.2), 0, 255)
        frame[:, :, 1] = np.clip(122 + 18 * np.sin(i * 0.2), 0, 255)
        frame[:, :, 2] = np.clip(118 + 19 * np.sin(i * 0.2), 0, 255)
        sensor_noise = np.random.normal(0, 3, frame.shape).astype(np.int16)
        frame = np.clip(frame.astype(np.int16) + sensor_noise, 0, 255).astype(np.uint8)
        real_video_frames.append(frame)
        
    # Create Fake Video Sample: synthetic frame-to-frame W+ latent jitter + upsampling artifacts
    fake_video_frames = []
    for i in range(16):
        frame = np.zeros((224, 224, 3), dtype=np.uint8)
        frame[:, :, 0] = np.clip(150 + 40 * np.sin(i * 0.5), 0, 255)
        frame[:, :, 1] = np.clip(100 + 40 * np.cos(i * 0.5), 0, 255)
        frame[:, :, 2] = 180
        # Add high frequency upsampling grid
        grid = (((np.arange(224)[:, None] // 4 + np.arange(224)[None, :] // 4) % 2) * 50).astype(np.uint8)
        frame[:, :, 0] = np.clip(frame[:, :, 0] + grid, 0, 255)
        fake_video_frames.append(frame)
        
    video_engine = VideoInferenceEngine(checkpoint_path="experiments/VID_STYLEFLOW_SMOKE_001/best_checkpoint.pth", num_frames=16)
    
    real_vid_res = video_engine.analyze_video(real_video_frames)
    fake_vid_res = video_engine.analyze_video(fake_video_frames)
    
    print("\n[+] VIDEO DIAGNOSTIC RESULTS:")
    print(f"    Real Video Sample  -> Verdict: {real_vid_res['verdict'].upper()} (Confidence: {real_vid_res['confidence']*100:.2f}%, Model Score: {real_vid_res['model_score']})")
    print(f"    Fake Video Sample  -> Verdict: {fake_vid_res['verdict'].upper()} (Confidence: {fake_vid_res['confidence']*100:.2f}%, Model Score: {fake_vid_res['model_score']})")
    
    # --- 2. AUDIO DIAGNOSTIC ---
    # Create Real Audio Sample: clean natural speech-like harmonic tone with smooth envelope
    sr = 16000
    duration = 4.0
    t = np.linspace(0, duration, int(sr * duration))
    # Natural harmonic series with natural decay
    real_wav = 0.5 * np.sin(2 * np.pi * 220 * t) + 0.2 * np.sin(2 * np.pi * 440 * t) + 0.05 * np.random.normal(0, 0.02, len(t))
    
    # Create Fake Audio Sample: TTS / VC synthetic audio with phase discontinuities & spectral high-freq robotic noise
    fake_wav = 0.5 * np.sin(2 * np.pi * 350 * t) + 0.4 * np.random.normal(0, 0.3, len(t))
    # Add phase jumps / synthesis frame boundary clicks
    for click_idx in range(0, len(fake_wav), 1600):
        fake_wav[click_idx:click_idx+50] += 0.8 * np.random.uniform(-1, 1, 50)
        
    audio_engine = AudioInferenceEngine(checkpoint_path="experiments/AUD_CLAD_SMOKE_001/best_checkpoint.pth")
    
    real_aud_res = audio_engine.analyze_audio(real_wav)
    fake_aud_res = audio_engine.analyze_audio(fake_wav)
    
    print("\n[+] AUDIO DIAGNOSTIC RESULTS:")
    print(f"    Real Audio Sample  -> Verdict: {real_aud_res['verdict'].upper()} (Confidence: {real_aud_res['confidence']*100:.2f}%, Model Score: {real_aud_res['model_score']})")
    print(f"    Fake Audio Sample  -> Verdict: {fake_aud_res['verdict'].upper()} (Confidence: {fake_aud_res['confidence']*100:.2f}%, Model Score: {fake_aud_res['model_score']})")
    print("==================================================")

if __name__ == "__main__":
    main()
