"""
VeriFace End-to-End System & Pipeline Verification Test Suite
"""

import os
import io
import pytest
import numpy as np
import cv2
import soundfile as sf
from fastapi.testclient import TestClient

from backend.main import app
from ml.inference.engine import MultimodalInferenceEngine

client = TestClient(app)

def test_multimodal_inference_engine_direct():
    engine = MultimodalInferenceEngine()
    
    # 1. Image test
    img = np.ones((224, 224, 3), dtype=np.uint8) * 120
    img_res = engine.img_engine.analyze_image(img, save_heatmap_dir="storage/artifacts/TEST_IMG")
    assert img_res["media_type"] == "image"
    assert img_res["verdict"] in ["real", "manipulated"]
    
    # 2. Video test
    vid_frames = [np.ones((200, 200, 3), dtype=np.uint8) * 100 for _ in range(8)]
    vid_res = engine.vid_engine.analyze_video(vid_frames, save_evidence_dir="storage/artifacts/TEST_VID")
    assert vid_res["media_type"] == "video"
    assert len(vid_res["temporal_timeline"]) in [7, 8] # Frame timeline sequence
    
    # 3. Audio test
    aud_wav = np.random.normal(0, 0.1, 64000).astype(np.float32)
    aud_res = engine.aud_engine.analyze_audio(aud_wav, save_evidence_dir="storage/artifacts/TEST_AUD")
    assert aud_res["media_type"] == "audio"
    assert len(aud_res["segment_saliency"]) == 8

def test_full_api_workflow():
    # Health
    h_resp = client.get("/api/health")
    assert h_resp.status_code == 200
    
    # Analyze Image via API
    img = np.ones((224, 224, 3), dtype=np.uint8) * 150
    _, img_buf = cv2.imencode(".jpg", img)
    res_img = client.post("/api/analyze/image", files={"file": ("e2e_face.jpg", img_buf.tobytes(), "image/jpeg")})
    assert res_img.status_code == 200
    case_img = res_img.json()
    
    # Analyze Audio via API
    sr = 16000
    wav_arr = np.random.normal(0, 0.1, sr * 4).astype(np.float32)
    bio = io.BytesIO()
    sf.write(bio, wav_arr, sr, format="WAV")
    bio.seek(0)
    res_aud = client.post("/api/analyze/audio", files={"file": ("e2e_audio.wav", bio.read(), "audio/wav")})
    assert res_aud.status_code == 200
    case_aud = res_aud.json()
    
    # Verify PDF Report Generation for created case
    pdf_resp = client.get(f"/api/reports/{case_img['case_id']}")
    assert pdf_resp.status_code == 200
    assert len(pdf_resp.content) > 1000 # Valid non-empty PDF document
