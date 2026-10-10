"""
Automated Integration & API Unit Tests for VeriFace Backend
"""

import os
import io
import pytest
import numpy as np
import cv2
import soundfile as sf
from fastapi.testclient import TestClient
from backend.main import app

client = TestClient(app)

def test_health_check():
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "device_info" in data
    assert "active_models" in data

def test_analyze_image_api():
    # Create synthetic test image
    img = np.ones((224, 224, 3), dtype=np.uint8) * 180
    cv2.rectangle(img, (50, 50), (150, 150), (0, 0, 255), -1)
    is_success, buffer = cv2.imencode(".jpg", img)
    
    response = client.post(
        "/api/analyze/image",
        files={"file": ("test_face.jpg", buffer.tobytes(), "image/jpeg")}
    )
    assert response.status_code == 200
    data = response.json()
    assert "case_id" in data
    assert data["verdict"] in ["real", "manipulated"]
    assert 0.0 <= data["confidence"] <= 1.0
    assert data["media_type"] == "image"
    assert "analysis_details" in data

def test_analyze_audio_api():
    # Create synthetic audio buffer
    sr = 16000
    wav_arr = (np.random.normal(0, 0.1, sr * 2)).astype(np.float32)
    bio = io.BytesIO()
    sf.write(bio, wav_arr, sr, format="WAV")
    bio.seek(0)
    
    response = client.post(
        "/api/analyze/audio",
        files={"file": ("test_voice.wav", bio.read(), "audio/wav")}
    )
    assert response.status_code == 200
    data = response.json()
    assert "case_id" in data
    assert data["verdict"] in ["real", "manipulated"]
    assert data["media_type"] == "audio"

def test_list_cases_api():
    response = client.get("/api/cases")
    assert response.status_code == 200
    cases = response.json()
    assert isinstance(cases, list)
    assert len(cases) >= 1

def test_pdf_report_api():
    # List cases to find created case ID
    cases_resp = client.get("/api/cases")
    case_id = cases_resp.json()[0]["case_id"]
    
    pdf_resp = client.get(f"/api/reports/{case_id}")
    assert pdf_resp.status_code == 200
    assert pdf_resp.headers["content-type"] == "application/pdf"
