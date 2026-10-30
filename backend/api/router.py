"""
VeriFace API Endpoints Router
Handles media uploads, routes inputs to MultimodalInferenceEngine,
persists analysis records to PostgreSQL/SQLite, and generates PDF forensic reports.
"""

import os
import time
import hashlib
import uuid
from typing import List
from fastapi import APIRouter, UploadFile, File, HTTPException, Depends
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session

from backend.db.database import get_db, engine, Base
from backend.db.models import CaseModel
from backend.schemas.analysis import CaseResponse, HealthResponse
from backend.services.report_service import generate_forensic_pdf_report
from ml.inference.engine import MultimodalInferenceEngine
from ml.common.device import get_device_info

# Initialize DB tables
Base.metadata.create_all(bind=engine)

router = APIRouter(prefix="/api")

# Lazy-loaded singleton inference engine
_inference_engine = None

def get_inference_engine():
    global _inference_engine
    if _inference_engine is None:
        img_candidates = [
            "checkpoints/image/efficientnet_b0_combined_multidataset_best.pth",
            "checkpoints/image/efficientnet_b0_best.pth"
        ]
        img_ckpt = next((c for c in img_candidates if os.path.exists(c)), None)
        
        vid_candidates = [
            "checkpoints/video/efficientnet_bigru_combined_best.pth",
            "checkpoints/video/frame_cnn___bigru_temporal_aggregator_best.pth"
        ]
        vid_ckpt = next((c for c in vid_candidates if os.path.exists(c)), None)
        
        aud_candidates = [
            "checkpoints/audio/aasist_combined_multidataset_best.pth",
            "checkpoints/audio/clad_combined_multidataset_best.pth",
            "checkpoints/audio/clad_full_model_aasist___contrastive___lengthloss_best.pth"
        ]
        aud_ckpt = next((c for c in aud_candidates if os.path.exists(c)), None)
        
        _inference_engine = MultimodalInferenceEngine(
            image_checkpoint=img_ckpt,
            video_checkpoint=vid_ckpt,
            audio_checkpoint=aud_ckpt
        )
    return _inference_engine

def calculate_file_hash(file_bytes: bytes) -> str:
    return hashlib.sha256(file_bytes).hexdigest()

def process_upload_and_analyze(file: UploadFile, expected_media_type: str, db: Session) -> dict:
    start_time = time.time()
    contents = file.file.read()
    
    if len(contents) == 0:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")
        
    if len(contents) > 100 * 1024 * 1024: # 100 MB max limit
        raise HTTPException(status_code=400, detail="File size exceeds 100 MB limit.")
        
    file_hash = calculate_file_hash(contents)
    case_id = f"CASE_{uuid.uuid4().hex[:12].upper()}"
    
    # Save file temporarily
    upload_dir = os.path.join("storage", "uploads")
    os.makedirs(upload_dir, exist_ok=True)
    temp_filepath = os.path.join(upload_dir, f"{case_id}_{file.filename}")
    
    with open(temp_filepath, "wb") as f:
        f.write(contents)
        
    engine = get_inference_engine()
    
    try:
        res = engine.analyze(temp_filepath, media_type=expected_media_type, case_id=case_id)
    except Exception as e:
        if os.path.exists(temp_filepath):
            os.remove(temp_filepath)
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")
        
    processing_time_ms = (time.time() - start_time) * 1000
    
    # Save Case in Database
    case_obj = CaseModel(
        case_id=case_id,
        filename=file.filename,
        media_type=expected_media_type,
        file_hash=file_hash,
        verdict=res["verdict"],
        confidence=res["confidence"],
        model_score=res["model_score"],
        model_name=res["model_name"],
        processing_time_ms=processing_time_ms,
        analysis_details=res
    )
    db.add(case_obj)
    db.commit()
    db.refresh(case_obj)
    
    # Generate Forensic PDF Report
    report_dir = "forensic_reports"
    pdf_path = os.path.join(report_dir, f"{case_id}_report.pdf")
    case_dict = case_obj.to_dict()
    generate_forensic_pdf_report(case_dict, pdf_path)
    
    return case_dict

@router.get("/health", response_model=HealthResponse)
def health_check():
    return {
        "status": "healthy",
        "version": "1.0.0",
        "device_info": get_device_info(),
        "active_models": {
            "image": "EfficientNet-B0 (Compound Scaling Face Deepfake Detector)",
            "video": "Frame-CNN + BiGRU Temporal Aggregator",
            "audio": "CLAD (Contrastive Learning Audio Deepfake Detector)"
        }
    }

@router.post("/analyze/image")
def analyze_image(file: UploadFile = File(...), db: Session = Depends(get_db)):
    return process_upload_and_analyze(file, expected_media_type="image", db=db)

@router.post("/analyze/video")
def analyze_video(file: UploadFile = File(...), db: Session = Depends(get_db)):
    return process_upload_and_analyze(file, expected_media_type="video", db=db)

@router.post("/analyze/audio")
def analyze_audio(file: UploadFile = File(...), db: Session = Depends(get_db)):
    return process_upload_and_analyze(file, expected_media_type="audio", db=db)

@router.get("/cases")
def list_cases(limit: int = 50, db: Session = Depends(get_db)):
    cases = db.query(CaseModel).order_by(CaseModel.created_at.desc()).limit(limit).all()
    return [c.to_dict() for c in cases]

@router.get("/cases/{case_id}")
def get_case_details(case_id: str, db: Session = Depends(get_db)):
    case = db.query(CaseModel).filter(CaseModel.case_id == case_id).first()
    if not case:
        raise HTTPException(status_code=404, detail="Case not found.")
    return case.to_dict()

@router.get("/reports/{case_id}")
def get_case_pdf_report(case_id: str):
    pdf_path = os.path.join("forensic_reports", f"{case_id}_report.pdf")
    if not os.path.exists(pdf_path):
        raise HTTPException(status_code=404, detail="Forensic PDF report not found.")
    return FileResponse(pdf_path, media_type="application/pdf", filename=f"VeriFace_{case_id}_Report.pdf")
