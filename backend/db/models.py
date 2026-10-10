"""
VeriFace Case History & Forensic Record Models
"""

from datetime import datetime
from sqlalchemy import Column, String, Float, DateTime, Text, JSON
from backend.db.database import Base

class CaseModel(Base):
    __tablename__ = "cases"

    case_id = Column(String, primary_key=True, index=True)
    filename = Column(String, nullable=False)
    media_type = Column(String, nullable=False)
    file_hash = Column(String, nullable=False)
    verdict = Column(String, nullable=False) # 'real' or 'manipulated'
    confidence = Column(Float, nullable=False)
    model_score = Column(Float, nullable=False)
    model_name = Column(String, nullable=False)
    processing_time_ms = Column(Float, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Structured JSON analysis details (bboxes, temporal timeline, segment saliency, heatmap paths)
    analysis_details = Column(JSON, nullable=True)

    def to_dict(self):
        return {
            "case_id": self.case_id,
            "filename": self.filename,
            "media_type": self.media_type,
            "file_hash": self.file_hash,
            "verdict": self.verdict,
            "confidence": round(self.confidence, 4),
            "model_score": round(self.model_score, 4),
            "model_name": self.model_name,
            "processing_time_ms": round(self.processing_time_ms, 2),
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "analysis_details": self.analysis_details
        }
