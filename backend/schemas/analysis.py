"""
Pydantic Schemas for VeriFace Backend APIs
"""

from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any

class CaseResponse(BaseModel):
    case_id: str
    filename: str
    media_type: str
    file_hash: str
    verdict: str
    confidence: float
    model_score: float
    model_name: str
    processing_time_ms: float
    created_at: Optional[str] = None
    analysis_details: Optional[Dict[str, Any]] = None

class HealthResponse(BaseModel):
    status: str
    version: str
    device_info: Dict[str, Any]
    active_models: Dict[str, str]
