"""
VeriFace FastAPI Main Web Server Application
"""

import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from backend.api.router import router as api_router

app = FastAPI(
    title="VeriFace — Verify the Media. Reveal the Truth.",
    description="Multimodal AI platform for detecting image, video, and audio deepfakes with explainable forensic reports.",
    version="1.0.0"
)

# Enable CORS for React frontend development and production deployments
allowed_origins_env = os.getenv("CORS_ORIGINS")
origins = [o.strip() for o in allowed_origins_env.split(",") if o.strip()] if allowed_origins_env else ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API Router
app.include_router(api_router)

# Mount static storage directories for media artifacts and PDF reports
os.makedirs("storage", exist_ok=True)
os.makedirs("forensic_reports", exist_ok=True)

app.mount("/storage", StaticFiles(directory="storage"), name="storage")
app.mount("/forensic_reports", StaticFiles(directory="forensic_reports"), name="reports")

# Mount React frontend static files if built
frontend_dist = os.path.join("frontend", "dist")
if os.path.exists(frontend_dist):
    app.mount("/", StaticFiles(directory=frontend_dist, html=True), name="frontend")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.main:app", host="0.0.0.0", port=8000, reload=True)
