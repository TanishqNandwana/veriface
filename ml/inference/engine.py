"""
VeriFace Master Multimodal Inference Engine
Routes uploaded media (Image, Video, Audio) to appropriate modality detectors
and executes media-aware late fusion for videos containing audio tracks.
"""

import os
import shutil
import subprocess
import numpy as np

from ml.image.inference import ImageInferenceEngine
from ml.video.inference import VideoInferenceEngine
from ml.audio.inference import AudioInferenceEngine

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
AUDIO_EXTENSIONS = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac"}

class MultimodalInferenceEngine:
    def __init__(
        self,
        image_checkpoint: str = None,
        video_checkpoint: str = None,
        audio_checkpoint: str = None,
        artifact_storage_dir: str = "storage/artifacts"
    ):
        print("[+] Initializing VeriFace Multimodal Inference Engine...")
        self.img_engine = ImageInferenceEngine(checkpoint_path=image_checkpoint)
        self.vid_engine = VideoInferenceEngine(checkpoint_path=video_checkpoint)
        self.aud_engine = AudioInferenceEngine(checkpoint_path=audio_checkpoint)
        self.artifact_storage_dir = artifact_storage_dir
        os.makedirs(self.artifact_storage_dir, exist_ok=True)

    def detect_media_type(self, filepath: str) -> str:
        ext = os.path.splitext(filepath)[1].lower()
        if ext in IMAGE_EXTENSIONS:
            return "image"
        elif ext in VIDEO_EXTENSIONS:
            return "video"
        elif ext in AUDIO_EXTENSIONS:
            return "audio"
        else:
            raise ValueError(f"Unsupported media file extension '{ext}'. Supported: Image, Video, Audio.")

    def extract_audio_from_video(self, video_path: str) -> str:
        """Extracts audio track from video file as WAV using FFmpeg if available."""
        if not shutil.which("ffmpeg"):
            return None
            
        audio_path = video_path + ".wav"
        cmd = [
            "ffmpeg", "-y", "-i", video_path,
            "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1",
            audio_path
        ]
        try:
            res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=10)
            if res.returncode == 0 and os.path.exists(audio_path) and os.path.getsize(audio_path) > 100:
                return audio_path
        except Exception:
            pass
        return None

    def analyze(self, filepath: str, media_type: str = None, case_id: str = "CASE_TEMP") -> dict:
        if not media_type:
            media_type = self.detect_media_type(filepath)
            
        case_artifact_dir = os.path.join(self.artifact_storage_dir, case_id)
        os.makedirs(case_artifact_dir, exist_ok=True)
        
        if media_type == "image":
            res = self.img_engine.analyze_image(filepath, save_heatmap_dir=case_artifact_dir)
            
        elif media_type == "video":
            vid_res = self.vid_engine.analyze_video(filepath, save_evidence_dir=case_artifact_dir)
            
            # Check for video's audio track and perform late fusion
            extracted_wav = self.extract_audio_from_video(filepath)
            if extracted_wav:
                try:
                    aud_res = self.aud_engine.analyze_audio(extracted_wav, save_evidence_dir=case_artifact_dir)
                    os.remove(extracted_wav)
                    
                    v_score = vid_res["model_score"]
                    a_score = aud_res["model_score"]
                    
                    # Late Fusion: 60% Visual + 40% Audio
                    fused_score = 0.6 * v_score + 0.4 * a_score
                    fused_verdict = "manipulated" if fused_score >= 0.5 else "real"
                    fused_conf = fused_score if fused_score >= 0.5 else (1.0 - fused_score)
                    
                    vid_res["verdict"] = fused_verdict
                    vid_res["confidence"] = round(float(fused_conf), 4)
                    vid_res["model_score"] = round(float(fused_score), 4)
                    vid_res["multimodal_fusion"] = {
                        "visual_score": round(float(v_score), 4),
                        "audio_score": round(float(a_score), 4),
                        "fusion_strategy": "late_fusion_weighted_average (60% Visual, 40% Audio)"
                    }
                    vid_res["audio_segment_saliency"] = aud_res.get("segment_saliency", [])
                except Exception as e:
                    print(f"[!] Audio track extraction/fusion skipped: {e}")
                    
            res = vid_res
            
        elif media_type == "audio":
            res = self.aud_engine.analyze_audio(filepath, save_evidence_dir=case_artifact_dir)
            
        else:
            raise ValueError(f"Invalid media type '{media_type}'")
            
        res["case_id"] = case_id
        return res

if __name__ == "__main__":
    engine = MultimodalInferenceEngine()
    print("Multimodal Engine initialized successfully!")
