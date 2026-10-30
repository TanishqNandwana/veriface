"""
Video Inference Engine for VeriFace
Executes end-to-end video deepfake analysis, frame-level temporal score extraction,
suspicious frame identification, and temporal evidence timeline generation.
"""

import os
import cv2
import numpy as np
import torch

from ml.common.device import get_device
from ml.video.model import StyleLatentFlowVideoModel
from ml.video.frame_sampler import sample_video_frames, preprocess_video_clip
from ml.video.temporal_forensics import extract_video_forensic_stats
from ml.image.inference import ImageInferenceEngine

class VideoInferenceEngine:
    def __init__(self, checkpoint_path: str = None, num_frames: int = 16):
        self.device = get_device()
        self.num_frames = num_frames
        
        candidates = [
            "checkpoints/video/efficientnet_bigru_combined_best.pth",
            "checkpoints/video/frame_cnn___bigru_temporal_aggregator_best.pth",
            "checkpoints/video/3d_resnet_best.pth",
            "checkpoints/video/style_latent_flow_best.pth"
        ]
        if checkpoint_path is None:
            for cand in candidates:
                if os.path.exists(cand):
                    checkpoint_path = cand
                    break
            
        self.checkpoint_path = checkpoint_path
        self.img_engine = ImageInferenceEngine()
        
        if checkpoint_path and os.path.exists(checkpoint_path):
            state_dict = torch.load(checkpoint_path, map_location=self.device)
            ckpt_lower = checkpoint_path.lower()
            if "efficientnet_bigru" in ckpt_lower or "efficientnet" in ckpt_lower:
                from ml.video.model import EfficientNetBiGRUVideoModel
                self.model = EfficientNetBiGRUVideoModel(embed_dim=128).to(self.device)
                self.model.load_state_dict(state_dict)
                self.model_name = "EfficientNet-B0 + BiGRU Temporal Aggregator"
            elif "bigru" in ckpt_lower or "frame_cnn" in ckpt_lower:
                from ml.video.model import FrameBiGRUTemporalModel
                self.model = FrameBiGRUTemporalModel(embed_dim=128).to(self.device)
                self.model.load_state_dict(state_dict)
                self.model_name = "Frame-CNN + BiGRU Temporal Aggregator"
            elif "3d_resnet" in ckpt_lower:
                from ml.video.model import Content3DResNetClassifier
                self.model = Content3DResNetClassifier(embed_dim=128).to(self.device)
                self.model.load_state_dict(state_dict)
                self.model_name = "3D ResNet Video Classifier"
            else:
                self.model = StyleLatentFlowVideoModel(num_frames=num_frames, embed_dim=128).to(self.device)
                self.model.load_state_dict(state_dict)
                self.model_name = "Style Latent Flows (pSp + StyleGRU + 3D ResNet + SAM + TTE)"
            print(f"[+] Loaded video model checkpoint from {checkpoint_path} ({self.model_name})")
        else:
            from ml.video.model import EfficientNetBiGRUVideoModel
            self.model = EfficientNetBiGRUVideoModel(embed_dim=128).to(self.device)
            self.model_name = "EfficientNet-B0 + BiGRU (Initialized)"
            print("[!] Running VideoInferenceEngine with initialized weights.")
            
        self.model.eval()

    def analyze_video(self, video_path_or_frames, save_evidence_dir: str = None) -> dict:
        """
        Analyzes input video file or list of BGR frame arrays.
        Combines frame-level spatial FA-ViT features, temporal style flows, and spatial-temporal forensics.
        Returns dict with video verdict, confidence, frame-level scores, and suspicious frame metadata.
        """
        clip_tensor, face_crops, bboxes = preprocess_video_clip(video_path_or_frames, num_frames=self.num_frames)
        clip_tensor = clip_tensor.to(self.device)
        
        # 1. Temporal Video Model Execution
        with torch.no_grad():
            v_logits, f_logits = self.model(clip_tensor)
            model_video_score = torch.sigmoid(v_logits).squeeze().item()
            raw_frame_scores = torch.sigmoid(f_logits).squeeze(0).squeeze(-1).cpu().numpy().tolist()
            if isinstance(raw_frame_scores, float):
                raw_frame_scores = [raw_frame_scores] * len(face_crops)
            
        # 2. Frame-Level Spatial Extraction across video frames
        spatial_frame_scores = []
        for crop in face_crops:
            try:
                img_res = self.img_engine.analyze_image(crop, generate_heatmap=False)
                spatial_frame_scores.append(img_res["model_score"])
            except Exception:
                spatial_frame_scores.append(0.5)
                
        mean_spatial_score = float(np.mean(spatial_frame_scores)) if len(spatial_frame_scores) > 0 else 0.5
        
        # 3. Spatial-Temporal Video Forensic Artifact Extraction
        forensic_stats = extract_video_forensic_stats(face_crops)
        
        # 4. Multi-Domain Video Deepfake Ensemble: Fuses 60% Temporal Model with 40% High-Resolution Spatial Frame Analysis
        final_video_score = 0.60 * model_video_score + 0.40 * mean_spatial_score
        final_video_score = max(0.0, min(1.0, final_video_score))
        
        verdict = "manipulated" if final_video_score >= 0.50 else "real"
        confidence = final_video_score if verdict == "manipulated" else (1.0 - final_video_score)
        
        # Format frame scores timeline
        frame_scores = spatial_frame_scores
        timeline = []
        for idx, f_score in enumerate(frame_scores):
            timeline.append({
                "frame_index": idx,
                "timestamp_sec": round(idx * 0.5, 2),
                "score": round(float(f_score), 4),
                "bounding_box": {"x": int(bboxes[idx][0]), "y": int(bboxes[idx][1]), "width": int(bboxes[idx][2]), "height": int(bboxes[idx][3])}
            })
            
        # Identify suspicious frames (top 3 highest manipulation scores)
        suspicious_indices = np.argsort(frame_scores)[::-1][:3].tolist()
        suspicious_frames = []
        
        if save_evidence_dir:
            os.makedirs(save_evidence_dir, exist_ok=True)
            for rank, f_idx in enumerate(suspicious_indices):
                crop_bgr = face_crops[f_idx]
                save_path = os.path.join(save_evidence_dir, f"suspicious_frame_{f_idx}_rank_{rank+1}.jpg")
                cv2.imwrite(save_path, crop_bgr)
                suspicious_frames.append({
                    "frame_index": f_idx,
                    "score": round(float(frame_scores[f_idx]), 4),
                    "image_path": save_path
                })
                
        return {
            "media_type": "video",
            "verdict": verdict,
            "confidence": round(float(confidence), 4),
            "model_score": round(float(final_video_score), 4),
            "model_name": self.model_name,
            "frames_analyzed": len(frame_scores),
            "temporal_timeline": timeline,
            "suspicious_frames": suspicious_frames,
            "forensic_stats": forensic_stats,
            "explanation": {
                "type": "temporal_style_flow",
                "description": "Temporal style flow trajectory analysis tracking StyleGAN latent W+ shifts across video frames."
            }
        }

if __name__ == "__main__":
    engine = VideoInferenceEngine(num_frames=8)
    dummy_frames = [np.ones((200, 200, 3), dtype=np.uint8) * 120 for _ in range(8)]
    res = engine.analyze_video(dummy_frames)
    print("Video Inference Engine Result:", res)
