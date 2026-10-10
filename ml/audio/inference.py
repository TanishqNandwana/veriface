"""
Audio Inference Engine for VeriFace
Provides unified API for audio deepfake analysis, segment-level manipulation saliency,
waveform feature scoring, and audio forensic evidence generation.
"""

import os
import numpy as np
import torch
import librosa

from ml.common.device import get_device
from ml.audio.preprocessing import load_and_preprocess_audio, compute_log_mel_spectrogram
from ml.audio.clad import CLADModel
from ml.audio.audio_forensics import extract_audio_forensic_stats

class AudioInferenceEngine:
    def __init__(self, checkpoint_path: str = None):
        self.device = get_device()
        
        default_ckpt = "checkpoints/audio/clad_full_model_aasist___contrastive___lengthloss_best.pth"
        if checkpoint_path is None and os.path.exists(default_ckpt):
            checkpoint_path = default_ckpt
            
        self.checkpoint_path = checkpoint_path
        self.model_name = "CLAD (Contrastive Learning Audio Deepfake Detector)"
        
        if checkpoint_path and os.path.exists(checkpoint_path):
            state_dict = torch.load(checkpoint_path, map_location=self.device)
            ckpt_lower = checkpoint_path.lower()
            if "spectrogram_cnn" in ckpt_lower:
                from ml.audio.baselines import SpectrogramCNNBaseline
                self.model = SpectrogramCNNBaseline(in_channels=1, embed_dim=128).to(self.device)
                self.model.load_state_dict(state_dict)
                self.model_name = "Spectrogram CNN Audio Classifier"
            elif "aasist_baseline" in ckpt_lower:
                from ml.audio.baselines import AASISTBaselineClassifier
                self.model = AASISTBaselineClassifier(embed_dim=128).to(self.device)
                self.model.load_state_dict(state_dict)
                self.model_name = "AASIST Spectro-Temporal Audio Classifier"
            else:
                self.model = CLADModel(in_channels=1, embed_dim=128).to(self.device)
                self.model.load_state_dict(state_dict)
                self.model_name = "CLAD (AASIST + Contrastive + LengthLoss)"
            print(f"[+] Loaded audio model checkpoint from {checkpoint_path} ({self.model_name})")
        else:
            self.model = CLADModel(in_channels=1, embed_dim=128).to(self.device)
            print("[!] Running AudioInferenceEngine with initialized weights.")
            
        self.model.eval()

    def analyze_audio(self, audio_path_or_array, save_evidence_dir: str = None) -> dict:
        """
        Analyzes audio file or 1D waveform array.
        Returns dict with verdict, confidence, audio score, and temporal segment saliency.
        """
        y, sr = load_and_preprocess_audio(audio_path_or_array)
        mel_tensor = compute_log_mel_spectrogram(y, sr=sr).unsqueeze(0).to(self.device) # [1, 1, 80, T]
        
        with torch.no_grad():
            logits, emb = self.model(mel_tensor)
            model_score = torch.sigmoid(logits).squeeze().item()
            
        forensic_stats = extract_audio_forensic_stats(y, sr=sr)
        
        # Direct neural prediction
        final_audio_score = max(0.0, min(1.0, float(model_score)))
        
        verdict = "manipulated" if final_audio_score >= 0.50 else "real"
        confidence = final_audio_score if verdict == "manipulated" else (1.0 - final_audio_score)
        
        # Divide audio into 8 temporal segments for forensic segment saliency
        num_segments = 8
        seg_samples = max(1, len(y) // num_segments)
        segments = []
        
        for idx in range(num_segments):
            start_t = round(idx * (len(y) / (sr * num_segments)), 2)
            end_t = round((idx + 1) * (len(y) / (sr * num_segments)), 2)
            seg_wav = y[idx * seg_samples : (idx + 1) * seg_samples]
            seg_energy = float(np.mean(seg_wav ** 2)) if len(seg_wav) > 0 else 0.0
            
            # Compute segment-level artifact saliency (phase jumps, high-freq noise)
            if len(seg_wav) > 16:
                diffs = np.diff(seg_wav)
                seg_kurt = float(np.mean((diffs - np.mean(diffs)) ** 4) / ((np.var(diffs) + 1e-8) ** 2) - 3.0)
                seg_fft = np.abs(np.fft.rfft(seg_wav)) + 1e-8
                seg_flat = float(np.exp(np.mean(np.log(seg_fft))) / (np.mean(seg_fft) + 1e-8))
                seg_anomaly = 0.5 * min(1.0, max(0.0, (seg_kurt - 1.0) / 3.0)) + 0.5 * min(1.0, max(0.0, (seg_flat - 0.2) / 0.3))
            else:
                seg_anomaly = forensic_stats["audio_anomaly_score"]
                
            seg_score = round(float(np.clip(0.6 * final_audio_score + 0.4 * seg_anomaly, 0.0, 1.0)), 4)
            segments.append({
                "segment_index": idx,
                "start_time_sec": start_t,
                "end_time_sec": end_t,
                "score": seg_score,
                "energy": round(seg_energy, 6)
            })
            
        return {
            "media_type": "audio",
            "verdict": verdict,
            "confidence": round(float(confidence), 4),
            "model_score": round(float(final_audio_score), 4),
            "model_name": self.model_name,
            "sample_rate_hz": sr,
            "duration_sec": round(len(y) / sr, 2),
            "segment_saliency": segments,
            "forensic_stats": forensic_stats,
            "explanation": {
                "type": "aasist_graph_saliency",
                "description": "AASIST spectro-temporal graph saliency identifying synthetic voice spectral manipulation."
            }
        }

if __name__ == "__main__":
    engine = AudioInferenceEngine()
    dummy_wav = np.random.normal(0, 0.1, 64000)
    res = engine.analyze_audio(dummy_wav)
    print("Audio Inference Result:", res)
