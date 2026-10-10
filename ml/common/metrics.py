"""
VeriFace Research & Evaluation Metrics Utility
Computes Accuracy, Precision, Recall, F1, ROC-AUC, EER, FAR, FRR, and Confusion Matrix.
"""

import numpy as np
from typing import Dict, Any, Tuple
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, roc_curve

def compute_eer(y_true: np.ndarray, y_score: np.ndarray) -> Tuple[float, float, float, float]:
    """
    Computes Equal Error Rate (EER), optimal threshold, FAR, and FRR.
    """
    try:
        fpr, tpr, thresholds = roc_curve(y_true, y_score, pos_label=1)
        fnr = 1 - tpr
        
        # EER is point where FPR ~ FNR
        diffs = np.abs(fpr - fnr)
        if len(diffs) == 0 or np.isnan(diffs).all():
            return 0.5, 0.5, 0.5, 0.5
            
        eer_idx = np.nanargmin(diffs)
        eer = float((fpr[eer_idx] + fnr[eer_idx]) / 2.0)
        optimal_threshold = float(thresholds[eer_idx]) if eer_idx < len(thresholds) else 0.5
        far = float(fpr[eer_idx])
        frr = float(fnr[eer_idx])
        return eer, optimal_threshold, far, frr
    except Exception:
        return 0.5, 0.5, 0.5, 0.5

def compute_all_metrics(y_true: np.ndarray, y_score: np.ndarray, threshold: float = 0.5) -> Dict[str, Any]:
    """
    Computes complete suite of evaluation metrics for deepfake detection models.
    y_true: binary labels (0 = Real, 1 = Manipulated)
    y_score: predicted continuous probabilities for class 1
    """
    y_true = np.array(y_true, dtype=int)
    y_score = np.array(y_score, dtype=float)
    
    y_pred = (y_score >= threshold).astype(int)
    
    acc = float(accuracy_score(y_true, y_pred))
    prec = float(precision_score(y_true, y_pred, zero_division=0))
    rec = float(recall_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    
    try:
        auc = float(roc_auc_score(y_true, y_score))
    except Exception:
        auc = 0.5
        
    eer, optimal_thresh, far, frr = compute_eer(y_true, y_score)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist()
    
    return {
        "accuracy": round(acc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1_score": round(f1, 4),
        "roc_auc": round(auc, 4),
        "eer": round(eer, 4),
        "far": round(far, 4),
        "frr": round(frr, 4),
        "threshold_used": threshold,
        "optimal_eer_threshold": round(optimal_thresh, 4),
        "confusion_matrix": cm,
        "sample_count": len(y_true)
    }

if __name__ == "__main__":
    y_true = np.array([0, 0, 0, 1, 1, 1])
    y_score = np.array([0.1, 0.2, 0.4, 0.7, 0.85, 0.9])
    print("Test Metrics:", compute_all_metrics(y_true, y_score))
