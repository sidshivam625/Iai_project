import numpy as np
from sklearn.metrics import f1_score

class EnsembleFusion:
    def __init__(self, w1=0.6, w2=0.4, tau=0.3):
        """
        w1: Weight for RoBERTa (Deep Semantic)
        w2: Weight for XGBoost (Statistical)
        tau: Threshold for uncertainty
        """
        self.w1 = w1
        self.w2 = w2
        self.tau = tau

    def fit_weights(self, y_true, r1_scores, r2_scores):
        best = {"f1": -1.0, "w1": self.w1, "w2": self.w2}
        y_true = np.asarray(y_true)
        r1_scores = np.asarray(r1_scores)
        r2_scores = np.asarray(r2_scores)

        for w1 in np.linspace(0.0, 1.0, 21):
            w2 = 1.0 - w1
            score = (w1 * r1_scores) + (w2 * r2_scores)
            preds = (score >= 0.5).astype(int)
            f1 = f1_score(y_true, preds, zero_division=0)
            if f1 > best["f1"]:
                best = {"f1": f1, "w1": float(w1), "w2": float(w2)}

        self.w1 = best["w1"]
        self.w2 = best["w2"]
        return best

    def fuse(self, r1: float, r2: float):
        ensemble_score = (self.w1 * r1) + (self.w2 * r2)
        
        uncertainty = 1 if abs(r1 - r2) > self.tau else 0
        
        return ensemble_score, uncertainty
