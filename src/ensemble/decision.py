import json
import numpy as np
from sklearn.metrics import f1_score


class DecisionEngine:
    def __init__(self, alpha=0.5, beta=0.2, gamma=0.3, delta=0.1, t_sanitize=0.4, t_block=0.7):
        """
        Weights for Risk Aggregation:
        alpha: Weight for ensemble_score
        beta: Weight for behavior_score (anomaly)
        gamma: Weight for rule_score
        delta: Penalty added if uncertainty is high
        
        Decision Thresholds:
        t_sanitize: Minimum risk to output SANITIZE
        t_block: Minimum risk to output BLOCK
        """
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.delta = delta
        
        self.t_sanitize = t_sanitize
        self.t_block = t_block

    def fit_thresholds(self, y_true, risk_scores, sanitize_quantile: float = 0.7):
        y_true = np.asarray(y_true)
        risk_scores = np.asarray(risk_scores)

        best_f1 = -1.0
        best_threshold = self.t_block

        for t_block in np.linspace(0.35, 0.95, 61):
            preds = (risk_scores >= t_block).astype(int)
            f1 = f1_score(y_true, preds, zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = float(t_block)

        self.t_block = best_threshold
        self.t_sanitize = float(np.quantile(risk_scores, sanitize_quantile))
        if self.t_sanitize >= self.t_block:
            self.t_sanitize = max(0.0, self.t_block - 0.1)

        return {
            "best_f1": best_f1,
            "t_block": self.t_block,
            "t_sanitize": self.t_sanitize,
        }

    def aggregate_and_decide(self, ensemble_score: float, behavior_score: float, rule_score: float, uncertainty: int):
        risk = (self.alpha * ensemble_score) + \
               (self.beta * behavior_score) + \
               (self.gamma * rule_score)
               
        if uncertainty == 1:
            risk += self.delta
            
        risk = min(risk, 1.0) # Cap at 1.0

        if risk >= self.t_block:
            decision = "BLOCK"
        elif risk >= self.t_sanitize:
            decision = "SANITIZE"
        else:
            decision = "ALLOW"
            
        return decision, risk

    def save(self, file_path: str):
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "alpha": self.alpha,
                    "beta": self.beta,
                    "gamma": self.gamma,
                    "delta": self.delta,
                    "t_sanitize": self.t_sanitize,
                    "t_block": self.t_block,
                },
                f,
                indent=2,
            )

    def load(self, file_path: str):
        with open(file_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        self.alpha = cfg["alpha"]
        self.beta = cfg["beta"]
        self.gamma = cfg["gamma"]
        self.delta = cfg["delta"]
        self.t_sanitize = cfg["t_sanitize"]
        self.t_block = cfg["t_block"]
