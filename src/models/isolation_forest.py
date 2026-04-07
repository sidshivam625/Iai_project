from sklearn.ensemble import IsolationForest
import numpy as np
from joblib import dump, load

class SequenceAnomalyDetector:
    def __init__(self, contamination=0.1):
        self.iso_forest = IsolationForest(contamination=contamination, random_state=42)
        self.is_trained = False

    def train(self, historical_embeddings):
        if len(historical_embeddings) > 0:
            self.iso_forest.fit(historical_embeddings)
            self.is_trained = True

    def detect(self, current_embedding, history_embeddings=None):
        if not self.is_trained:
            # Assume normal if no baseline exists
            return np.zeros(len(current_embedding), dtype=int)
        
        # In a real environment, we would refit or check context dynamically
        # ISO forest outputs -1 for anomalous, 1 for normal
        pred = self.iso_forest.predict(current_embedding)
        
        # Map to behavioral risk: 1 if anomaly (-1), 0 if normal (1)
        risk = np.where(pred == -1, 1, 0)
        return risk

    def anomaly_score(self, current_embedding):
        if not self.is_trained:
            return np.zeros(len(current_embedding), dtype=float)
        raw = self.iso_forest.decision_function(current_embedding)
        return -raw

    def save(self, path: str):
        dump(self.iso_forest, path)

    def load(self, path: str):
        self.iso_forest = load(path)
        self.is_trained = True
