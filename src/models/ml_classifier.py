import numpy as np
from joblib import dump, load
from xgboost import XGBClassifier

class MLClassifierWrapper:
    def __init__(self):
        self.model = XGBClassifier(
            use_label_encoder=False,
            eval_metric='logloss',
            n_estimators=300,
            max_depth=6,
            learning_rate=0.08,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=42
        )
        self.is_trained = False

    def train(self, X, y):
        self.model.fit(X, y)
        self.is_trained = True

    def predict_proba(self, X):
        if not self.is_trained:
            # Random fallback if not trained
            return np.random.uniform(0, 1, size=len(X))
        probs = self.model.predict_proba(X)
        output = probs[:, 1] if probs.shape[1] > 1 else probs[:, 0]
        return output

    def save(self, path: str):
        dump(self.model, path)

    def load(self, path: str):
        self.model = load(path)
        self.is_trained = True
