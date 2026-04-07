from src.models.roberta_classifier import RobertaWrapper
from src.models.ml_classifier import MLClassifierWrapper
from src.models.isolation_forest import SequenceAnomalyDetector
from src.models.rule_engine import KnowledgeRuleEngine
from src.ensemble.fusion import EnsembleFusion
from src.ensemble.decision import DecisionEngine
import numpy as np
from pathlib import Path
import json

class PromptInjectionPipeline:
    def __init__(self, device: str = 'cpu'):
        # 1. Representation & Deep Semantic Model
        self.roberta = RobertaWrapper(device=device)
        
        # 2. ML Classifier
        self.ml_classifier = MLClassifierWrapper()
        
        # 3. Decision Fusion
        self.fusion = EnsembleFusion(w1=0.6, w2=0.4, tau=0.3)
        
        # 4. Behavioral Anomaly
        self.behavior_detector = SequenceAnomalyDetector(contamination=0.1)
        self.history_embeddings = []
        
        # 5. Rule Engine
        self.rule_engine = KnowledgeRuleEngine()
        
        # 6. Final Decision Engine
        self.decision_engine = DecisionEngine()

    def train(self, train_texts, train_labels, val_texts, val_labels):
        print("Training RoBERTa semantic model...")
        train_history = self.roberta.fit(train_texts, train_labels, val_texts, val_labels)

        print("Extracting contextual embeddings...")
        train_embeddings = self.roberta.embed(train_texts)
        val_embeddings = self.roberta.embed(val_texts)

        print("Training ML classifier on embeddings...")
        self.ml_classifier.train(train_embeddings, train_labels)

        print("Training sequence anomaly detector using benign context...")
        train_labels_arr = np.asarray(train_labels)
        benign_embeddings = train_embeddings[train_labels_arr == 0]
        if len(benign_embeddings) > 10:
            self.behavior_detector.train(benign_embeddings)
        else:
            self.behavior_detector.train(train_embeddings)

        print("Calibrating fusion weights and decision thresholds...")
        r1_val = self.roberta.predict_proba(val_texts)
        r2_val = self.ml_classifier.predict_proba(val_embeddings)
        fusion_info = self.fusion.fit_weights(val_labels, r1_val, r2_val)

        ensemble_val = np.asarray([self.fusion.fuse(float(a), float(b))[0] for a, b in zip(r1_val, r2_val)])
        behavior_val = self.behavior_detector.detect(val_embeddings).astype(float)
        rule_val = np.asarray([self.rule_engine.evaluate(x) for x in val_texts], dtype=float)
        uncertainty_val = np.asarray([self.fusion.fuse(float(a), float(b))[1] for a, b in zip(r1_val, r2_val)], dtype=float)

        risk_scores = (
            self.decision_engine.alpha * ensemble_val
            + self.decision_engine.beta * behavior_val
            + self.decision_engine.gamma * rule_val
            + self.decision_engine.delta * uncertainty_val
        )
        threshold_info = self.decision_engine.fit_thresholds(val_labels, risk_scores)

        return {
            "roberta_train_history": train_history,
            "fusion": fusion_info,
            "thresholds": threshold_info,
        }

    def analyze_prompt(self, prompt: str):
        # Layer 1 & 2: Representation & Extract Probs
        embedding = self.roberta.embed([prompt])
        
        r1 = self.roberta.predict_proba([prompt])[0]
        r2 = self.ml_classifier.predict_proba(embedding)[0]
        
        # Layer 3: Ensemble Fusion
        ensemble_score, uncertainty = self.fusion.fuse(r1, r2)
        
        # Layer 4: Behavioral Analysis
        behavior_score = self.behavior_detector.detect(embedding)[0]
        self.history_embeddings.append(embedding[0]) # update context
        
        # Layer 5: Knowledge Rule Evaluator
        rule_score = self.rule_engine.evaluate(prompt)
        
        # Layer 6: Risk Aggregation
        decision, risk_score = self.decision_engine.aggregate_and_decide(
            ensemble_score, behavior_score, rule_score, uncertainty
        )
        
        return {
            "prompt": prompt,
            "decision": decision,
            "risk_score": risk_score,
            "breakdown": {
                "roberta_prob": r1,
                "ml_prob": r2,
                "fusion_score": ensemble_score,
                "uncertainty": uncertainty,
                "behavior_score": behavior_score,
                "rule_score": rule_score
            }
        }

    def save(self, output_dir: str):
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        roberta_dir = out / "roberta"
        self.roberta.save(str(roberta_dir))
        self.ml_classifier.save(str(out / "ml_classifier.joblib"))
        self.behavior_detector.save(str(out / "isolation_forest.joblib"))
        self.decision_engine.save(str(out / "decision_config.json"))

        with open(out / "fusion_config.json", "w", encoding="utf-8") as f:
            json.dump({"w1": self.fusion.w1, "w2": self.fusion.w2, "tau": self.fusion.tau}, f, indent=2)

    def load(self, model_dir: str):
        root = Path(model_dir)
        self.roberta.load(str(root / "roberta"))
        self.ml_classifier.load(str(root / "ml_classifier.joblib"))
        self.behavior_detector.load(str(root / "isolation_forest.joblib"))
        self.decision_engine.load(str(root / "decision_config.json"))

        with open(root / "fusion_config.json", "r", encoding="utf-8") as f:
            cfg = json.load(f)
        self.fusion.w1 = cfg["w1"]
        self.fusion.w2 = cfg["w2"]
        self.fusion.tau = cfg["tau"]
