# Prompt Injection Detector: Multi-Layer Ensemble Framework

For a full end-to-end setup and execution manual, see `PROJECT_GUIDE.md`.

This repository implements a research-oriented pipeline for prompt injection detection in LLM systems using:
- Semantic classifier: fine-tuned RoBERTa
- Statistical classifier: XGBoost over contextual embeddings
- Fusion layer: weighted ensemble with uncertainty signal
- Behavioral layer: Isolation Forest sequence anomaly detector
- Rule layer: interpretable regex-based safety rules
- Risk engine: calibrated decision into ALLOW / SANITIZE / BLOCK

## 1) Dataset Processing (What is done)

### Input folders
- Malicious prompts: `data/`
- Benign prompts: `Benign data/`

### Preprocessing steps
Implemented in `src/data/preprocessing.py`:
1. Load every CSV in both folders.
2. Auto-detect prompt text columns from candidates:
   - Prompt, Text, MutatedPrompt, prompt, text, query, input
3. Remove unnamed/index-only columns (for inconsistent exported CSV schemas).
4. Normalize text:
   - newline removal
   - whitespace normalization
   - trim
5. Remove empty rows.
6. Label assignment:
   - benign files -> label 0
   - malicious files -> label 1
7. Deduplicate by prompt text (default on).
8. Stratified split into Train/Val/Test = 70/15/15.
9. Save split CSV files and a source-level dataset summary.

### Saved split artifacts
After training, these are saved under `artifacts/splits/`:
- `train.csv`
- `val.csv`
- `test.csv`

## 2) Model Pipeline (What is done in classification)

Implemented in `src/pipeline.py` with 5 layers:

1. Representation + Semantic Classification
   - `src/models/roberta_classifier.py`
   - Fine-tunes `roberta-base` on binary malicious/benign labels.
   - Produces probability: `r1 = P(malicious | prompt)`.

2. Statistical Backup Classifier
   - `src/models/ml_classifier.py`
   - Uses RoBERTa [CLS] embeddings.
   - Trains XGBoost classifier.
   - Produces probability: `r2 = P(malicious | embedding)`.

3. Fusion + Uncertainty
   - `src/ensemble/fusion.py`
   - Weighted score: `ensemble = w1 * r1 + w2 * r2`
   - Uncertainty flag when `|r1 - r2| > tau`
   - Weights are tuned on validation set for best F1.

4. Sequence Behavior Anomaly
   - `src/models/isolation_forest.py`
   - Trained mostly on benign train embeddings.
   - Returns anomaly risk signal for unusual prompt behavior.

5. Rule-Based Knowledge Layer
   - `src/models/rule_engine.py`
   - Pattern-based rules for known prompt injection instructions.

6. Risk Aggregation + Decision
   - `src/ensemble/decision.py`
   - Risk:
     - `risk = alpha * ensemble + beta * behavior + gamma * rule + delta * uncertainty`
   - Thresholds calibrated on validation data.
   - Final decision:
     - `BLOCK` if risk >= T_block
     - `SANITIZE` if T_sanitize <= risk < T_block
     - `ALLOW` otherwise

## 3) Training

## Environment
```bash
pip install -r requirements.txt
```

## Run training
```bash
python train.py --epochs 2 --batch-size 16 --output-dir artifacts
```

Optional CUDA:
```bash
python train.py --epochs 2 --batch-size 16 --output-dir artifacts --use-cuda
```

### Training outputs
- `artifacts/models/`:
  - RoBERTa fine-tuned weights
  - XGBoost model
  - Isolation Forest model
  - Fusion config
  - Decision config
- `artifacts/training_metadata.json`
- `artifacts/dataset_summary.csv`
- `artifacts/splits/*.csv`

## 4) Evaluation and Visualizations

Run evaluation:
```bash
python evaluate.py --model-dir artifacts/models --output-dir artifacts/evaluation
```

### Metrics computed
For each model variant (`roberta`, `ml`, `ensemble`, `final_risk`):
- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC
- PR-AUC
- Brier Score (calibration)

### Evaluation artifacts
Saved under `artifacts/evaluation/`:
- `metrics_comparison.csv`
- `classification_reports.txt`
- `test_predictions.csv`
- `per_source_metrics.csv`
- `top_false_positives.csv`
- `top_false_negatives.csv`
- `qualitative_examples.csv`
- `evaluation_summary.json`
- `confusion_matrices.png`
- `roc_curves.png`
- `pr_curves.png`
- `risk_distribution.png`
- `metrics_comparison.png`

## 5) Interactive Testing

```bash
python test_terminal.py --model-dir artifacts/models
```

Type prompts and get decision + score breakdown.

## 6) Parameters to Report in Paper

Include these in the Methodology / Experiments section:

### Data and split
- Number of total samples after deduplication
- Number of benign and malicious samples
- Per-source dataset counts
- Train/Val/Test split ratio and random seed

### RoBERTa
- Base model: `roberta-base`
- Max token length
- Batch size
- Learning rate
- Epochs
- Optimizer (AdamW)
- Device (CPU/GPU)

### XGBoost
- `n_estimators`
- `max_depth`
- `learning_rate`
- `subsample`
- `colsample_bytree`
- `random_state`

### Fusion and decision
- Fusion weights `w1, w2`
- Uncertainty threshold `tau`
- Risk weights `alpha, beta, gamma, delta`
- Decision thresholds `T_sanitize, T_block`

### Isolation Forest
- Contamination value
- Features used (RoBERTa embeddings)
- Training subset strategy (benign-focused baseline)

### Rule-based layer
- Number of active rules
- Rule weighting strategy

### Evaluation protocol
- Metrics list above
- Per-source performance variation
- Error analysis (false positives / false negatives)
- Ablation:
  - RoBERTa only
  - XGBoost only
  - Fusion only
  - Full system (Fusion + Behavior + Rules)

## 7) Suggested Tables/Figures for Report

Tables:
1. Dataset composition by source and label
2. Hyperparameter table for all components
3. Metrics comparison table across model variants
4. Per-source metrics table
5. Error-analysis examples table (FP/FN)

Figures:
1. Architecture diagram (pipeline flow)
2. Confusion matrix grid
3. ROC curve comparison
4. Precision-Recall curve comparison
5. Risk score distribution by class
6. Metric bar chart for model ablation

## 8) Reproducibility Notes

- Set random seeds in every run.
- Keep exact split files for publication reproducibility.
- Report whether deduplication was enabled.
- Always evaluate on held-out test split only.
- If model changes, retrain all components and regenerate all artifacts.
