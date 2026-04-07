# Prompt Injection Detector: Complete Project Guide

This document is the full end-to-end setup and execution guide for the research project.

## 1. Project Goal

Build and evaluate a multi-layer ensemble system for prompt injection detection using:
1. RoBERTa semantic classifier
2. XGBoost statistical classifier
3. Weighted fusion with uncertainty
4. Isolation Forest behavioral anomaly detector
5. Rule-based knowledge layer
6. Final risk aggregation engine with ALLOW / SANITIZE / BLOCK outputs

## 2. Folder Structure

Expected workspace:
- `prompt_injection_detector/`
- `prompt_injection_detector/data/` (malicious CSV files)
- `prompt_injection_detector/Benign data/` (benign CSV files)
- `prompt_injection_detector/src/`

Main scripts:
- `train.py`
- `evaluate.py`
- `test_terminal.py`

Core pipeline modules:
- `src/data/preprocessing.py`
- `src/pipeline.py`
- `src/models/roberta_classifier.py`
- `src/models/ml_classifier.py`
- `src/models/isolation_forest.py`
- `src/models/rule_engine.py`
- `src/ensemble/fusion.py`
- `src/ensemble/decision.py`

## 3. Environment Setup (Windows)

Run these commands in PowerShell from project root (`prompt_injection_detector`).

### 3.1 Create virtual environment
```powershell
python -m venv .venv
```

### 3.2 Activate virtual environment
```powershell
.\.venv\Scripts\Activate.ps1
```

If execution policy blocks activation:
```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

### 3.3 Upgrade packaging tools
```powershell
python -m pip install --upgrade pip setuptools wheel
```

### 3.4 Install project dependencies
```powershell
pip install -r requirements.txt
```

## 4. Data Ingestion and Preprocessing

Handled by `src/data/preprocessing.py`.

What it does automatically:
1. Loads all CSV files from benign and malicious directories.
2. Detects prompt text column from candidate names:
   - `Prompt`, `Text`, `MutatedPrompt`, `prompt`, `text`, `query`, `input`
3. Removes auto-generated unnamed index columns.
4. Normalizes text (newline cleanup, whitespace normalization, trimming).
5. Drops empty text samples.
6. Assigns labels:
   - benign = `0`
   - malicious = `1`
7. Deduplicates by text (enabled by default).
8. Creates stratified train/val/test split (70/15/15).
9. Saves split files.

## 5. Training Workflow

Training command (CPU):
```powershell
python train.py --epochs 2 --batch-size 16 --output-dir artifacts
```

Training command (GPU, if CUDA available):
```powershell
python train.py --epochs 2 --batch-size 16 --output-dir artifacts --use-cuda
```

### 5.1 What gets trained
1. RoBERTa classifier: trained on both benign and malicious samples.
2. XGBoost classifier: trained on RoBERTa embeddings from both classes.
3. Isolation Forest: trained on benign embeddings when enough benign samples exist (fallback to all train embeddings otherwise).
4. Fusion weights (`w1`, `w2`): tuned on validation set for best F1.
5. Decision thresholds (`T_block`, `T_sanitize`): calibrated on validation risk scores.

### 5.2 Training artifacts generated
Under `artifacts/`:
- `models/`
- `models/roberta/`
- `models/ml_classifier.joblib`
- `models/isolation_forest.joblib`
- `models/fusion_config.json`
- `models/decision_config.json`
- `splits/train.csv`
- `splits/val.csv`
- `splits/test.csv`
- `dataset_summary.csv`
- `training_metadata.json`

## 6. Evaluation Workflow

Run evaluation:
```powershell
python evaluate.py --model-dir artifacts/models --output-dir artifacts/evaluation
```

### 6.1 Metrics computed
For each variant (`roberta`, `ml`, `ensemble`, `final_risk`):
1. Accuracy
2. Precision
3. Recall
4. F1-score
5. ROC-AUC
6. PR-AUC
7. Brier score

### 6.2 Evaluation artifacts generated
Under `artifacts/evaluation/`:
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

## 7. Interactive Inference

Run:
```powershell
python test_terminal.py --model-dir artifacts/models
```

Type prompt examples and see:
- Final decision
- Risk score
- Component-level scores (`roberta_prob`, `ml_prob`, `rule_score`, etc.)

## 8. Hyperparameters and Defaults

### RoBERTa (`src/models/roberta_classifier.py`)
- Base model: `roberta-base`
- `epochs`: default `2`
- `batch_size`: default `16`
- `max_length`: default `256`
- `learning_rate`: default `2e-5`
- Optimizer: `AdamW`

### XGBoost (`src/models/ml_classifier.py`)
- `n_estimators=300`
- `max_depth=6`
- `learning_rate=0.08`
- `subsample=0.9`
- `colsample_bytree=0.9`
- `random_state=42`

### Isolation Forest (`src/models/isolation_forest.py`)
- `contamination=0.1`

### Fusion (`src/ensemble/fusion.py`)
- Initial `w1=0.6`, `w2=0.4`, `tau=0.3`
- Weights tuned on validation data

### Decision (`src/ensemble/decision.py`)
- `alpha=0.5`
- `beta=0.2`
- `gamma=0.3`
- `delta=0.1`
- Thresholds tuned on validation data

## 9. Reproducibility Protocol

1. Keep random seed fixed (default: `42`).
2. Keep saved split files (`artifacts/splits/*.csv`) for all experiments.
3. Always evaluate on test split only.
4. Do not tune thresholds on test data.
5. Save full training metadata and model configs per run.

## 10. Suggested Commands for Clean Experiment Cycle

From project root with venv activated:

```powershell
Remove-Item -Recurse -Force artifacts -ErrorAction SilentlyContinue
python train.py --epochs 3 --batch-size 16 --output-dir artifacts
python evaluate.py --model-dir artifacts/models --output-dir artifacts/evaluation
python test_terminal.py --model-dir artifacts/models
```

## 11. Paper Reporting Checklist

### Methodology section
1. Data sources and label construction
2. Preprocessing and split strategy
3. Full architecture and layer roles
4. Training objective and calibration process

### Experiments section
1. Hyperparameter table
2. Main metrics table (all model variants)
3. Ablation table (`roberta`, `ml`, `ensemble`, `final_risk`)
4. Per-source performance table
5. Error analysis table (top FP/FN examples)

### Figures
1. Pipeline architecture diagram
2. Confusion matrices
3. ROC curves
4. PR curves
5. Risk distribution plot
6. Metrics comparison bar chart

## 12. Troubleshooting

### Issue: `Import ... could not be resolved`
Cause: dependencies not installed in active interpreter.
Fix:
```powershell
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### Issue: RoBERTa download fails
Cause: internet/proxy restrictions.
Fix:
1. Ensure internet connectivity.
2. Retry command.
3. Optionally pre-download model with Hugging Face cache configured.

### Issue: CUDA not used
Cause: CUDA unavailable or PyTorch CPU build.
Fix:
1. Verify GPU support.
2. Install CUDA-enabled PyTorch.
3. Re-run with `--use-cuda`.

### Issue: out-of-memory
Fix:
1. Reduce `--batch-size`.
2. Reduce `--max-length`.
3. Use CPU for stability if GPU memory is limited.

## 13. Minimal Quick Start

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python train.py --epochs 2 --batch-size 16 --output-dir artifacts
python evaluate.py --model-dir artifacts/models --output-dir artifacts/evaluation
python test_terminal.py --model-dir artifacts/models
```
