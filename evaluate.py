import argparse
import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from src.data.preprocessing import load_labeled_dataset, save_splits, stratified_split
from src.pipeline import PromptInjectionPipeline


sns.set_theme(style="whitegrid")


def compute_metrics(y_true, y_pred, y_score):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_score),
        "pr_auc": average_precision_score(y_true, y_score),
        "brier": brier_score_loss(y_true, np.clip(y_score, 0, 1)),
    }


def evaluate_models(pipeline: PromptInjectionPipeline, test_df: pd.DataFrame):
    texts = test_df["text"].tolist()
    y_true = test_df["label"].astype(int).to_numpy()

    embeddings = pipeline.roberta.embed(texts)

    roberta_scores = pipeline.roberta.predict_proba(texts)
    ml_scores = pipeline.ml_classifier.predict_proba(embeddings)
    ensemble_scores = np.asarray([
        pipeline.fusion.fuse(float(r1), float(r2))[0] for r1, r2 in zip(roberta_scores, ml_scores)
    ])

    behavior_scores = pipeline.behavior_detector.detect(embeddings).astype(float)
    rule_scores = np.asarray([pipeline.rule_engine.evaluate(t) for t in texts], dtype=float)
    uncertainty_scores = np.asarray([
        pipeline.fusion.fuse(float(r1), float(r2))[1] for r1, r2 in zip(roberta_scores, ml_scores)
    ], dtype=float)

    risk_scores = (
        pipeline.decision_engine.alpha * ensemble_scores
        + pipeline.decision_engine.beta * behavior_scores
        + pipeline.decision_engine.gamma * rule_scores
        + pipeline.decision_engine.delta * uncertainty_scores
    )
    risk_scores = np.clip(risk_scores, 0, 1)

    roberta_pred = (roberta_scores >= 0.5).astype(int)
    ml_pred = (ml_scores >= 0.5).astype(int)
    ensemble_pred = (ensemble_scores >= 0.5).astype(int)
    final_pred = (risk_scores >= pipeline.decision_engine.t_block).astype(int)

    records = {
        "roberta": (roberta_pred, roberta_scores),
        "ml": (ml_pred, ml_scores),
        "ensemble": (ensemble_pred, ensemble_scores),
        "final_risk": (final_pred, risk_scores),
    }

    metrics = []
    reports = {}
    for name, (pred, score) in records.items():
        row = compute_metrics(y_true, pred, score)
        row["model"] = name
        metrics.append(row)
        reports[name] = classification_report(y_true, pred, target_names=["benign", "malicious"], digits=4)

    predictions_df = test_df.copy()
    predictions_df["roberta_score"] = roberta_scores
    predictions_df["ml_score"] = ml_scores
    predictions_df["ensemble_score"] = ensemble_scores
    predictions_df["behavior_score"] = behavior_scores
    predictions_df["rule_score"] = rule_scores
    predictions_df["uncertainty"] = uncertainty_scores
    predictions_df["risk_score"] = risk_scores
    predictions_df["pred_roberta"] = roberta_pred
    predictions_df["pred_ml"] = ml_pred
    predictions_df["pred_ensemble"] = ensemble_pred
    predictions_df["pred_final"] = final_pred

    return pd.DataFrame(metrics), reports, predictions_df, records


def plot_confusion_matrices(output_dir: str, y_true: np.ndarray, records: dict):
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()

    for idx, (name, (pred, _)) in enumerate(records.items()):
        cm = confusion_matrix(y_true, pred)
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            cbar=False,
            ax=axes[idx],
            xticklabels=["benign", "malicious"],
            yticklabels=["benign", "malicious"],
        )
        axes[idx].set_title(f"Confusion Matrix: {name}")
        axes[idx].set_xlabel("Predicted")
        axes[idx].set_ylabel("Actual")

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "confusion_matrices.png"), dpi=200)
    plt.close()


def plot_roc_pr_curves(output_dir: str, y_true: np.ndarray, records: dict):
    plt.figure(figsize=(8, 6))
    for name, (_, scores) in records.items():
        fpr, tpr, _ = roc_curve(y_true, scores)
        auc = roc_auc_score(y_true, scores)
        plt.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "roc_curves.png"), dpi=200)
    plt.close()

    plt.figure(figsize=(8, 6))
    for name, (_, scores) in records.items():
        precision, recall, _ = precision_recall_curve(y_true, scores)
        ap = average_precision_score(y_true, scores)
        plt.plot(recall, precision, label=f"{name} (AP={ap:.3f})")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curves")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "pr_curves.png"), dpi=200)
    plt.close()


def plot_score_distribution(output_dir: str, predictions_df: pd.DataFrame):
    plt.figure(figsize=(8, 6))
    sns.kdeplot(
        data=predictions_df,
        x="risk_score",
        hue="label",
        fill=True,
        common_norm=False,
        palette={0: "#2B6CB0", 1: "#C53030"},
    )
    plt.title("Final Risk Score Distribution by Class")
    plt.xlabel("Risk Score")
    plt.ylabel("Density")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "risk_distribution.png"), dpi=200)
    plt.close()


def plot_metrics_bar(output_dir: str, metrics_df: pd.DataFrame):
    long_df = metrics_df.melt(id_vars=["model"], value_vars=["accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc"])
    plt.figure(figsize=(10, 6))
    sns.barplot(data=long_df, x="model", y="value", hue="variable")
    plt.ylim(0, 1)
    plt.title("Model Comparison Across Metrics")
    plt.ylabel("Score")
    plt.xlabel("Model")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "metrics_comparison.png"), dpi=220)
    plt.close()


def per_source_breakdown(predictions_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for source, group in predictions_df.groupby("source_file"):
        y_true = group["label"].to_numpy()
        y_pred = group["pred_final"].to_numpy()
        rows.append(
            {
                "source_file": source,
                "samples": len(group),
                "accuracy": accuracy_score(y_true, y_pred),
                "precision": precision_score(y_true, y_pred, zero_division=0),
                "recall": recall_score(y_true, y_pred, zero_division=0),
                "f1": f1_score(y_true, y_pred, zero_division=0),
            }
        )
    return pd.DataFrame(rows).sort_values("f1", ascending=False)


def main(args):
    os.makedirs(args.output_dir, exist_ok=True)

    device = "cuda" if getattr(args, "use_cuda", False) else "cpu"
    pipeline = PromptInjectionPipeline(device=device)
    pipeline.load(args.model_dir)

    if args.test_csv and os.path.exists(args.test_csv):
        test_df = pd.read_csv(args.test_csv)
    else:
        dataset = load_labeled_dataset(args.benign_dir, args.malicious_dir, deduplicate=True)
        splits = stratified_split(dataset, random_state=args.random_state)
        split_dir = os.path.join(args.output_dir, "splits")
        save_splits(splits, split_dir)
        test_df = splits.test

    metrics_df, reports, predictions_df, records = evaluate_models(pipeline, test_df)

    metrics_path = os.path.join(args.output_dir, "metrics_comparison.csv")
    metrics_df.to_csv(metrics_path, index=False)

    y_true = test_df["label"].astype(int).to_numpy()
    plot_confusion_matrices(args.output_dir, y_true, records)
    plot_roc_pr_curves(args.output_dir, y_true, records)
    plot_score_distribution(args.output_dir, predictions_df)
    plot_metrics_bar(args.output_dir, metrics_df)

    source_df = per_source_breakdown(predictions_df)
    source_df.to_csv(os.path.join(args.output_dir, "per_source_metrics.csv"), index=False)

    fp = predictions_df[(predictions_df["label"] == 0) & (predictions_df["pred_final"] == 1)]
    fn = predictions_df[(predictions_df["label"] == 1) & (predictions_df["pred_final"] == 0)]
    fp.sort_values("risk_score", ascending=False).head(50).to_csv(
        os.path.join(args.output_dir, "top_false_positives.csv"),
        index=False,
    )
    fn.sort_values("risk_score", ascending=True).head(50).to_csv(
        os.path.join(args.output_dir, "top_false_negatives.csv"),
        index=False,
    )

    examples = pd.concat(
        [
            predictions_df[(predictions_df["label"] == 1) & (predictions_df["pred_final"] == 1)].head(25),
            predictions_df[(predictions_df["label"] == 0) & (predictions_df["pred_final"] == 0)].head(25),
            fp.head(25),
            fn.head(25),
        ],
        ignore_index=True,
    )
    examples.to_csv(os.path.join(args.output_dir, "qualitative_examples.csv"), index=False)

    with open(os.path.join(args.output_dir, "classification_reports.txt"), "w", encoding="utf-8") as f:
        for name, report in reports.items():
            f.write(f"[{name}]\n")
            f.write(report)
            f.write("\n\n")

    summary = {
        "n_test": int(len(test_df)),
        "best_model_by_f1": metrics_df.sort_values("f1", ascending=False).iloc[0].to_dict(),
        "decision_thresholds": {
            "t_block": pipeline.decision_engine.t_block,
            "t_sanitize": pipeline.decision_engine.t_sanitize,
            "alpha": pipeline.decision_engine.alpha,
            "beta": pipeline.decision_engine.beta,
            "gamma": pipeline.decision_engine.gamma,
            "delta": pipeline.decision_engine.delta,
            "fusion_w1": pipeline.fusion.w1,
            "fusion_w2": pipeline.fusion.w2,
        },
    }

    with open(os.path.join(args.output_dir, "evaluation_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    predictions_df.to_csv(os.path.join(args.output_dir, "test_predictions.csv"), index=False)

    print("Evaluation complete.")
    print(f"Saved metrics and visualizations to: {args.output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate prompt injection ensemble pipeline.")
    parser.add_argument("--model-dir", type=str, default="artifacts/models")
    parser.add_argument("--output-dir", type=str, default="artifacts/evaluation")
    parser.add_argument("--test-csv", type=str, default="")

    parser.add_argument("--benign-dir", type=str, default="Benign data")
    parser.add_argument("--malicious-dir", type=str, default="data")
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--use-cuda", action="store_true", help="Use GPU for evaluation")

    args = parser.parse_args()
    main(args)
