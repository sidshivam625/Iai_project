import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.preprocessing import dataset_summary, load_labeled_dataset, save_splits, stratified_split
from src.pipeline import PromptInjectionPipeline


def select_device(use_cuda: bool) -> str:
    if not use_cuda:
        return "cpu"
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def train_system(args):
    dataset = load_labeled_dataset(args.benign_dir, args.malicious_dir, deduplicate=not args.keep_duplicates)
    splits = stratified_split(
        dataset,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        random_state=args.random_state,
    )

    os.makedirs(args.output_dir, exist_ok=True)
    split_paths = save_splits(splits, os.path.join(args.output_dir, "splits"))

    summary = dataset_summary(dataset)
    summary_path = os.path.join(args.output_dir, "dataset_summary.csv")
    summary.to_csv(summary_path, index=False)

    device = select_device(args.use_cuda)
    pipeline = PromptInjectionPipeline(device=device)

    pipeline.roberta.epochs = args.epochs
    pipeline.roberta.batch_size = args.batch_size
    pipeline.roberta.max_length = args.max_length
    pipeline.roberta.learning_rate = args.learning_rate

    train_texts = splits.train["text"].tolist()
    train_labels = splits.train["label"].astype(int).tolist()
    val_texts = splits.val["text"].tolist()
    val_labels = splits.val["label"].astype(int).tolist()

    training_info = pipeline.train(train_texts, train_labels, val_texts, val_labels)

    model_dir = os.path.join(args.output_dir, "models")
    pipeline.save(model_dir)

    metadata = {
        "device": device,
        "n_total": int(len(dataset)),
        "n_train": int(len(splits.train)),
        "n_val": int(len(splits.val)),
        "n_test": int(len(splits.test)),
        "split_paths": split_paths,
        "model_dir": model_dir,
        "train_args": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "max_length": args.max_length,
            "learning_rate": args.learning_rate,
            "random_state": args.random_state,
        },
        "training_info": training_info,
    }

    metadata_path = os.path.join(args.output_dir, "training_metadata.json")
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print("Training complete.")
    print(f"Dataset size: {len(dataset)}")
    print(f"Train/Val/Test: {len(splits.train)}/{len(splits.val)}/{len(splits.test)}")
    print(f"Artifacts saved under: {args.output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train prompt injection detection ensemble pipeline.")
    parser.add_argument("--benign-dir", type=str, default="Benign data")
    parser.add_argument("--malicious-dir", type=str, default="data")
    parser.add_argument("--output-dir", type=str, default="artifacts")

    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)

    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--random-state", type=int, default=42)

    parser.add_argument("--use-cuda", action="store_true")
    parser.add_argument("--keep-duplicates", action="store_true")

    args = parser.parse_args()
    train_system(args)
