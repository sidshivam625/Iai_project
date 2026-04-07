import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import pandas as pd
from sklearn.model_selection import train_test_split


TEXT_COLUMN_CANDIDATES = [
    "Prompt",
    "prompt",
    "Text",
    "text",
    "MutatedPrompt",
    "mutated_prompt",
    "query",
    "input",
]


@dataclass
class DatasetSplits:
    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame


def normalize_text(text: str) -> str:
    text = str(text)
    text = text.replace("\r", " ").replace("\n", " ")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _drop_unnamed_columns(df: pd.DataFrame) -> pd.DataFrame:
    keep_columns = []
    for col in df.columns:
        col_str = str(col)
        if col_str.startswith("Unnamed"):
            continue
        if re.fullmatch(r"H\d+", col_str):
            continue
        if col_str.strip() == "":
            continue
        keep_columns.append(col)
    return df[keep_columns]


def _select_text_column(df: pd.DataFrame) -> Optional[str]:
    for candidate in TEXT_COLUMN_CANDIDATES:
        if candidate in df.columns:
            return candidate
    return None


def _read_single_csv(file_path: Path, label: int, group_name: str) -> pd.DataFrame:
    df = pd.read_csv(file_path, on_bad_lines="skip", encoding_errors="ignore")
    df = _drop_unnamed_columns(df)

    text_column = _select_text_column(df)
    if text_column is None:
        return pd.DataFrame(columns=["text", "label", "source_file", "source_group"])

    out = pd.DataFrame()
    out["text"] = df[text_column].astype(str).map(normalize_text)
    out = out[out["text"].str.len() > 0]
    out["label"] = label
    out["source_file"] = file_path.name
    out["source_group"] = group_name
    out["char_len"] = out["text"].str.len()
    out["word_len"] = out["text"].str.split().str.len()
    return out


def load_labeled_dataset(benign_dir: str, malicious_dir: str, deduplicate: bool = True) -> pd.DataFrame:
    benign_root = Path(benign_dir)
    malicious_root = Path(malicious_dir)

    benign_frames: List[pd.DataFrame] = []
    malicious_frames: List[pd.DataFrame] = []

    for csv_file in sorted(benign_root.glob("*.csv")):
        benign_frames.append(_read_single_csv(csv_file, label=0, group_name="benign"))

    for csv_file in sorted(malicious_root.glob("*.csv")):
        malicious_frames.append(_read_single_csv(csv_file, label=1, group_name="malicious"))

    dataset = pd.concat(benign_frames + malicious_frames, ignore_index=True)

    dataset = dataset.dropna(subset=["text"])
    dataset["text"] = dataset["text"].astype(str).map(normalize_text)
    dataset = dataset[dataset["text"].str.len() > 0]

    if deduplicate:
        dataset = dataset.drop_duplicates(subset=["text"]).reset_index(drop=True)

    return dataset


def stratified_split(
    dataset: pd.DataFrame,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    random_state: int = 42,
) -> DatasetSplits:
    total = train_ratio + val_ratio + test_ratio
    if abs(total - 1.0) > 1e-6:
        raise ValueError("train_ratio + val_ratio + test_ratio must equal 1.0")

    train_df, temp_df = train_test_split(
        dataset,
        test_size=(1.0 - train_ratio),
        stratify=dataset["label"],
        random_state=random_state,
    )

    relative_test_ratio = test_ratio / (val_ratio + test_ratio)
    val_df, test_df = train_test_split(
        temp_df,
        test_size=relative_test_ratio,
        stratify=temp_df["label"],
        random_state=random_state,
    )

    return DatasetSplits(
        train=train_df.reset_index(drop=True),
        val=val_df.reset_index(drop=True),
        test=test_df.reset_index(drop=True),
    )


def save_splits(splits: DatasetSplits, output_dir: str) -> Dict[str, str]:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    train_path = out / "train.csv"
    val_path = out / "val.csv"
    test_path = out / "test.csv"

    splits.train.to_csv(train_path, index=False)
    splits.val.to_csv(val_path, index=False)
    splits.test.to_csv(test_path, index=False)

    return {
        "train": str(train_path),
        "val": str(val_path),
        "test": str(test_path),
    }


def dataset_summary(df: pd.DataFrame) -> pd.DataFrame:
    summary = (
        df.groupby(["source_group", "source_file", "label"]) 
        .size()
        .reset_index(name="samples")
        .sort_values(["source_group", "samples"], ascending=[True, False])
    )
    return summary
