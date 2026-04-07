from pathlib import Path
from typing import List, Sequence

import numpy as np
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer


class PromptDataset(Dataset):
    def __init__(self, texts: Sequence[str], labels: Sequence[int], tokenizer, max_length: int):
        print(f"Pre-tokenizing {len(texts)} samples...")
        # Batch tokenization is much faster than one-by-one inside __getitem__
        self.encodings = tokenizer(
            list(texts),
            truncation=True,
            padding="max_length",
            max_length=max_length,
            return_tensors="pt"
        )
        self.labels = torch.tensor(list(labels), dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "input_ids": self.encodings["input_ids"][idx],
            "attention_mask": self.encodings["attention_mask"][idx],
            "labels": self.labels[idx],
        }


class RobertaWrapper:
    def __init__(
        self,
        model_name: str = "roberta-base",
        device: str = "cpu",
        learning_rate: float = 2e-5,
        max_length: int = 256,
        batch_size: int = 16,
        epochs: int = 2,
    ):
        self.model_name = model_name
        self.device = torch.device(device)
        self.learning_rate = learning_rate
        self.max_length = max_length
        self.batch_size = batch_size
        self.epochs = epochs

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2).to(self.device)
        self.is_trained = False

    def _to_loader(self, texts: Sequence[str], labels: Sequence[int], shuffle: bool) -> DataLoader:
        dataset = PromptDataset(texts, labels, self.tokenizer, self.max_length)
        # pin_memory=True speeds up transfer to GPU
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=shuffle, pin_memory=(self.device.type == "cuda"))

    def fit(
        self,
        train_texts: Sequence[str],
        train_labels: Sequence[int],
        val_texts: Sequence[str],
        val_labels: Sequence[int],
    ) -> List[dict]:
        optimizer = AdamW(self.model.parameters(), lr=self.learning_rate)

        train_loader = self._to_loader(train_texts, train_labels, shuffle=True)
        val_loader = self._to_loader(val_texts, val_labels, shuffle=False)

        history = []
        best_val_loss = float("inf")
        best_state = None

        for epoch in range(self.epochs):
            self.model.train()
            train_loss = 0.0

            for batch in tqdm(train_loader, desc=f"RoBERTa Train Epoch {epoch + 1}/{self.epochs}", leave=False):
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)

                optimizer.zero_grad()
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
                loss.backward()
                optimizer.step()

                train_loss += loss.item()

            train_loss /= max(len(train_loader), 1)
            val_loss = self._validate_loss(val_loader)
            history.append({"epoch": epoch + 1, "train_loss": train_loss, "val_loss": val_loss})

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()}

        if best_state is not None:
            self.model.load_state_dict(best_state)

        self.is_trained = True
        return history

    def _validate_loss(self, val_loader: DataLoader) -> float:
        self.model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)

                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                val_loss += outputs.loss.item()

        return val_loss / max(len(val_loader), 1)

    def predict_proba(self, texts: Sequence[str]) -> np.ndarray:
        self.model.eval()
        probs = []

        with torch.no_grad():
            for start in tqdm(range(0, len(texts), self.batch_size), desc="Predicting Probabilities"):
                chunk = list(texts[start:start + self.batch_size])
                encoded = self.tokenizer(
                    chunk,
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt",
                ).to(self.device)

                outputs = self.model(**encoded)
                batch_probs = torch.softmax(outputs.logits, dim=-1)[:, 1]
                probs.extend(batch_probs.detach().cpu().numpy().tolist())

        return np.asarray(probs, dtype=float)

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        self.model.eval()
        vectors = []

        with torch.no_grad():
            for start in tqdm(range(0, len(texts), self.batch_size), desc="Extracting Embeddings"):
                chunk = list(texts[start:start + self.batch_size])
                encoded = self.tokenizer(
                    chunk,
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt",
                ).to(self.device)

                outputs = self.model(
                    input_ids=encoded["input_ids"],
                    attention_mask=encoded["attention_mask"],
                    output_hidden_states=True,
                )
                cls_embedding = outputs.hidden_states[-1][:, 0, :]
                vectors.append(cls_embedding.detach().cpu().numpy())

        if not vectors:
            return np.zeros((0, self.model.config.hidden_size), dtype=float)

        return np.vstack(vectors)

    def save(self, output_dir: str) -> None:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(out)
        self.tokenizer.save_pretrained(out)

    def load(self, model_dir: str) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_dir).to(self.device)
        self.is_trained = True
