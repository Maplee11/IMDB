from __future__ import annotations

import csv
import random
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import torch
from torch.utils.data import DataLoader, Dataset

from config import (
    EVAL_BATCH_SIZE,
    EVAL_RATIO,
    MAX_SEQ_LEN,
    NUM_WORKERS,
    PREDICT_BATCH_SIZE,
    RANDOM_SEED,
    SAMPLE_SUBMISSION_FILE,
    TEST_FILE,
    TRAIN_BATCH_SIZE,
    TRAIN_FILE,
)
from datasets.base import DatasetBuilder, TaskDataloaders


def build_text(row: Dict[str, str]) -> str:
    parts: List[str] = []
    keyword = (row.get("keyword") or "").strip()
    location = (row.get("location") or "").strip()
    text = (row.get("text") or "").strip()

    if keyword:
        parts.append(f"keyword: {keyword}")
    if location:
        parts.append(f"location: {location}")
    if text:
        parts.append(f"text: {text}")

    return " | ".join(parts) if parts else ""


def read_train_rows(csv_path: Path) -> List[Tuple[str, int]]:
    rows: List[Tuple[str, int]] = []
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append((build_text(row), int(row["target"])))
    return rows


def read_test_rows(csv_path: Path) -> List[Tuple[str, str]]:
    rows: List[Tuple[str, str]] = []
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append((row["id"], build_text(row)))
    return rows


def read_submission_header(csv_path: Path) -> List[str]:
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        return next(reader)


def stratified_split(
    rows: Sequence[Tuple[str, int]],
    eval_ratio: float,
    seed: int,
) -> Tuple[List[Tuple[str, int]], List[Tuple[str, int]]]:
    positives = [row for row in rows if row[1] == 1]
    negatives = [row for row in rows if row[1] == 0]
    rng = random.Random(seed)
    rng.shuffle(positives)
    rng.shuffle(negatives)

    def split_one_group(group: List[Tuple[str, int]]) -> Tuple[List[Tuple[str, int]], List[Tuple[str, int]]]:
        eval_size = max(1, int(len(group) * eval_ratio))
        eval_group = group[:eval_size]
        train_group = group[eval_size:]
        if not train_group:
            raise ValueError("Train split is empty after stratified split.")
        return train_group, eval_group

    train_pos, eval_pos = split_one_group(positives)
    train_neg, eval_neg = split_one_group(negatives)

    train_rows = train_pos + train_neg
    eval_rows = eval_pos + eval_neg
    rng.shuffle(train_rows)
    rng.shuffle(eval_rows)
    return train_rows, eval_rows


class EncodedTextDataset(Dataset):
    def __init__(self, texts: Sequence[str], tokenizer, labels: Sequence[int] | None = None):
        self.texts = list(texts)
        self.labels = list(labels) if labels is not None else None
        self.encodings = tokenizer(
            self.texts,
            truncation=True,
            padding="max_length",
            max_length=MAX_SEQ_LEN,
            return_tensors="pt",
        )

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int):
        item = {
            "input_ids": self.encodings["input_ids"][idx],
            "attention_mask": self.encodings["attention_mask"][idx],
        }
        if self.labels is not None:
            item["labels"] = torch.tensor(float(self.labels[idx]), dtype=torch.float32)
        return item


class NLPDTDatasetBuilder(DatasetBuilder):
    name = "nlpdt"

    def __init__(self, root_dir: Path, tokenizer):
        self.root_dir = root_dir
        self.tokenizer = tokenizer

    def build(self) -> TaskDataloaders:
        train_path = self.root_dir / TRAIN_FILE
        test_path = self.root_dir / TEST_FILE
        sample_submission_path = self.root_dir / SAMPLE_SUBMISSION_FILE

        train_rows = read_train_rows(train_path)
        train_split, eval_split = stratified_split(train_rows, EVAL_RATIO, RANDOM_SEED)
        test_rows = read_test_rows(test_path)

        train_dataset = EncodedTextDataset(
            texts=[text for text, _ in train_split],
            labels=[label for _, label in train_split],
            tokenizer=self.tokenizer,
        )
        eval_dataset = EncodedTextDataset(
            texts=[text for text, _ in eval_split],
            labels=[label for _, label in eval_split],
            tokenizer=self.tokenizer,
        )
        test_dataset = EncodedTextDataset(
            texts=[text for _, text in test_rows],
            tokenizer=self.tokenizer,
        )

        generator = torch.Generator()
        generator.manual_seed(RANDOM_SEED)

        train_loader = DataLoader(
            train_dataset,
            batch_size=TRAIN_BATCH_SIZE,
            shuffle=True,
            num_workers=NUM_WORKERS,
            generator=generator,
        )
        eval_loader = DataLoader(
            eval_dataset,
            batch_size=EVAL_BATCH_SIZE,
            shuffle=False,
            num_workers=NUM_WORKERS,
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=PREDICT_BATCH_SIZE,
            shuffle=False,
            num_workers=NUM_WORKERS,
        )

        meta = {
            "dataset_name": self.name,
            "train_size": len(train_dataset),
            "eval_size": len(eval_dataset),
            "test_size": len(test_dataset),
            "test_ids": [sample_id for sample_id, _ in test_rows],
            "submission_header": read_submission_header(sample_submission_path),
        }
        return TaskDataloaders(train=train_loader, eval=eval_loader, test=test_loader, meta=meta)
