import csv
import os
import random
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import swanlab
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from transformers import GPT2Tokenizer

from config import *

from transformer import BinaryClassifyModel


ROOT_DIR = Path(__file__).resolve().parent
DATASET_DIR = ROOT_DIR / DATASET_DIR
TRAIN_CSV_PATH = ROOT_DIR / TRAIN_CSV_PATH
TEST_CSV_PATH = ROOT_DIR / TEST_CSV_PATH
SAMPLE_SUBMISSION_PATH = ROOT_DIR / SAMPLE_SUBMISSION_PATH
PREDICT_CSV_PATH = ROOT_DIR / PREDICT_CSV_PATH
TOKENIZER_CACHE_DIR = ROOT_DIR / TOKENIZER_CACHE_DIR
CKPT_PATH = ROOT_DIR / CKPT_PATH
BEST_CKPT_PATH = ROOT_DIR / BEST_CKPT_PATH


torch.set_default_dtype(DEFAULT_DTYPE)
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
AMP_DTYPE = torch.bfloat16
USE_AMP = DEVICE.type == "mps"
SWANLAB_API_KEY = os.getenv("SWANLAB_API_KEY", "ZnWwDCFzy78QEM12FZXCr")


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

    return " | ".join(parts)


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
        header = next(reader)
    return header


def split_train_eval(
    samples: Sequence[Tuple[str, int]],
    eval_ratio: float,
    seed: int,
) -> Tuple[List[Tuple[str, int]], List[Tuple[str, int]]]:
    if not samples:
        raise ValueError("Training dataset is empty.")

    indices = list(range(len(samples)))
    random.Random(seed).shuffle(indices)

    eval_size = max(1, int(len(samples) * eval_ratio))
    eval_indices = set(indices[:eval_size])

    train_samples: List[Tuple[str, int]] = []
    eval_samples: List[Tuple[str, int]] = []
    for idx, sample in enumerate(samples):
        if idx in eval_indices:
            eval_samples.append(sample)
        else:
            train_samples.append(sample)

    if not train_samples:
        raise ValueError("Train split is empty after sampling.")
    if not eval_samples:
        raise ValueError("Eval split is empty after sampling.")

    return train_samples, eval_samples


def tokenize_texts(
    texts: Iterable[str],
    tokenizer: GPT2Tokenizer,
    max_seq_len: int,
    pooling_type: str,
) -> Tuple[List[List[int]], List[List[int]]]:
    input_ids_list: List[List[int]] = []
    attention_mask_list: List[List[int]] = []
    pooling_type = pooling_type.lower()

    if pooling_type == "cls":
        cls_id = tokenizer.cls_token_id
        pad_id = tokenizer.pad_token_id

    for text in tqdm(list(texts), desc="Tokenizing", leave=False):
        if pooling_type == "cls":
            tokens = tokenizer(
                text,
                truncation=True,
                max_length=max_seq_len - 1,
                padding=False,
                add_special_tokens=False,
            )
            input_ids = [cls_id] + tokens["input_ids"]
            attention_mask = [1] + tokens["attention_mask"]

            pad_len = max_seq_len - len(input_ids)
            input_ids += [pad_id] * pad_len
            attention_mask += [0] * pad_len
        else:
            tokens = tokenizer(
                text,
                truncation=True,
                max_length=max_seq_len,
                padding="max_length",
            )
            input_ids = tokens["input_ids"]
            attention_mask = tokens["attention_mask"]

        input_ids_list.append(input_ids)
        attention_mask_list.append(attention_mask)

    return input_ids_list, attention_mask_list


def build_tensor_dataset(
    texts: Sequence[str],
    tokenizer: GPT2Tokenizer,
    labels: Sequence[int] | None = None,
) -> TensorDataset:
    input_ids, attention_masks = tokenize_texts(texts, tokenizer, MAX_SEQ_LEN, POOLING_TYPE)

    tensors = [
        torch.tensor(input_ids, dtype=torch.long),
        torch.tensor(attention_masks, dtype=torch.long),
    ]

    if labels is not None:
        tensors.append(torch.tensor(labels, dtype=torch.float32).view(-1, 1))

    return TensorDataset(*tensors)


def save_checkpoint(path: Path, model: nn.Module, optimizer, scheduler=None, metadata=None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }
    if scheduler is not None:
        payload["scheduler_state_dict"] = scheduler.state_dict()
    if metadata is not None:
        payload["metadata"] = metadata

    torch.save(payload, path)


def build_lr_scheduler(optimizer, total_steps: int):
    warmup_steps = max(1, int(total_steps * WARMUP_RATIO))

    def lr_lambda(current_step: int) -> float:
        if current_step < warmup_steps:
            return float(current_step + 1) / float(warmup_steps)

        decay_steps = max(1, total_steps - warmup_steps)
        progress = min(1.0, float(current_step - warmup_steps) / float(decay_steps))
        return max(MIN_LR_RATIO, 1.0 - (1.0 - MIN_LR_RATIO) * progress)

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def compute_binary_f1(preds: torch.Tensor, targets: torch.Tensor) -> float:
    preds = preds.view(-1).long()
    targets = targets.view(-1).long()

    true_positive = ((preds == 1) & (targets == 1)).sum().item()
    false_positive = ((preds == 1) & (targets == 0)).sum().item()
    false_negative = ((preds == 0) & (targets == 1)).sum().item()

    precision = true_positive / max(true_positive + false_positive, 1)
    recall = true_positive / max(true_positive + false_negative, 1)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    desc: str = "Validate",
) -> Tuple[float, float, float]:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    total_true_positive = 0
    total_false_positive = 0
    total_false_negative = 0

    progress = tqdm(dataloader, total=len(dataloader), desc=desc, leave=False)
    try:
        with torch.no_grad():
            for batch in progress:
                input_ids, attention_mask, labels = [tensor.to(DEVICE) for tensor in batch]
                with torch.autocast(device_type=DEVICE.type, dtype=AMP_DTYPE, enabled=USE_AMP):
                    logits = model(input_ids, attention_mask)
                    loss = criterion(logits, labels)

                probs = torch.sigmoid(logits.float())
                preds = (probs > 0.5).long().view(-1)
                targets = labels.long().view(-1)

                batch_size = targets.size(0)
                total_loss += loss.item() * batch_size
                total_correct += (preds == targets).sum().item()
                total_samples += batch_size
                total_true_positive += ((preds == 1) & (targets == 1)).sum().item()
                total_false_positive += ((preds == 1) & (targets == 0)).sum().item()
                total_false_negative += ((preds == 0) & (targets == 1)).sum().item()
    finally:
        progress.close()

    avg_loss = total_loss / max(total_samples, 1)
    accuracy = total_correct / max(total_samples, 1)
    precision = total_true_positive / max(total_true_positive + total_false_positive, 1)
    recall = total_true_positive / max(total_true_positive + total_false_negative, 1)
    f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
    return avg_loss, accuracy, f1


def evaluate_repeatedly(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    repeat_times: int,
    desc: str = "Validate",
) -> Tuple[float, float, float]:
    repeat_times = max(1, repeat_times)
    total_loss = 0.0
    total_acc = 0.0
    total_f1 = 0.0

    for repeat_idx in range(repeat_times):
        eval_loss, eval_acc, eval_f1 = evaluate(
            model,
            dataloader,
            criterion,
            desc=f"{desc} [{repeat_idx + 1}/{repeat_times}]",
        )
        total_loss += eval_loss
        total_acc += eval_acc
        total_f1 += eval_f1

    return (
        total_loss / repeat_times,
        total_acc / repeat_times,
        total_f1 / repeat_times,
    )


def build_tokenizer() -> GPT2Tokenizer:
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2", cache_dir=str(TOKENIZER_CACHE_DIR))
    tokenizer.pad_token = tokenizer.eos_token
    if POOLING_TYPE.lower() == "cls":
        tokenizer.add_special_tokens({"cls_token": "<|cls|>"})
    return tokenizer


def build_model(tokenizer: GPT2Tokenizer) -> nn.Module:
    return BinaryClassifyModel(
        len(tokenizer),
        HIDDEN_DIM,
        MAX_SEQ_LEN,
        DROPOUT_RATE,
        N_ENCODER_LAYER,
        N_HEAD,
        pooling_type=POOLING_TYPE,
    ).to(DEVICE)


def train() -> Path:
    samples = read_train_rows(TRAIN_CSV_PATH)
    train_samples, eval_samples = split_train_eval(samples, EVAL_RATIO, RANDOM_SEED)

    train_texts = [text for text, _ in train_samples]
    train_labels = [label for _, label in train_samples]
    eval_texts = [text for text, _ in eval_samples]
    eval_labels = [label for _, label in eval_samples]

    tokenizer = build_tokenizer()
    train_dataset = build_tensor_dataset(train_texts, tokenizer, train_labels)
    eval_dataset = build_tensor_dataset(eval_texts, tokenizer, eval_labels)

    train_dataloader = DataLoader(train_dataset, batch_size=TRAIN_BATCH_SIZE, shuffle=True)
    eval_dataloader = DataLoader(eval_dataset, batch_size=VALID_BATCH_SIZE, shuffle=False)

    model = build_model(tokenizer)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    total_steps = TOTAL_EPOCHS * len(train_dataloader)
    scheduler = build_lr_scheduler(optimizer, total_steps)

    best_eval_acc = 0.0
    best_eval_f1 = 0.0
    global_step = 0
    swanlab.login(api_key=SWANLAB_API_KEY)
    swanlab.init(
        project="nlpdt-disaster-tweets-classification",
        experiment_name=f"transformer-bs{TRAIN_BATCH_SIZE}-lr{LR}",
        config={
            "model": "BinaryClassifyModel",
            "dataset": "NLP Disaster Tweets",
            "max_seq_len": MAX_SEQ_LEN,
            "hidden_dim": HIDDEN_DIM,
            "dropout_rate": DROPOUT_RATE,
            "n_encoder_layer": N_ENCODER_LAYER,
            "n_head": N_HEAD,
            "learning_rate": LR,
            "min_lr_ratio": MIN_LR_RATIO,
            "weight_decay": WEIGHT_DECAY,
            "warmup_ratio": WARMUP_RATIO,
            "grad_clip_norm": GRAD_CLIP_NORM,
            "train_batch_size": TRAIN_BATCH_SIZE,
            "valid_batch_size": VALID_BATCH_SIZE,
            "total_epochs": TOTAL_EPOCHS,
            "validate_interval": VALIDATE_INTERVAL,
            "validate_repeat_times": VALIDATE_REPEAT_TIMES,
            "eval_ratio": EVAL_RATIO,
            "random_seed": RANDOM_SEED,
            "device": str(DEVICE),
            "amp_dtype": str(AMP_DTYPE),
            "use_amp": USE_AMP,
            "train_size": len(train_dataset),
            "valid_size": len(eval_dataset),
            "tokenizer": "gpt2",
            "pooling": POOLING_TYPE,
            "cls_token": tokenizer.cls_token if POOLING_TYPE.lower() == "cls" else None,
        },
    )

    try:
        for epoch in range(TOTAL_EPOCHS):
            progress = tqdm(
                train_dataloader,
                total=len(train_dataloader),
                desc=f"Epoch {epoch + 1}/{TOTAL_EPOCHS}",
                leave=False,
            )
            for batch in progress:
                input_ids, attention_mask, labels = [tensor.to(DEVICE) for tensor in batch]
                model.train()
                optimizer.zero_grad(set_to_none=True)

                with torch.autocast(device_type=DEVICE.type, dtype=AMP_DTYPE, enabled=USE_AMP):
                    logits = model(input_ids, attention_mask)
                    loss = criterion(logits, labels)

                probs = torch.sigmoid(logits.detach().float())
                preds = (probs > 0.5).long().view(-1)
                targets = labels.long().view(-1)
                batch_acc = (preds == targets).float().mean().item()
                batch_f1 = compute_binary_f1(preds, targets)

                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=GRAD_CLIP_NORM)
                optimizer.step()
                scheduler.step()

                global_step += 1
                lr = scheduler.get_last_lr()[0]
                progress.set_postfix(
                    loss=f"{loss.item():.4f}",
                    acc=f"{batch_acc:.4f}",
                    f1=f"{batch_f1:.4f}",
                    lr=f"{lr:.2e}",
                    grad=f"{float(grad_norm):.2f}",
                )
                swanlab.log(
                    {
                        "train/loss": loss.item(),
                        "train/acc": batch_acc,
                        "train/f1": batch_f1,
                        "train/lr": lr,
                        "train/grad_norm": float(grad_norm),
                        "train/epoch": epoch + 1,
                        "train/global_step": global_step,
                    },
                    step=global_step,
                )

                if global_step % VALIDATE_INTERVAL == 0 or global_step == 1 or global_step == total_steps:
                    progress.clear()
                    eval_loss, eval_acc, eval_f1 = evaluate_repeatedly(
                        model,
                        eval_dataloader,
                        criterion,
                        VALIDATE_REPEAT_TIMES,
                        desc=f"Validate @{global_step}",
                    )
                    tqdm.write(
                        f"step={global_step} "
                        f"eval_loss={eval_loss:.6f} "
                        f"eval_acc={eval_acc:.4f} "
                        f"eval_f1={eval_f1:.4f}"
                    )
                    swanlab.log(
                        {
                            "valid/loss": eval_loss,
                            "valid/acc": eval_acc,
                            "valid/acc_percent": eval_acc * 100,
                            "valid/f1": eval_f1,
                            "valid/epoch": epoch + 1,
                            "valid/global_step": global_step,
                            "valid/best_eval_acc": max(best_eval_acc, eval_acc),
                            "valid/best_eval_f1": max(best_eval_f1, eval_f1),
                        },
                        step=global_step,
                    )
                    progress.refresh()

                    if eval_acc > best_eval_acc:
                        best_eval_acc = eval_acc
                        best_eval_f1 = max(best_eval_f1, eval_f1)
                        save_checkpoint(
                            BEST_CKPT_PATH,
                            model,
                            optimizer,
                            scheduler=scheduler,
                            metadata={
                                "epoch": epoch + 1,
                                "global_step": global_step,
                                "best_eval_acc": best_eval_acc,
                                "best_eval_f1": best_eval_f1,
                                "eval_loss": eval_loss,
                                "eval_ratio": EVAL_RATIO,
                                "random_seed": RANDOM_SEED,
                            },
                        )

            progress.close()
    finally:
        swanlab.finish()

    save_checkpoint(
        CKPT_PATH,
        model,
        optimizer,
        scheduler=scheduler,
        metadata={
            "epoch": TOTAL_EPOCHS,
            "global_step": global_step,
            "best_eval_acc": best_eval_acc,
            "best_eval_f1": best_eval_f1,
            "eval_ratio": EVAL_RATIO,
            "random_seed": RANDOM_SEED,
        },
    )

    return BEST_CKPT_PATH if BEST_CKPT_PATH.exists() else CKPT_PATH


def load_trained_model(checkpoint_path: Path, tokenizer: GPT2Tokenizer) -> nn.Module:
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
    model = build_model(tokenizer)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def predict(checkpoint_path: Path) -> None:
    tokenizer = build_tokenizer()
    model = load_trained_model(checkpoint_path, tokenizer)

    test_rows = read_test_rows(TEST_CSV_PATH)
    test_ids = [sample_id for sample_id, _ in test_rows]
    test_texts = [text for _, text in test_rows]

    test_dataset = build_tensor_dataset(test_texts, tokenizer)
    test_dataloader = DataLoader(test_dataset, batch_size=VALID_BATCH_SIZE, shuffle=False)

    predictions: List[int] = []
    progress = tqdm(test_dataloader, total=len(test_dataloader), desc="Predict", leave=False)
    try:
        with torch.no_grad():
            for batch in progress:
                input_ids, attention_mask = [tensor.to(DEVICE) for tensor in batch]
                with torch.autocast(device_type=DEVICE.type, dtype=AMP_DTYPE, enabled=USE_AMP):
                    logits = model(input_ids, attention_mask)
                probs = torch.sigmoid(logits.float()).view(-1)
                predictions.extend((probs > 0.5).long().cpu().tolist())
    finally:
        progress.close()

    if len(test_ids) != len(predictions):
        raise ValueError("Prediction count does not match test sample count.")

    submission_header = read_submission_header(SAMPLE_SUBMISSION_PATH)
    PREDICT_CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    with PREDICT_CSV_PATH.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(submission_header)
        writer.writerows(zip(test_ids, predictions))


def main() -> None:
    print(f"Using device: {DEVICE}")

    if RUN_MODE == "train":
        train()
        return

    if RUN_MODE == "predict":
        predict(BEST_CKPT_PATH)
        return

    if RUN_MODE == "all":
        best_checkpoint = train()
        predict(best_checkpoint)
        return

    raise ValueError(f"Unsupported RUN_MODE: {RUN_MODE}")


if __name__ == "__main__":
    main()
