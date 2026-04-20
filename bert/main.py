from __future__ import annotations

import csv
import os
import random
import swanlab
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, Tuple

import torch
import torch.nn as nn
from tqdm import tqdm
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

from config import *
from datasets import get_builder
from model import BertBinaryClassifier


ROOT_DIR = Path(__file__).resolve().parent


def write_with_progress(progress: tqdm | None, message: str) -> None:
    if progress is not None:
        progress.clear()
    tqdm.write(message)
    if progress is not None:
        progress.refresh()


def close_progress(progress: tqdm | None) -> None:
    if progress is None:
        return
    progress.clear()
    progress.close()


def resolve_device() -> torch.device:
    if DEVICE != "auto":
        return torch.device(DEVICE)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

def resolve_autocast(device: torch.device) -> Tuple[bool, torch.dtype]:
    if device.type == "cuda":
        return True, torch.float16
    # MPS mixed precision still hits dtype mismatch issues on some transformer ops.
    # Keep MPS in float32 for stability.
    if device.type == "mps":
        return False, torch.float32
    return False, torch.float32


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def to_device(batch: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    return {key: value.to(device) for key, value in batch.items()}


def compute_metrics(logits: torch.Tensor, labels: torch.Tensor, threshold: float) -> Tuple[float, float]:
    probs = torch.sigmoid(logits)
    preds = (probs >= threshold).long()
    labels = labels.long()

    accuracy = (preds == labels).float().mean().item()

    true_positive = ((preds == 1) & (labels == 1)).sum().item()
    false_positive = ((preds == 1) & (labels == 0)).sum().item()
    false_negative = ((preds == 0) & (labels == 1)).sum().item()

    precision = true_positive / max(true_positive + false_positive, 1)
    recall = true_positive / max(true_positive + false_negative, 1)
    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)
    return accuracy, f1


def evaluate(
    model: nn.Module,
    dataloader,
    criterion: nn.Module,
    device: torch.device,
    threshold: float,
    use_amp: bool,
    amp_dtype: torch.dtype,
) -> Dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_samples = 0
    all_logits = []
    all_labels = []

    with torch.no_grad():
        for batch in dataloader:
            batch = to_device(batch, device)
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                logits = model(batch["input_ids"], batch["attention_mask"])
                labels = batch["labels"]
                loss = criterion(logits, labels)

            batch_size = labels.size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size
            all_logits.append(logits.detach().cpu())
            all_labels.append(labels.detach().cpu())

    all_logits_tensor = torch.cat(all_logits, dim=0)
    all_labels_tensor = torch.cat(all_labels, dim=0)
    acc, f1 = compute_metrics(all_logits_tensor, all_labels_tensor, threshold)
    return {
        "loss": total_loss / max(total_samples, 1),
        "acc": acc,
        "f1": f1,
    }


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer,
    scheduler,
    epoch: int,
    global_step: int,
    best_eval_f1: float,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "epoch": epoch,
            "global_step": global_step,
            "best_eval_f1": best_eval_f1,
        },
        path,
    )


def load_checkpoint(path: Path, model: nn.Module, optimizer=None, scheduler=None):
    checkpoint = torch.load(path, map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    if scheduler is not None and "scheduler_state_dict" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    return checkpoint


class SwanLabLogger:
    def __init__(self, enabled: bool, config_dict: Dict):
        self.enabled = enabled and swanlab is not None
        if not self.enabled:
            return

        swanlab.login(api_key=SWANLAB_API_KEY)
        swanlab.init(
            project=SWANLAB_PROJECT,
            experiment_name=f"{SWANLAB_EXPERIMENT_NAME}-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
            config=config_dict,
        )

    def log(self, metrics: Dict, step: int) -> None:
        if self.enabled:
            swanlab.log(metrics, step=step)

    def finish(self) -> None:
        if self.enabled:
            swanlab.finish()


def predict(model: nn.Module, dataloader, device: torch.device) -> Iterable[int]:
    model.eval()
    predictions = []
    use_amp, amp_dtype = resolve_autocast(device)
    with torch.no_grad():
        for batch in dataloader:
            batch = to_device(batch, device)
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                logits = model(batch["input_ids"], batch["attention_mask"])
            probs = torch.sigmoid(logits.float())
            preds = (probs >= THRESHOLD).long().cpu().tolist()
            predictions.extend(preds)
    return predictions


def write_predictions(path: Path, header, ids, preds) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()
        for sample_id, pred in zip(ids, preds):
            writer.writerow({"id": sample_id, "target": int(pred)})


def train_and_predict() -> None:
    os.makedirs(ROOT_DIR / CKPT_DIR, exist_ok=True)
    torch.set_default_dtype(DEFAULT_DTYPE)
    set_seed(RANDOM_SEED)
    device = resolve_device()
    use_amp, amp_dtype = resolve_autocast(device)
    print(f"Using device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        local_files_only=LOCAL_FILES_ONLY,
    )

    builder = get_builder(DATASET_NAME, root_dir=ROOT_DIR, tokenizer=tokenizer)
    task_data = builder.build()
    meta = task_data.meta or {}

    model = BertBinaryClassifier().to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    total_training_steps = TOTAL_EPOCHS * len(task_data.train)
    warmup_steps = int(total_training_steps * WARMUP_RATIO)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_training_steps,
    )

    logger = SwanLabLogger(
        enabled=USE_SWANLAB,
        config_dict={
            "dataset_name": DATASET_NAME,
            "model_name": MODEL_NAME,
            "train_size": meta.get("train_size"),
            "eval_size": meta.get("eval_size"),
            "test_size": meta.get("test_size"),
            "epochs": TOTAL_EPOCHS,
            "lr": LR,
            "weight_decay": WEIGHT_DECAY,
            "warmup_ratio": WARMUP_RATIO,
            "grad_clip_norm": GRAD_CLIP_NORM,
            "threshold": THRESHOLD,
        },
    )

    best_eval_f1 = -1.0
    global_step = 0
    try:
        if RUN_MODE in {"train", "all"}:
            for epoch in range(TOTAL_EPOCHS):
                model.train()
                progress = tqdm(
                    task_data.train,
                    desc=f"Epoch {epoch + 1}/{TOTAL_EPOCHS}",
                    total=len(task_data.train),
                    leave=False,
                )
                for batch in progress:
                    batch = to_device(batch, device)
                    with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                        logits = model(batch["input_ids"], batch["attention_mask"])
                        labels = batch["labels"]
                        loss = criterion(logits, labels)

                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=GRAD_CLIP_NORM)
                    optimizer.step()
                    scheduler.step()

                    global_step += 1
                    current_lr = scheduler.get_last_lr()[0]
                    progress.set_postfix(loss=f"{loss.item():.4f}", lr=f"{current_lr:.2e}")

                    logger.log(
                        {
                            "train/loss": loss.item(),
                            "train/lr": current_lr,
                            "train/grad_norm": float(grad_norm),
                            "train/epoch": epoch + 1,
                        },
                        step=global_step,
                    )

                    if global_step % VALIDATE_INTERVAL == 0 or global_step == total_training_steps:
                        eval_metrics = evaluate(
                            model,
                            task_data.eval,
                            criterion,
                            device,
                            THRESHOLD,
                            use_amp=use_amp,
                            amp_dtype=amp_dtype,
                        )
                        write_with_progress(
                            progress,
                            f"[Eval] step={global_step} "
                            f"loss={eval_metrics['loss']:.4f} "
                            f"acc={eval_metrics['acc']:.4f} "
                            f"f1={eval_metrics['f1']:.4f}",
                        )
                        logger.log(
                            {
                                "eval/loss": eval_metrics["loss"],
                                "eval/acc": eval_metrics["acc"],
                                "eval/f1": eval_metrics["f1"],
                                "eval/best_f1": max(best_eval_f1, eval_metrics["f1"]),
                            },
                            step=global_step,
                        )

                        if eval_metrics["f1"] > best_eval_f1:
                            best_eval_f1 = eval_metrics["f1"]
                            save_checkpoint(
                                ROOT_DIR / BEST_CKPT_PATH,
                                model,
                                optimizer,
                                scheduler,
                                epoch=epoch + 1,
                                global_step=global_step,
                                best_eval_f1=best_eval_f1,
                            )
                            write_with_progress(progress, f"Saved best checkpoint with f1={best_eval_f1:.4f}")

                save_checkpoint(
                    ROOT_DIR / LAST_CKPT_PATH,
                    model,
                    optimizer,
                    scheduler,
                    epoch=epoch + 1,
                    global_step=global_step,
                    best_eval_f1=best_eval_f1,
                )
                close_progress(progress)

        if RUN_MODE in {"predict", "all"}:
            best_ckpt_path = ROOT_DIR / BEST_CKPT_PATH
            if not best_ckpt_path.exists():
                raise FileNotFoundError(f"Best checkpoint not found: {best_ckpt_path}")

            load_checkpoint(best_ckpt_path, model)
            predictions = predict(model, task_data.test, device)
            output_path = ROOT_DIR / PREDICT_FILE
            write_predictions(
                output_path,
                meta["submission_header"],
                meta["test_ids"],
                predictions,
            )
            print(f"Saved predictions to: {output_path}")
    finally:
        logger.finish()


if __name__ == "__main__":
    train_and_predict()
