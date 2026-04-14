from transformers import GPT2Tokenizer
from tqdm import tqdm
from torch.utils.data import TensorDataset, DataLoader

import os
import json
import sys
import torch
import torch.nn as nn
import swanlab

from transformer import BinaryClassifyModel
from config import *


def colorize(text, color):
    return f"{color}{text}{RESET}" if sys.stdout.isatty() else text


def load_data(data_dir):
    x , y = [], []
    with open(data_dir, "r") as f:
        dic = json.load(f)
    for data in dic:
        x.append(data[0])
        y.append(data[1])

    return x, y


def tokenize(x, tokenizer, max_seq_len):
    input_ids_list = []
    attention_mask_list = []
    for review in tqdm(x):
        tokens = tokenizer(review, truncation=True, max_length=max_seq_len, padding="max_length")
        input_ids_list.append(tokens["input_ids"])
        attention_mask_list.append(tokens["attention_mask"])
    return input_ids_list, attention_mask_list


def save_checkpoint(path, model, optimizer, scheduler=None, metadata=None):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    payload = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }
    if scheduler is not None:
        payload["scheduler_state_dict"] = scheduler.state_dict()
    if metadata is not None:
        payload["metadata"] = metadata
    torch.save(
        payload,
        path,
    )
    print(f"Saved checkpoint to: {path}")


def load_checkpoint(path, model, optimizer=None, scheduler=None, device=None):
    ckpt = torch.load(path, map_location=device if device is not None else "cpu")
    model.load_state_dict(ckpt["model_state_dict"])
    if optimizer is not None and "optimizer_state_dict" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    if scheduler is not None and "scheduler_state_dict" in ckpt:
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
    print(f"Loaded checkpoint from: {path}")
    return model, optimizer, scheduler, ckpt.get("metadata")


def evaluate(model, dataloader, criterion, device, amp_dtype, use_amp):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for batch in dataloader:
            input_ids, attention_mask, labels = [t.to(device) for t in batch]
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                logits = model(input_ids, attention_mask)
                loss = criterion(logits, labels)

            probs = torch.sigmoid(logits.float())
            preds = (probs > 0.5).long().view(-1)
            y_true = (labels > 0.5).long().view(-1)

            batch_size = y_true.size(0)
            total_loss += loss.item() * batch_size
            total_correct += (preds == y_true).sum().item()
            total_samples += batch_size

    avg_loss = total_loss / max(total_samples, 1)
    accuracy = total_correct / max(total_samples, 1)
    return avg_loss, accuracy


def build_lr_scheduler(optimizer, total_steps):
    warmup_steps = max(1, int(total_steps * WARMUP_RATIO))

    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return float(current_step + 1) / float(warmup_steps)

        decay_steps = max(1, total_steps - warmup_steps)
        progress = min(1.0, float(current_step - warmup_steps) / float(decay_steps))
        return max(MIN_LR_RATIO, 1.0 - (1.0 - MIN_LR_RATIO) * progress)

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# initialize
torch.set_default_dtype(DEFAULT_DTYPE)
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
amp_dtype = torch.bfloat16
use_amp = device.type == "mps"
SWANLAB_API_KEY = os.getenv("SWANLAB_API_KEY", "ZnWwDCFzy78QEM12FZXCr")
print("Using device: ", device)

# load and shuffle data
x_train, y_train = load_data(os.path.join(IMDB_DATASET_DIR, "train.json"))
x_valid, y_valid = load_data(os.path.join(IMDB_DATASET_DIR, "test.json"))

# get tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("gpt2", cache_dir="./gpt2_tokenizer")
tokenizer.pad_token = tokenizer.eos_token

# tokenize and to tensor
print("Tokenizing...")
x_train, attention_mask_train = tokenize(x_train, tokenizer, MAX_SEQ_LEN)
x_valid, attention_mask_valid = tokenize(x_valid, tokenizer, MAX_SEQ_LEN)
x_train = torch.tensor(x_train, dtype=torch.long)
x_valid = torch.tensor(x_valid, dtype=torch.long)
y_train = torch.tensor(y_train, dtype=torch.float).view(-1, 1)
y_valid = torch.tensor(y_valid, dtype=torch.float).view(-1, 1)
attention_mask_train = torch.tensor(attention_mask_train, dtype=torch.long)
attention_mask_valid = torch.tensor(attention_mask_valid, dtype=torch.long)

# create dataset and dataloader
train_dataset = TensorDataset(x_train, attention_mask_train, y_train)
valid_dataset = TensorDataset(x_valid, attention_mask_valid, y_valid)
train_dataloader = DataLoader(train_dataset, batch_size=TRAIN_BATCH_SIZE, shuffle=True)
valid_dataloader = DataLoader(valid_dataset, batch_size=VALID_BATCH_SIZE, shuffle=False)

# define model
model = BinaryClassifyModel(len(tokenizer), HIDDEN_DIM, MAX_SEQ_LEN, DROPOUT_RATE, N_ENCODER_LAYER, N_HEAD).to(device)
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
total_training_steps = TOTAL_EPOCHS * len(train_dataloader)
scheduler = build_lr_scheduler(optimizer, total_training_steps)
best_valid_acc = 0.0
global_step = 0

swanlab.login(api_key=SWANLAB_API_KEY)
swanlab.init(
    project="imdb-sentiment-classification",
    experiment_name=f"transformer-bs{TRAIN_BATCH_SIZE}-lr{LR}",
    config={
        "model": "BinaryClassifyModel",
        "dataset": "IMDB",
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
        "train_log_interval": TRAIN_LOG_INTERVAL,
        "device": str(device),
        "amp_dtype": str(amp_dtype),
        "use_amp": use_amp,
        "train_size": len(train_dataset),
        "valid_size": len(valid_dataset),
        "tokenizer": "gpt2",
    },
)

# train loop
try:
    for epoch in range(TOTAL_EPOCHS):
        pbar = tqdm(train_dataloader, total=len(train_dataloader), desc=f"Epoch {epoch+1}/{TOTAL_EPOCHS}", leave=False)
        for step, batch in enumerate(pbar):
            input_ids, attention_mask, labels = [t.to(device) for t in batch]
            model.train()
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                logits = model(input_ids, attention_mask)
                loss = criterion(logits, labels)
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=GRAD_CLIP_NORM)
            optimizer.step()
            scheduler.step()
            global_step += 1
            lr = scheduler.get_last_lr()[0]

            # log loss
            if step % TRAIN_LOG_INTERVAL == 0 or step == len(train_dataloader) - 1:
                pbar.set_postfix_str(f"Loss={loss.item():.6f} LR={lr:.2e}")
                tqdm.write(
                    colorize(
                        f"Epoch {epoch + 1}, Step {step + 1}, Loss: {loss.item():.6f}, "
                        f"GradNorm: {float(grad_norm):.4f}, LR: {lr:.2e}",
                        YELLOW,
                    )
                )
                swanlab.log(
                    {
                        "train/loss": loss.item(),
                        "train/lr": lr,
                        "train/grad_norm": float(grad_norm),
                        "train/epoch": epoch + 1,
                        "train/step_in_epoch": step + 1,
                        "train/global_step": global_step,
                    }
                )

            # validate
            if step % VALIDATE_INTERVAL == 0 or step == len(train_dataloader) - 1:
                valid_loss, valid_acc = evaluate(model, valid_dataloader, criterion, device, amp_dtype, use_amp)
                tqdm.write(colorize(f"Valid: Loss={valid_loss:.6f} Acc={valid_acc * 100:.2f}%", GREEN))
                swanlab.log(
                    {
                        "valid/loss": valid_loss,
                        "valid/acc": valid_acc,
                        "valid/acc_percent": valid_acc * 100,
                        "valid/epoch": epoch + 1,
                        "valid/step_in_epoch": step + 1,
                        "valid/global_step": global_step,
                        "best_valid_acc": max(best_valid_acc, valid_acc),
                    }
                )

                if valid_acc > best_valid_acc:
                    best_valid_acc = valid_acc
                    save_checkpoint(
                        BEST_CKPT_PATH,
                        model,
                        optimizer,
                        scheduler=scheduler,
                        metadata={
                            "epoch": epoch + 1,
                            "step": step + 1,
                            "global_step": global_step,
                            "best_valid_acc": best_valid_acc,
                            "valid_loss": valid_loss,
                        },
                    )
                    tqdm.write(colorize(f"New best checkpoint: Acc={best_valid_acc * 100:.2f}%", CYAN))
                    swanlab.log(
                        {
                            "best/valid_acc": best_valid_acc,
                            "best/valid_acc_percent": best_valid_acc * 100,
                            "best/epoch": epoch + 1,
                            "best/step_in_epoch": step + 1,
                            "best/global_step": global_step,
                        }
                    )

    save_checkpoint(
        CKPT_PATH,
        model,
        optimizer,
        scheduler=scheduler,
        metadata={
            "epoch": TOTAL_EPOCHS,
            "global_step": global_step,
            "best_valid_acc": best_valid_acc,
        },
    )
finally:
    swanlab.finish()

# loaded_model = BinaryClassifyModel(len(tokenizer), HIDDEN_DIM, MAX_SEQ_LEN, DROPOUT_RATE, N_ENCODER_LAYER, N_HEAD).to(device)
# load_checkpoint(CKPT_PATH, loaded_model, optimizer=None, scheduler=None, device=device)
