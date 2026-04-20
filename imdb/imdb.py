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


def write_with_pbar(pbar, message):
    if pbar is not None:
        pbar.clear()
    tqdm.write(message)
    if pbar is not None:
        pbar.refresh()


def load_data(data_dir):
    x , y = [], []
    with open(data_dir, "r") as f:
        dic = json.load(f)
    for data in dic:
        x.append(data[0])
        y.append(data[1])

    return x, y


def tokenize(x, tokenizer, max_seq_len, pooling_type):
    input_ids_list = []
    attention_mask_list = []
    pooling_type = pooling_type.lower()

    if pooling_type == "cls":
        cls_id = tokenizer.cls_token_id
        pad_id = tokenizer.pad_token_id

    for review in tqdm(x):
        if pooling_type == "cls":
            tokens = tokenizer(
                review,
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
            tokens = tokenizer(review, truncation=True, max_length=max_seq_len, padding="max_length")
            input_ids = tokens["input_ids"]
            attention_mask = tokens["attention_mask"]

        input_ids_list.append(input_ids)
        attention_mask_list.append(attention_mask)
    return input_ids_list, attention_mask_list

def glove_candidates_for_token(token):
    if token is None or token.startswith("<|"):
        return []

    stripped = token.lstrip("ĠĊ")
    candidates = []
    for candidate in (token, token.lower(), stripped, stripped.lower()):
        if candidate and candidate not in candidates and not candidate.startswith("<|"):
            candidates.append(candidate)
    return candidates


def build_glove_embedding_weight(glove_path, tokenizer, embedding_dim):
    token_to_ids = {}
    for token_id in range(len(tokenizer)):
        token = tokenizer.convert_ids_to_tokens(token_id)
        for candidate in glove_candidates_for_token(token):
            token_to_ids.setdefault(candidate, []).append(token_id)

    embedding_weight = torch.empty(len(tokenizer), embedding_dim, dtype=torch.float32)
    nn.init.normal_(embedding_weight, mean=0.0, std=0.02)

    matched_token_ids = set()
    with open(glove_path, "r", encoding="utf-8") as f:
        for line in tqdm(f, desc="Loading GloVe", leave=False):
            values = line.rstrip().split()
            if len(values) != embedding_dim + 1:
                continue

            word = values[0]
            token_ids = token_to_ids.get(word)
            if not token_ids:
                continue

            vector = torch.tensor([float(v) for v in values[1:]], dtype=embedding_weight.dtype)
            for token_id in token_ids:
                if token_id not in matched_token_ids:
                    embedding_weight[token_id] = vector
                    matched_token_ids.add(token_id)

    if tokenizer.pad_token_id is not None:
        embedding_weight[tokenizer.pad_token_id].zero_()

    return embedding_weight, len(matched_token_ids)


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
GLOVE_PATH = "/Users/bytedance/code/ml/glove.6B.300d.txt"
GLOVE_DIM = 300
print("Using device: ", device)

# load and shuffle data
x_train, y_train = load_data(os.path.join(IMDB_DATASET_DIR, "train.json"))
x_valid, y_valid = load_data(os.path.join(IMDB_DATASET_DIR, "test.json"))

# get tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("gpt2", cache_dir="./gpt2_tokenizer", local_files_only=True)
tokenizer.pad_token = tokenizer.eos_token
if POOLING_TYPE.lower() == "cls":
    tokenizer.add_special_tokens({"cls_token": "<|cls|>"})

if HIDDEN_DIM != GLOVE_DIM:
    raise ValueError(f"HIDDEN_DIM ({HIDDEN_DIM}) must match GLOVE_DIM ({GLOVE_DIM}) when projection is disabled.")

glove_embedding_weight, glove_matched_count = build_glove_embedding_weight(
    GLOVE_PATH,
    tokenizer,
    embedding_dim=GLOVE_DIM,
)
print(
    f"GloVe initialized {glove_matched_count}/{len(tokenizer)} tokenizer entries "
    f"({glove_matched_count / len(tokenizer):.2%})"
)

# tokenize and to tensor
print("Tokenizing...")
x_train, attention_mask_train = tokenize(x_train, tokenizer, MAX_SEQ_LEN, POOLING_TYPE)
x_valid, attention_mask_valid = tokenize(x_valid, tokenizer, MAX_SEQ_LEN, POOLING_TYPE)
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
model = BinaryClassifyModel(
    len(tokenizer),
    HIDDEN_DIM,
    MAX_SEQ_LEN,
    DROPOUT_RATE,
    N_ENCODER_LAYER,
    N_HEAD,
    pooling_type=POOLING_TYPE,
    pretrained_embedding_weight=glove_embedding_weight,
).to(device)
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
total_training_steps = TOTAL_EPOCHS * len(train_dataloader)
scheduler = build_lr_scheduler(optimizer, total_training_steps)
best_valid_acc = 0.0

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
        "device": str(device),
        "amp_dtype": str(amp_dtype),
        "use_amp": use_amp,
        "train_size": len(train_dataset),
        "valid_size": len(valid_dataset),
        "tokenizer": "gpt2",
        "pooling": POOLING_TYPE,
        "cls_token": tokenizer.cls_token if POOLING_TYPE.lower() == "cls" else None,
        "glove_path": GLOVE_PATH,
        "glove_dim": GLOVE_DIM,
        "glove_matched_count": glove_matched_count,
        "glove_match_ratio": glove_matched_count / len(tokenizer),
    },
)

# train loop
global_step = 0
try:
    for epoch in range(TOTAL_EPOCHS):
        pbar = tqdm(train_dataloader, total=len(train_dataloader), desc=f"Epoch {epoch+1}/{TOTAL_EPOCHS}", leave=False)
        for batch in pbar:
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
            lr = scheduler.get_last_lr()[0]

            global_step += 1

            # log loss
            pbar.set_postfix_str(f"Loss={loss.item():.6f} LR={lr:.2e}")
            swanlab.log(
                {
                    "train/loss": loss.item(),
                    "train/lr": lr,
                    "train/grad_norm": float(grad_norm),
                    "train/epoch": epoch + 1,
                    "train/global_step": global_step,
                },
                step=global_step,
            )

            # validate
            if global_step % VALIDATE_INTERVAL == 0 or global_step == 1 or global_step == total_training_steps:
                valid_loss, valid_acc = evaluate(model, valid_dataloader, criterion, device, amp_dtype, use_amp)
                write_with_pbar(pbar, colorize(f"Valid: Loss={valid_loss:.6f} Acc={valid_acc * 100:.2f}%", GREEN))
                swanlab.log(
                    {
                        "valid/loss": valid_loss,
                        "valid/acc": valid_acc,
                        "valid/acc_percent": valid_acc * 100,
                        "valid/epoch": epoch + 1,
                        "valid/global_step": global_step,
                        "valid/best_valid_acc": max(best_valid_acc, valid_acc),
                    },
                    step=global_step,
                )

                # save best ckpt
                if valid_acc > best_valid_acc:
                    best_valid_acc = valid_acc
                    if pbar is not None:
                        pbar.clear()
                    save_checkpoint(
                        BEST_CKPT_PATH,
                        model,
                        optimizer,
                        scheduler=scheduler,
                        metadata={
                            "epoch": epoch + 1,
                            "global_step": global_step,
                            "best_valid_acc": best_valid_acc,
                            "valid_loss": valid_loss,
                        },
                    )
                    write_with_pbar(pbar, colorize(f"New best checkpoint: Acc={best_valid_acc * 100:.2f}%", CYAN))

        pbar.close()

    pbar = None
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
