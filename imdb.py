from transformers import GPT2Tokenizer
from tqdm import tqdm
from torch.utils.data import TensorDataset, DataLoader

import os
import json
import sys
import torch
import torch.nn as nn

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


def save_checkpoint(path, model, optimizer):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        },
        path,
    )
    print(f"Saved checkpoint to: {path}")


def load_checkpoint(path, model, optimizer=None, device=None):
    ckpt = torch.load(path, map_location=device if device is not None else "cpu")
    model.load_state_dict(ckpt["model_state_dict"])
    if optimizer is not None and "optimizer_state_dict" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    print(f"Loaded checkpoint from: {path}")
    return model, optimizer


# initialize
torch.set_default_dtype(DEFAULT_DTYPE)
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
amp_dtype = torch.bfloat16
use_amp = device.type == "mps"
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
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

# train loop
for epoch in range(TOTAL_EPOCHS):
    pbar = tqdm(train_dataloader, total=len(train_dataloader), desc=f"Epoch {epoch+1}/{TOTAL_EPOCHS}", leave=False)
    for step, batch in enumerate(pbar):
        input_ids, attention_mask, labels = [t.to(device) for t in batch]
        model.train()
        optimizer.zero_grad()
        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
            logits = model(input_ids, attention_mask)
            loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        # log loss
        if step % TRAIN_LOG_INTERVAL == 0 or step == len(train_dataloader) - 1:
            pbar.set_postfix_str(f"Loss={loss.item():.6f}")
            tqdm.write(colorize(f"Epoch {epoch}, Step {step}, Loss: {loss.item():.6f}", YELLOW))

        # validate
        if step % VALIDATE_INTERVAL == 0 or step == len(train_dataloader) - 1:
            model.eval()
            with torch.no_grad():
                total_correct = 0
                total_samples = 0
                for batch in valid_dataloader:
                    input_ids, attention_mask, labels = [t.to(device) for t in batch]
                    with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                        logits = model(input_ids, attention_mask)
                        loss = criterion(logits, labels)

                    # accuracy
                    probs = torch.sigmoid(logits.float())
                    preds = (probs > 0.5).long().view(-1)
                    y_true = (labels > 0.5).long().view(-1)
                    total_correct += (preds == y_true).sum().item()
                    total_samples += y_true.size(0)

            tqdm.write(colorize(f"Valid: Acc={total_correct / total_samples * 100:.2f}%", GREEN))


save_checkpoint(CKPT_PATH, model, optimizer)

# loaded_model = BinaryClassifyModel(len(tokenizer), HIDDEN_DIM, MAX_SEQ_LEN, DROPOUT_RATE, N_ENCODER_LAYER, N_HEAD).to(device)
# load_checkpoint(CKPT_PATH, loaded_model, optimizer=None, device=device)
