import torch
import os


# ========== Dataset Config ==========
IMDB_DATASET_DIR = "dataset/aclImdb"
MAX_SEQ_LEN = 384

# ========== Model Config ==========
HIDDEN_DIM = 384
DEFAULT_DTYPE = torch.bfloat16
DROPOUT_RATE = 0.1
N_ENCODER_LAYER = 4
N_HEAD = 4

# ========== Training Config ==========
LR = 3e-4
TOTAL_EPOCHS = 2
TRAIN_BATCH_SIZE = 128
VALID_BATCH_SIZE = 128
VALIDATE_INTERVAL = 50
TRAIN_LOG_INTERVAL = 25

# ========== Color Config ==========
RESET = "\033[0m"
BOLD = "\033[1m"
CYAN = "\033[36m"
GREEN = "\033[32m"
YELLOW = "\033[33m"

# ========== Checkpoint Config ==========
CKPT_PATH = os.path.join("checkpoints", "imdb_binary_cls.pt")
