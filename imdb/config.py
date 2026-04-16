import torch
import os


# ========== Dataset Config ==========
IMDB_DATASET_DIR = "dataset/aclImdb"
MAX_SEQ_LEN = 384

# ========== Model Config ==========
HIDDEN_DIM = 384
DEFAULT_DTYPE = torch.float32
DROPOUT_RATE = 0.1
N_ENCODER_LAYER = 2
N_HEAD = 4
POOLING_TYPE = "cls"  # options: "cls", "max_mean"

# ========== Training Config ==========
LR = 3e-4
MIN_LR_RATIO = 0.1
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.05
GRAD_CLIP_NORM = 1.0

TOTAL_EPOCHS = 15

TRAIN_BATCH_SIZE = 128
VALID_BATCH_SIZE = 128

VALIDATE_INTERVAL = 200

# ========== Color Config ==========
RESET = "\033[0m"
BOLD = "\033[1m"
CYAN = "\033[36m"
GREEN = "\033[32m"
YELLOW = "\033[33m"

# ========== Checkpoint Config ==========
CKPT_PATH = os.path.join("ckpts", "imdb_binary_cls.pt")
BEST_CKPT_PATH = os.path.join("ckpts", "imdb_binary_cls_best.pt")
