import os
import torch


# ========== Dataset Config ==========
DATASET_DIR = os.path.join("dataset", "nlpdt")
TRAIN_CSV_PATH = os.path.join(DATASET_DIR, "train.csv")
TEST_CSV_PATH = os.path.join(DATASET_DIR, "test.csv")
SAMPLE_SUBMISSION_PATH = os.path.join(DATASET_DIR, "sample_submission.csv")
PREDICT_CSV_PATH = "predict.csv"
TOKENIZER_CACHE_DIR = "gpt2_tokenizer"
EVAL_RATIO = 0.1
RANDOM_SEED = 42
RUN_MODE = "all"  # options: train, predict, all

# ========== Model Config ==========
MAX_SEQ_LEN = 64
HIDDEN_DIM = 300
DEFAULT_DTYPE = torch.float32
DROPOUT_RATE = 0.3
N_ENCODER_LAYER = 4
N_HEAD = 6
POOLING_TYPE = "cls"  # options: "cls", "max_mean"

# ========== Training Config ==========
LR = 3e-4
MIN_LR_RATIO = 0.1
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.05
GRAD_CLIP_NORM = 1.0

TOTAL_EPOCHS = 10

TRAIN_BATCH_SIZE = 128
VALID_BATCH_SIZE = 128

VALIDATE_INTERVAL = 100
VALIDATE_REPEAT_TIMES = 4

# ========== Color Config ==========
RESET = "\033[0m"
BOLD = "\033[1m"
CYAN = "\033[36m"
GREEN = "\033[32m"
YELLOW = "\033[33m"

# ========== Checkpoint Config ==========
CKPT_PATH = os.path.join("ckpts", "nlpdt_binary_cls.pt")
BEST_CKPT_PATH = os.path.join("ckpts", "nlpdt_binary_cls_best.pt")
