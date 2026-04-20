import os
import torch


# ========== Dataset Config ==========
DATASET_NAME = "nlpdt"
DATASET_DIR = os.path.join("dataset", DATASET_NAME)
TRAIN_FILE = os.path.join(DATASET_DIR, "train.csv")
TEST_FILE = os.path.join(DATASET_DIR, "test.csv")
SAMPLE_SUBMISSION_FILE = os.path.join(DATASET_DIR, "sample_submission.csv")
PREDICT_FILE = "predict.csv"
EVAL_RATIO = 0.1
RANDOM_SEED = 42

# ========== Runtime Config ==========
RUN_MODE = "all"  # options: train, predict, all
NUM_WORKERS = 0
DEVICE = "auto"  # options: auto, cpu, mps, cuda

# ========== Model Config ==========
MODEL_NAME = "microsoft/deberta-v3-small"
LOCAL_FILES_ONLY = True
MAX_SEQ_LEN = 64
HIDDEN_DROPOUT_PROB = 0.1
DEFAULT_DTYPE = torch.float32

# ========== Training Config ==========
TOTAL_EPOCHS = 3
TRAIN_BATCH_SIZE = 64
EVAL_BATCH_SIZE = 64
PREDICT_BATCH_SIZE = 64
LR = 2e-5
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.1
GRAD_CLIP_NORM = 1.0
VALIDATE_INTERVAL = 50
THRESHOLD = 0.5

# ========== SwanLab Config ==========
USE_SWANLAB = True
SWANLAB_PROJECT = "bert-binary-classification"
SWANLAB_EXPERIMENT_NAME = f"{DATASET_NAME}-{MODEL_NAME.split('/')[-1]}"
SWANLAB_API_KEY = os.getenv("SWANLAB_API_KEY", "ZnWwDCFzy78QEM12FZXCr")

# ========== Checkpoint Config ==========
CKPT_DIR = "ckpts"
LAST_CKPT_PATH = os.path.join(CKPT_DIR, f"{DATASET_NAME}_last.pt")
BEST_CKPT_PATH = os.path.join(CKPT_DIR, f"{DATASET_NAME}_best.pt")
