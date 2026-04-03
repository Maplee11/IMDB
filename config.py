import torch


# ========== Dataset Config ==========
IMDB_DATASET_DIR = "dataset/aclImdb"
MAX_SEQ_LEN = 384

# ========== Model Config ==========
HIDDEN_DIM = 384
DEFAULT_DTYPE = torch.float32
DROPOUT_RATE = 0.1
N_ENCODER_LAYER = 2
N_HEAD = 4

# ========== Training Config ==========
LR = 3e-4
TOTAL_EPOCHS = 2
TRAIN_BATCH_SIZE = 128
VALID_BATCH_SIZE = 128
VALIDATE_INTERVAL = 50
TRAIN_LOG_INTERVAL = 25