from __future__ import annotations

import torch
import torch.nn as nn
from transformers import AutoModel

from config import HIDDEN_DROPOUT_PROB, LOCAL_FILES_ONLY, MODEL_NAME


class BertBinaryClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(
            MODEL_NAME,
            local_files_only=LOCAL_FILES_ONLY,
            dtype=torch.float32,
        )
        self.encoder = self.encoder.float()
        hidden_size = self.encoder.config.hidden_size
        self.dropout = nn.Dropout(HIDDEN_DROPOUT_PROB)
        self.classifier = nn.Linear(hidden_size, 1)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls_embedding = outputs.last_hidden_state[:, 0, :]
        cls_embedding = self.dropout(cls_embedding)
        logits = self.classifier(cls_embedding)
        return logits.view(-1)
