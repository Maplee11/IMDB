import torch
import torch.nn as nn
import torch.nn.functional as F


class TransformerEmbedding(nn.Module):
    def __init__(
        self,
        vocab_size,
        n_embed,
        max_seq_len,
        dropout_rate,
        pretrained_weight=None,
        freeze_embeddings=False,
    ):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, n_embed)
        if pretrained_weight is not None:
            if pretrained_weight.shape != (vocab_size, n_embed):
                raise ValueError(
                    "Expected pretrained_weight shape "
                    f"{(vocab_size, n_embed)}, got {tuple(pretrained_weight.shape)}"
                )
            self.token_embedding.weight.data.copy_(pretrained_weight)
        self.token_embedding.weight.requires_grad = not freeze_embeddings

        self.position_embedding = nn.Embedding(max_seq_len, n_embed)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, input_ids):
        batch_size, seq_len = input_ids.shape
        positions = torch.arange(0, seq_len, dtype=torch.long, device=input_ids.device)
        positions = positions.unsqueeze(0).expand(batch_size, seq_len)

        token_emb = self.token_embedding(input_ids)
        position_emb = self.position_embedding(positions)

        x = token_emb + position_emb
        x = self.dropout(x)

        return x


class SingleHeadCausalSelfAttention(nn.Module):
    def __init__(self, hidden_dim, dropout_rate):
        super().__init__()
        self.hidden_dim = hidden_dim

        # 默认 bias=True，虽然原始论文中 QKV 矩阵不带 bias，但实际实现中可以带上
        self.W_q = nn.Linear(hidden_dim, hidden_dim)
        self.W_k = nn.Linear(hidden_dim, hidden_dim)
        self.W_v = nn.Linear(hidden_dim, hidden_dim)
        self.W_o = nn.Linear(hidden_dim, hidden_dim)

        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x, attention_mask):
        Q = self.W_q(x)
        K = self.W_k(x)
        V = self.W_v(x)

        scores = torch.matmul(Q, torch.transpose(K, -2, -1)) / (self.hidden_dim ** 0.5)

        # mask
        mask = torch.triu(torch.ones_like(scores), diagonal=1).bool()
        mask = (mask | ~(attention_mask.bool().unsqueeze(1))) # attn_mask: 0 for padding, 1 for valid
        scores = scores.masked_fill(mask == 1, float('-inf'))

        scores = F.softmax(scores, dim=-1)

        # softmax 之后再 dropout
        scores = self.dropout(scores)

        output = torch.matmul(scores, V)
        output = self.W_o(output)

        return output


class Decoder(nn.Module):
    def __init__(self, hidden_dim, dropout_rate):
        super().__init__()
        self.hidden_dim = hidden_dim

        self.attn_layer = SingleHeadCausalSelfAttention(hidden_dim, dropout_rate)
        self.attn_dropout = nn.Dropout(dropout_rate)
        self.ln_1 = nn.LayerNorm(hidden_dim)
        self.ln_2 = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Dropout(dropout_rate)
        )

    def forward(self, x, attention_mask):
        residual = x
        x = self.ln_1(x)
        x = self.attn_layer(x, attention_mask)
        x = self.attn_dropout(x) + residual

        residual = x
        x = self.ln_2(x)
        x = self.ffn(x) + residual

        return x


class BinaryClassifyModel(nn.Module):
    def __init__(
        self,
        vocab_size,
        hidden_dim,
        max_seq_len,
        dropout_rate,
        n_encoder_layer,
        n_head,
        pooling_type="cls",
        pretrained_embedding_weight=None,
        freeze_embeddings=False,
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.max_seq_len = max_seq_len
        self.pooling_type = pooling_type.lower()

        self.embd_layer = TransformerEmbedding(
            vocab_size,
            hidden_dim,
            max_seq_len,
            dropout_rate,
            pretrained_weight=pretrained_embedding_weight,
            freeze_embeddings=freeze_embeddings,
        )
        
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, 
            nhead=n_head, 
            dropout=dropout_rate, 
            batch_first=True,
            dim_feedforward=hidden_dim * 4
        )

        self.encoder = nn.TransformerEncoder(
            self.encoder_layer, 
            num_layers=n_encoder_layer, 
            enable_nested_tensor=False
        )

        self.final_ln = nn.LayerNorm(hidden_dim)
        self.pooling_dropout = nn.Dropout(dropout_rate)
        classifier_input_dim = hidden_dim if self.pooling_type == "cls" else hidden_dim * 2
        self.classifier = nn.Linear(classifier_input_dim, 1)


    def forward(self, input_ids, attention_mask):
        # attention mask: [bsz, seq_len]
        x = self.embd_layer(input_ids)
        x = self.encoder(x, src_key_padding_mask=(attention_mask == 0))
        x = self.final_ln(x)

        if self.pooling_type == "cls":
            pooled = x[:, 0, :]
        elif self.pooling_type == "max_mean":
            mask = attention_mask.unsqueeze(-1).bool()
            masked_x = x.masked_fill(~mask, float("-inf"))
            max_pooled = masked_x.max(dim=1).values
            max_pooled = torch.where(torch.isfinite(max_pooled), max_pooled, torch.zeros_like(max_pooled))

            masked_sum = (x * attention_mask.unsqueeze(-1)).sum(dim=1)
            valid_len = attention_mask.sum(dim=1, keepdim=True).clamp(min=1).to(x.dtype)
            mean_pooled = masked_sum / valid_len
            pooled = torch.cat([max_pooled, mean_pooled], dim=-1)
        else:
            raise ValueError(f"Unsupported pooling type: {self.pooling_type}")

        pooled = self.pooling_dropout(pooled)
        logits = self.classifier(pooled) # [bs, 1]

        return logits


class BinaryClassifyModel_(nn.Module):
    def __init__(self, vocab_size, hidden_dim, max_seq_len, dropout_rate, n_decoder):
        super().__init__()

        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.max_seq_len = max_seq_len

        self.embd_layer = TransformerEmbedding(
            vocab_size,
            hidden_dim,
            hidden_dim,
            max_seq_len,
            dropout_rate,
        )
        self.decoder_layers = nn.ModuleList([Decoder(hidden_dim, dropout_rate) for _ in range(n_decoder)])
        self.final_ln = nn.LayerNorm(hidden_dim)
        self.classifier = nn.Linear(hidden_dim, 1)

    def forward(self, input_ids, attention_mask):
        x = self.embd_layer(input_ids)

        for decoder in self.decoder_layers:
            x = decoder(x, attention_mask)

        x = self.final_ln(x)

        # 预测任务，只取最后一个非 padding token
        # cusum: [bs, seq_len], 表示每个样本每个位置的前缀和，例如 [1, 2, 3, 3, 3]
        # sum: [bs, 1], 表示每个样本的 mask 总和，例如 [3]
        # 因此等于 sum 的位置，就是最后非 padding token 位置 或者 padding 位置，只要再加一个条件：本身就是 1
        # idx: [bs, seq_len]，只有一个 true 的 bool 矩阵
        mask = attention_mask.bool()
        idx = (mask.cumsum(1) == mask.sum(1, keepdim=True)) & (mask == 1)
        # TODO: 这里语法正确性存疑
        x = x[idx] # [bs, hidden_dim]
        logits = self.classifier(x) # [bs, 1]

        return logits
        
