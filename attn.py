import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")


class AttentionLayer(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.q = nn.Linear(embed_dim, embed_dim, bias=False)
        self.k = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v = nn.Linear(embed_dim, embed_dim, bias=False)

    def forward(self, x):
        q = self.q(x)
        k = self.k(x)
        v = self.v(x)
        scale = q.size(-1) ** 0.5
        scores = torch.matmul(q, k.transpose(-2, -1)) / scale
        weights = F.softmax(scores, dim=-1)
        return torch.matmul(weights, v)


x = torch.tensor(
    [
        [[1.0, 0.0, 1.0, 0.0], [0.5, 1.0, 0.0, 1.0], [1.0, 1.0, 0.5, 0.0]],
        [[0.0, 1.0, 1.0, 0.5], [1.0, 0.5, 0.0, 1.0], [0.5, 0.0, 1.0, 1.0]],
    ],
    device=device,
)

attn = AttentionLayer(embed_dim=4).to(device)
out = attn(x)
print("input:", x)
print("output:", out)
