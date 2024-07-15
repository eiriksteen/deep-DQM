import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadAttentionBlock(nn.Module):

    def __init__(
            self,
            d,
            num_heads=4
    ):
        super().__init__()

        self.num_heads = num_heads
        self.norm = nn.LayerNorm(d)
        self.W = nn.Linear(d, 3*d)
        self.proj = nn.Linear(d, d)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):

        b, s, d = x.shape
        q, k, v = self.W(x).chunk(3, dim=-1)
        q, k, v = (z.reshape(b, s, self.num_heads, d//self.num_heads).transpose(1, 2)
                   for z in (q, k, v))

        attn = torch.einsum("bhqd,bhkd->bhqk", q, k) / (d**0.5)
        attn_weights = F.softmax(attn, dim=-1)
        attn_logits = attn_weights @ v
        out = self.proj(attn_logits.reshape(b, s, d))
        out = self.dropout(out)

        return self.norm(x + out), attn_weights


class MLPBlock(nn.Module):

    def __init__(self, d):
        super().__init__()

        self.norm = nn.LayerNorm(d)
        self.mlp = nn.Sequential(
            nn.Linear(d, d),
            nn.ReLU(),
            nn.Linear(d, d),
            nn.Dropout(0.2)
        )

    def forward(self, x):

        return self.norm(self.mlp(x) + x)


class Transformer(nn.Module):

    def __init__(
            self,
            in_dim: int,
            in_channels: int,
            hidden_dim: int):

        super().__init__()

        self.patch_size = in_dim
        self.in_dim = in_channels
        self.hidden_dim = hidden_dim

        self.embed = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.mha = MultiHeadAttentionBlock(hidden_dim)

        self.mlp = MLPBlock(hidden_dim)

        self.head = nn.Linear(hidden_dim, 1)

    def forward(self, x):

        embeddings = self.embed(x)

        attn_logits, attn_weights = self.mha(embeddings)

        logits = self.mlp(attn_logits)

        out = self.head(logits.mean(dim=1))

        return {"logits": out, "attn_weights": attn_weights}
