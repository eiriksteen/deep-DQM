import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def compute_rpb_idx(window_size: int):
    w = torch.arange(window_size)
    mg = torch.meshgrid(w, w)
    return (mg[1] - mg[0] + window_size - 1).long()


class MultiHeadAttentionBlock(nn.Module):

    def __init__(
            self,
            d,
            num_heads=4,
            causal=False,
            apply_relative_pos_bias: bool = False,
            in_dim: int | None = None,
    ):
        super().__init__()

        self.num_heads = num_heads
        self.causal = causal
        self.apply_relative_pos_bias = apply_relative_pos_bias

        if apply_relative_pos_bias:
            self.rpb = nn.Parameter(torch.randn(2*in_dim - 1, num_heads))
            self.register_buffer("rpb_idx", compute_rpb_idx(in_dim))

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

        if self.causal:
            attn += float("-inf") * torch.triu(torch.ones(
                x.shape[1], x.shape[1]), diagonal=1).to(x.device)

        if self.apply_relative_pos_bias:
            attn += self.rpb[self.rpb_idx].permute(2, 0, 1).unsqueeze(0)

        attn_weights = F.softmax(attn, dim=-1)
        attn_logits = attn_weights @ v
        out = self.proj(attn_logits.reshape(b, s, d))
        out = self.dropout(out)

        return self.norm(x + out)


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
            patch_size: int,
            in_dim: int,
            hidden_dim: int,
            num_classes: int):

        super().__init__()

        self.patch_size = patch_size
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes

        self.proj = nn.Sequential(
            nn.Linear(patch_size, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.network = nn.Sequential(
            MultiHeadAttentionBlock(
                hidden_dim,
                apply_relative_pos_bias=False,
                in_dim=in_dim),
            MLPBlock(hidden_dim)
        )

        self.head = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):

        logits = self.proj(x)
        logits = self.network(logits)
        out = self.head(logits.mean(dim=1))

        return out
