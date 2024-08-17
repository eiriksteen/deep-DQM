import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadAttention(nn.Module):

    def __init__(
            self,
            d,
            num_heads=4,
            sigmoid: bool = False
    ):
        super().__init__()

        self.num_heads = num_heads
        self.sigmoid = sigmoid

        self.W = nn.Linear(d, 3*d)
        self.proj = nn.Linear(d, d)
        self.dropout = nn.Dropout(0.2)
        self.norm = nn.LayerNorm(d)

    def forward(self, x):

        b, s, d = x.shape
        q, k, v = self.W(x).chunk(3, dim=-1)
        q, k, v = (z.reshape(b, s, self.num_heads, d//self.num_heads).transpose(1, 2)
                   for z in (q, k, v))

        attn = torch.einsum("bhqd,bhkd->bhqk", q, k) / (d**0.5)

        if self.sigmoid:
            attn_weights = F.sigmoid(attn)
        else:
            attn_weights = F.softmax(attn, dim=-1)

        attn_logits = attn_weights @ v
        out = self.proj(attn_logits.reshape(b, s, d))
        out = self.dropout(out)

        return self.norm(x + out), attn_weights


class MLP(nn.Module):

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


class AttentionPool(nn.Module):

    def __init__(self, n_vars: int, d: int):
        super().__init__()

        self.q_embed = nn.Embedding(n_vars, d)
        self.norm = nn.LayerNorm(d)
        self.attn = nn.Linear(d, 1)

        self.Wq = nn.Linear(d, d)
        self.Wkv = nn.Linear(d, 2*d)

    def forward(self, x):

        _, _, _, d = x.shape
        q = self.Wq(self.q_embed.weight)
        k, v = self.Wkv(x).chunk(2, dim=-1)
        v = x

        prod = torch.einsum("nd,bsnd->bs", q, k) / d**0.5
        attn = F.softmax(prod, dim=1)

        attn_logits = (attn[:, :, None, None] * v).sum(1)

        return attn_logits
