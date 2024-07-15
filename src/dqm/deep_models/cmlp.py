import torch
import torch.nn as nn
import torch.nn.functional as F


class MLPBlock(nn.Module):

    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()

        self.mlp = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.ReLU(),
            nn.Linear(out_dim, out_dim),
            nn.Dropout(0.1)
        )

        self.rs = nn.Linear(
            in_dim, out_dim) if in_dim != out_dim else nn.Identity()

        self.norm = nn.LayerNorm(out_dim)

    def forward(self, x):

        return self.norm(self.mlp(x) + self.rs(x))


class MultiHeadAttentionBlock(nn.Module):

    def __init__(
            self,
            d,
            num_heads=1
    ):
        super().__init__()

        self.num_heads = num_heads
        self.norm = nn.LayerNorm(d)
        self.W = nn.Linear(d, 3*d)
        self.proj = nn.Linear(d, d)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):

        b, s, d = x.shape
        q, k, v = self.W(x).chunk(3, dim=-1)
        q, k, v = (z.reshape(b, s, self.num_heads, d//self.num_heads).transpose(1, 2)
                   for z in (q, k, v))

        attn = torch.einsum("bhqd,bhkd->bhqk", q, k) / (d**0.5)
        attn_weights = F.softmax(attn, dim=-1)
        attn_logits = (attn_weights @ v).reshape(b, s, d)
        out = self.proj(attn_logits)
        out = self.dropout(out)

        return self.norm(x + out)


class ContextMLP(nn.Module):

    def __init__(
            self,
            in_dim: int,
            in_channels: int,
            hidden_dim: int):

        super().__init__()

        self.in_dim = in_dim
        self.in_channels = in_channels
        self.hidden_dim = hidden_dim

        self.pos_embed = nn.Embedding(in_channels, in_dim)

        self.network = nn.Sequential(
            nn.Linear(3*in_dim, hidden_dim),
            MLPBlock(hidden_dim, hidden_dim),
        )

        self.head = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor, reference: torch.Tensor):

        ref = reference.to(x.device).repeat(x.shape[0], 1, 1)
        pw = self.pos_embed.weight.repeat(x.shape[0], 1, 1)
        x_ = torch.cat((x, ref, pw), dim=-1)

        logits = self.network(x_)
        scores = self.head(logits)
        out = scores.max(dim=1).values
        prob_scores = F.sigmoid(scores)

        return {"logits": out, "prob": prob_scores}
