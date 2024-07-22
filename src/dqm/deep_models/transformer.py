import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadAttentionBlock(nn.Module):

    def __init__(
            self,
            d,
            num_heads=4,
            sigmoid: bool = False
    ):
        super().__init__()

        self.num_heads = num_heads
        self.sigmoid = sigmoid
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

        if self.sigmoid:
            attn_weights = F.sigmoid(attn)
        else:
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
            hidden_dim: int,
            sigmoid_attn: bool = False,
            use_ref: bool = False
    ):

        super().__init__()

        self.patch_size = in_dim
        self.in_dim = in_channels
        self.hidden_dim = hidden_dim
        self.use_ref = use_ref

        self.pos_embed = nn.Embedding(in_channels, in_dim)

        self.change_detector = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.Dropout(0.2),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        ) if use_ref else None

        self.embed = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.Dropout(0.2),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.mha = MultiHeadAttentionBlock(
            hidden_dim,
            sigmoid=sigmoid_attn,
            num_heads=8
        )

        self.mlp = MLPBlock(hidden_dim)

        self.head = nn.Linear(hidden_dim, 1)

    def forward(self, x, ref: torch.Tensor | None = None):

        pe = self.pos_embed.weight.repeat(x.shape[0], 1, 1)
        x_ = x + pe  # torch.cat((x, pe), dim=-1)
        embeddings = self.embed(x_)

        if self.use_ref:
            ref = ref.repeat(x.shape[0], 1, 1)
            c = self.change_detector((x - ref)**2)
            embeddings += c

        attn_logits, attn_weights = self.mha(embeddings)
        logits = self.mlp(attn_logits)
        out = self.head(logits.mean(dim=1))
        scores = attn_weights.mean(dim=1).mean(dim=1)

        return {"logits": out, "attn_weights": attn_weights, "prob": scores}
