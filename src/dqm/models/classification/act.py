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


class AdaptiveConvolutionalTransformer(nn.Module):

    def __init__(
            self,
            in_dim: int,
            in_channels: int,
            hidden_dim: int,
            k_past: int
    ):

        super().__init__()

        self.in_dim = in_dim
        self.in_channels = in_channels
        self.hidden_dim = hidden_dim
        self.k_past = k_past

        self.pos_embed = nn.Embedding(in_channels, in_dim)

        self.change_detect = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.embed = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.mha = MultiHeadAttentionBlock(
            hidden_dim,
            sigmoid=False,
            num_heads=4
        )

        self.mlp = MLPBlock(hidden_dim)

        self.head = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor, past: torch.Tensor):

        x_latents = self.embed(x + self.pos_embed.weight)
        past_latents = self.embed(past + self.pos_embed.weight)

        w = torch.ones((len(x), self.k_past)).to(x.device) / self.k_past
        past_latents = (past_latents * w[:, :, None, None]).sum(dim=1)

        abs_diff = (past_latents - x_latents).abs()
        c = self.change_detect(abs_diff)
        attn_logits, attn_weights = self.mha(x_latents + c)
        logits = self.mlp(attn_logits)

        with torch.no_grad():
            source_preds = attn_weights.sum(dim=[1, 2])
            source_preds /= source_preds.max(-1, keepdim=True).values

        logits = self.head(x_latents.mean(1))

        return {"logits": logits, "source_preds": source_preds}
