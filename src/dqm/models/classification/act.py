import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadAttentionBlock(nn.Module):

    def __init__(
            self,
            d,
            num_heads=4,
            sigmoid: bool = False,
            cross: bool = False
    ):
        super().__init__()

        self.num_heads = num_heads
        self.sigmoid = sigmoid
        self.cross = cross

        if cross:
            self.W = nn.Linear(d, 2*d)
            self.Wc = nn.Linear(d, d)
        else:
            self.W = nn.Linear(d, 3*d)

        self.proj = nn.Linear(d, d)
        self.dropout = nn.Dropout(0.2)
        self.norm = nn.LayerNorm(d)

    def forward(self, x, c=None):

        b, s, d = x.shape

        assert (c is not None) == self.cross

        if self.cross:
            q = self.Wc(c).unsqueeze(0).repeat(b, 1, 1)
            k, v = self.W(x).chunk(2, dim=-1)
        else:
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

        self.patch_size = in_dim
        self.in_dim = in_channels
        self.hidden_dim = hidden_dim
        self.k_past = k_past

        self.pos_embed = nn.Embedding(in_channels, in_dim)

        self.change_detector = nn.Sequential(
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

        self.W_conv = nn.Sequential(
            nn.Conv2d(k_past+1, k_past+1, (in_channels, 1)),
            nn.ReLU(),
            nn.Conv2d(k_past+1, k_past+1, (1, in_dim)),
            nn.Flatten()
        )

        self.mha = MultiHeadAttentionBlock(
            hidden_dim,
            sigmoid=False,
            num_heads=4
        )

        self.mlp = MLPBlock(hidden_dim)

        self.head = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor, past: torch.Tensor | None = None):

        # b,k,n,d -> b,k+1,n,d -> b,n,k+1,d -> conv -> b, n

        z = torch.cat([x.unsqueeze(1), past], dim=1)
        w = F.softmax(self.W_conv(z)[:, :-1], dim=-1)

        x_latents = self.embed(x + self.pos_embed.weight)
        past_latents = self.embed(past + self.pos_embed.weight)
        past_latents = (past_latents * w[:, :, None, None]).sum(dim=1)
        c = self.change_detector((past_latents - x_latents).abs())
        x_latents += c

        attn_logits, attn_weights = self.mha(x_latents)
        logits = self.mlp(attn_logits)
        out = self.head(logits.mean(dim=1))
        scores = attn_weights.mean(dim=1).mean(dim=1)

        return {"logits": out, "attn_weights": attn_weights, "prob": scores}

    # def forward(self, x: torch.Tensor, past: torch.Tensor | None = None):

    #     x_latents = self.embed(x + self.pos_embed.weight)

    #     # (b, k, n, d) -> mean(2) -> (b, k, d) -> append x.mean(1) -> (b, k+1, d)
    #     # -> attention -> mean(-1) -> (b, k+1) -> sigmoid -> weights
    #     past_latents = self.embed(past + self.pos_embed.weight)
    #     past_weights = F.softmax(self.past_weights)[None, :, None, None]
    #     past_latents = (past_latents * past_weights).sum(dim=1)
    #     c = self.change_detector((past_latents - x_latents).abs())
    #     x_latents += c

    #     attn_logits, attn_weights = self.mha(x_latents)
    #     logits = self.mlp(attn_logits)
    #     out = self.head(logits.mean(dim=1))
    #     scores = attn_weights.mean(dim=1).mean(dim=1)

    #     return {"logits": out, "attn_weights": attn_weights, "prob": scores}
