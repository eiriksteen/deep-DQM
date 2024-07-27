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


class AutoMLP(nn.Module):

    def __init__(
            self,
            in_dim: int,
            in_channels: int,
            hidden_dim: int,
            # sigmoid_attn: bool = False
    ):

        super().__init__()

        self.patch_size = in_dim
        self.in_dim = in_channels
        self.hidden_dim = hidden_dim

        self.pos_embed = nn.Embedding(in_channels, in_dim)

        self.encoder = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.Dropout(0.2),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # self.mha = MultiHeadAttentionBlock(
        #     hidden_dim,
        #     sigmoid=sigmoid_attn,
        #     num_heads=8
        # )

        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Dropout(0.2),
            nn.ReLU(),
            nn.Linear(hidden_dim, in_dim),
        )

    def forward(self, x: torch.Tensor):

        x_ = x + self.pos_embed.weight
        latents = self.encoder(x_)
        # attn_logits, _ = self.mha(latents)
        # out = self.decoder(attn_logits)
        out = self.decoder(latents)

        return out


class AutoMLPSep(nn.Module):

    def __init__(
            self,
            in_dim: int,
            hidden_dim: int,
    ):

        super().__init__()

        self.patch_size = in_dim
        self.hidden_dim = hidden_dim

        self.encoder = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.Dropout(0.2),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Dropout(0.2),
            nn.ReLU(),
            nn.Linear(hidden_dim, in_dim),
        )

    def forward(self, x: torch.Tensor):

        latents = self.encoder(x)
        out = self.decoder(latents)

        return out
