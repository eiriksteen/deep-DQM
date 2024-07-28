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


class ConvTran(nn.Module):

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
        self.in_channels = in_channels
        ###
        self.hidden_channels = 64
        ###
        self.use_ref = use_ref

        self.embed = nn.Sequential(
            nn.Conv1d(self.in_channels, self.hidden_channels, 5, 2, 2),
            nn.ReLU(),
            nn.Conv1d(self.hidden_channels, self.hidden_channels, 5, 1, 2),
        )

        self.change_detector = nn.Sequential(
            nn.Conv1d(self.hidden_channels, self.hidden_channels, 5, 1, 2),
            nn.ReLU(),
            nn.Conv1d(self.hidden_channels, self.hidden_channels, 5, 1, 2),
        ) if use_ref else None

        self.out_conv = nn.Sequential(
            nn.Conv1d(self.hidden_channels, self.hidden_channels, 5, 2, 2),
            nn.ReLU(),
            nn.Conv1d(self.hidden_channels, self.hidden_channels, 5, 1, 2),
        )

        self.head = nn.Linear(64*25, 1)

    def forward(
            self,
            x: torch.Tensor,
            ref: torch.Tensor | None = None
    ):

        x_latents = self.embed(x)

        if self.use_ref:
            ref_latents = self.embed(ref)
            c = self.change_detector((ref_latents - x_latents).abs())
            x_latents += c

        # print(x_latents.shape)

        out = self.out_conv(x_latents)
        out = self.head(out.flatten(1))

        return {"logits": out}
