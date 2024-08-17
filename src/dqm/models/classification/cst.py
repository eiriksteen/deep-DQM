import torch
import torch.nn as nn
from ..blocks import MultiHeadAttention, MLP, AttentionPool


class ContinualShiftingTransformer(nn.Module):

    def __init__(
            self,
            in_dim: int,
            n_vars: int,
            hidden_dim: int,
            k_past: int
    ):

        super().__init__()

        self.in_dim = in_dim
        self.n_vars = n_vars
        self.hidden_dim = hidden_dim
        self.k_past = k_past

        self.pos_embed = nn.Embedding(n_vars, in_dim)

        self.embed = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.change_detect = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.mha = MultiHeadAttention(
            hidden_dim,
            sigmoid=False,
            num_heads=4
        )

        self.mlp = MLP(hidden_dim)

        self.head = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor, past: torch.Tensor):

        x_latents = self.embed(x + self.pos_embed.weight)
        past_latents = self.embed(past + self.pos_embed.weight)
        past_latents = past_latents.mean(1)

        abs_diff = (past_latents - x_latents).abs()
        c = self.change_detect(abs_diff)
        attn_logits, attn_weights = self.mha(x_latents + c)
        logits = self.mlp(attn_logits)

        with torch.no_grad():
            source_preds = attn_weights.sum(dim=[1, 2])
            source_preds /= source_preds.max(-1, keepdim=True).values

        logits = self.head(logits.mean(1))

        return {"logits": logits, "source_preds": source_preds}
