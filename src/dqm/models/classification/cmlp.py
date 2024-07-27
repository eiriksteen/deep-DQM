import torch
import torch.nn as nn
import torch.nn.functional as F
from .transformer import MLPBlock


class ContextMLP(nn.Module):

    def __init__(
            self,
            in_dim: int,
            in_channels: int,
            hidden_dim: int,
            use_ref: bool = False
    ):

        super().__init__()

        self.in_dim = in_dim
        self.in_channels = in_channels
        self.hidden_dim = hidden_dim
        self.use_ref = use_ref

        self.pos_embed = nn.Embedding(in_channels, in_dim)
        self.proj = nn.Linear((2+(1 if use_ref else 0))*in_dim, hidden_dim)
        # self.mha = MultiHeadAttentionBlock(hidden_dim)
        self.mlp = MLPBlock(hidden_dim)
        self.head = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor, ref: torch.Tensor | None = None):

        pw = self.pos_embed.weight.repeat(x.shape[0], 1, 1)

        # Concate the input, positional information (which physics variable),
        # optionally reference
        # x_ = torch.cat((x, ref, pw), dim=-1)
        if self.use_ref:
            ref = ref.repeat(x.shape[0], 1, 1)
            x_ = torch.cat((x, ref, pw), dim=-1)
        else:
            x_ = torch.cat((x, pw), dim=-1)

        logits = self.proj(x_)
        # logits, attn_weights = self.mha(logits)
        logits = self.mlp(logits)
        # Compute the anomaly scores for each physics variable histogram
        scores = self.head(logits).squeeze(-1)
        # Output only the max, since one anomaly is enough to flag the whole input
        # (this makes the model look for anomalies in each physics variable, not only in the whole input)
        out = scores.max(dim=1, keepdim=True).values
        # out = scores.mean(1)
        # Sigmoid to get a probability distribution over anomalies per physics variable
        prob_scores = F.sigmoid(scores)

        return {"logits": out, "prob": prob_scores}
