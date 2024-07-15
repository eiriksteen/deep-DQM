import torch
import torch.nn as nn
import torch.nn.functional as F


class RefBuilder(nn.Module):

    """
    Model for learning how much to update the reference 
    given the current input
    """

    def __init__(self, in_channels: int, in_dim: int):
        super().__init__()

        self.pos_embed = nn.Embedding(in_channels, in_dim)

        self.mlp = nn.Sequential(
            nn.Linear(3*in_dim, in_dim),
            nn.ReLU(),
            nn.Linear(in_dim, 1),
        )

    def forward(self, x: torch.Tensor, reference: torch.Tensor):

        ref = reference.to(x.device).repeat(x.shape[0], 1, 1)
        pw = self.pos_embed.weight.repeat(x.shape[0], 1, 1)
        x_ = torch.cat((x, ref, pw), dim=-1)
        logits = self.mlp(x_).mean(1)
        alpha = F.sigmoid(logits)

        return alpha

    def update_reference(
            self,
            x: torch.Tensor,
            labels: torch.Tensor,
            reference:
            torch.Tensor, alpha: torch.Tensor):

        neg_idx = (labels == 0).squeeze(-1)
        x_neg = x[neg_idx]

        for i, x_ in enumerate(x_neg):
            alpha_ = alpha[i]
            reference = (1 - alpha_) * x_ + alpha_ * reference

        return reference
