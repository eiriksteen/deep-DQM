import torch
import torch.nn as nn
import torch.nn.functional as F


class AlphaNet(nn.Module):

    """
    Model for learning how much to update the reference 
    given the current input
    """

    def __init__(self, in_channels: int, in_dim: int, hidden_dim: int = 128):
        super().__init__()

        self.mlp = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.ReLU(),
            nn.Linear(in_dim, 1),
        )

    def forward(self, x: torch.Tensor, ref: torch.Tensor, ref_var: torch.Tensor):

        logits = self.mlp((x-ref).abs()).mean(dim=1)
        alpha = F.sigmoid(logits)

        return alpha

    def update_reference(
            self,
            x: torch.Tensor,
            labels: torch.Tensor,
            ref: torch.Tensor,
            ref_var: torch.Tensor,
            alpha: torch.Tensor):

        x_neg = x[(labels == 0).squeeze(-1)]

        for alpha_, x_ in zip(alpha, x_neg):
            ref = (1 - alpha_) * x_ + alpha_ * ref
            ref_var = (1 - alpha_) * (x_ - ref)**2 + alpha_ * ref_var

        return ref, ref_var
