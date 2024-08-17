import torch
import torch.nn as nn
import torch.nn.functional as F
from ...blocks import AttentionPool


class TemporalContinualTransformer(nn.Module):

    def __init__(
            self,
            backbone: nn.Module,
            n_vars: int,
            hidden_dim: int,
            k_past: int
    ):

        super().__init__()

        self.n_vars = n_vars
        self.hidden_dim = hidden_dim
        self.k_past = k_past

        self.feature_extractor = backbone

        self.change_detect = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.pool = AttentionPool(n_vars, hidden_dim)

        self.head = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor, past: torch.Tensor):

        z = torch.cat([x.unsqueeze(1), past], dim=1)
        z = z.reshape(len(z) * (self.k_past+1), *z.shape[2:])
        logits = self.feature_extractor(z).reshape(
            len(x), self.k_past+1, self.n_vars, self.hidden_dim)

        x_logits = logits[:, -1]
        past_logits = logits[:, :-1]
        past_logits = self.pool(past_logits)

        abs_diff = (x_logits - past_logits).abs()
        change = self.change_detect(abs_diff)
        out = self.head((x_logits + change).mean(1))

        return {"logits": out}
