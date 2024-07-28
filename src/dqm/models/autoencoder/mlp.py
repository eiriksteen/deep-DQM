import torch
import torch.nn as nn


class AutoMLP(nn.Module):

    def __init__(
            self,
            in_dim: int,
            in_channels: int,
            hidden_dim: int
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

        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Dropout(0.2),
            nn.ReLU(),
            nn.Linear(hidden_dim, in_dim),
        )

    def forward(self, x: torch.Tensor):

        x_ = x + self.pos_embed.weight
        latents = self.encoder(x_)
        out = self.decoder(latents)

        return out


class AutoMLPSep(nn.Module):

    def __init__(
            self,
            in_dim: int,
            hidden_dim: int,
    ):

        super().__init__()

        self.in_dim = in_dim
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


class AutoMLPSepParallel(nn.Module):

    def __init__(self,
                 in_dim: int,
                 in_channels: int,
                 hidden_dim: int):
        super().__init__()

        self.models = nn.ModuleList([
            AutoMLPSep(in_dim, hidden_dim) for _ in range(in_channels)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.stack([model(x[:, i]) for i, model in enumerate(self.models)], dim=1)
