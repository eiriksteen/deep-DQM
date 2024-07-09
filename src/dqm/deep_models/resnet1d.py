import torch.nn as nn


class ResBlock1D(nn.Module):

    def __init__(self, channels):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv1d(channels, channels, 5, 1, 2),
            nn.ReLU(),
            nn.Conv1d(channels, channels, 5, 1, 2),
        )

        # Since the batch statistics are nonstationary,
        # identity is probably better than norming
        self.norm = nn.Identity()  # nn.BatchNorm1d(channels)

    def forward(self, x):
        return self.norm(self.block(x) + x)


class ResNet1D(nn.Module):

    def __init__(
            self,
            in_channels,
            hidden_channels,
            num_classes,
            mlp_dim=416
    ):
        super().__init__()

        self.fcn = nn.Sequential(
            nn.Conv1d(in_channels, hidden_channels, 5, 1, 2),
            ResBlock1D(hidden_channels),
            nn.Conv1d(hidden_channels, hidden_channels, 5, 2, 2),
            ResBlock1D(hidden_channels),
            nn.Conv1d(hidden_channels, hidden_channels, 5, 2, 2),
            ResBlock1D(hidden_channels),
            nn.Conv1d(hidden_channels, hidden_channels, 5, 2, 2)
        )

        self.mlp = nn.Sequential(
            nn.Flatten(),
            nn.Linear(mlp_dim, mlp_dim//2),
            nn.Dropout(0.2),
            nn.ReLU(),
            nn.Linear(mlp_dim//2, num_classes),
        )

    def forward(self, x):

        logits = self.fcn(x)
        out = self.mlp(logits)

        return out
