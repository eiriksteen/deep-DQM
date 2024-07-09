import torch.nn as nn


class CNN1D(nn.Module):

    def __init__(
            self,
            in_channels,
            hidden_channels,
            num_classes,
            mlp_dim=416
    ):
        super().__init__()

        self.fcn = nn.Sequential(
            nn.Conv1d(in_channels, hidden_channels, 5, 2, 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Conv1d(hidden_channels, hidden_channels, 5, 2, 2),
            nn.ReLU(),
            nn.Conv1d(hidden_channels, hidden_channels, 5, 2, 2),
        )

        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(mlp_dim, num_classes)
        )

    def forward(self, x):

        logits = self.fcn(x)
        out = self.head(logits)

        return out


class CNN2D(nn.Module):

    def __init__(
            self,
            in_channels,
            hidden_channels,
            num_classes,
            mlp_dim=512
    ):
        super().__init__()

        self.fcn = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, 3, 2, 1),
            nn.ReLU(),
            nn.BatchNorm2d(hidden_channels),
            nn.Conv2d(hidden_channels, hidden_channels, 3, 2, 1),
            nn.ReLU(),
            nn.BatchNorm2d(hidden_channels),
            nn.Conv2d(hidden_channels, hidden_channels, 3, 2, 1),
            nn.ReLU(),
            nn.BatchNorm2d(hidden_channels),
            nn.Conv2d(hidden_channels, hidden_channels, 3, 2, 1),
            nn.ReLU(),
            nn.BatchNorm2d(hidden_channels),
            nn.Conv2d(hidden_channels, hidden_channels, 3, 2, 1),
        )

        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(mlp_dim, mlp_dim//2),
            nn.Dropout(0.2),
            nn.ReLU(),
            nn.Linear(mlp_dim//2, num_classes)
        )

    def forward(self, x):

        logits = self.fcn(x.unsqueeze(1))
        out = self.head(logits)

        return out
