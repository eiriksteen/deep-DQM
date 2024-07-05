import torch.nn as nn


class CNN(nn.Module):

    def __init__(
            self,
            in_channels,
            hidden_channels,
            num_classes,
            mlp_dim=1160
    ):
        super().__init__()

        self.fcn = nn.Sequential(
            nn.Conv1d(in_channels, hidden_channels, 5, 2, 2),
            nn.ReLU(),
            nn.Conv1d(hidden_channels, hidden_channels, 5, 2, 2),
            nn.ReLU(),
            nn.Conv1d(hidden_channels, hidden_channels, 5, 2, 2),
        )

        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(mlp_dim, mlp_dim//2),
            nn.Dropout(0.2),
            nn.ReLU(),
            nn.Linear(mlp_dim//2, num_classes)
        )

    def forward(self, x):

        logits = self.fcn(x)
        out = self.head(logits)

        return out
