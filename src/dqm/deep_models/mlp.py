import torch.nn as nn


class MLP(nn.Module):

    def __init__(self, in_dim, num_classes):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.Dropout(0.2),
            nn.ReLU(),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        return self.network(x)
