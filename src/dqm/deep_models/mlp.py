import torch.nn as nn


class MLP(nn.Module):

    def __init__(self, in_dim, num_classes):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(in_dim, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.Dropout(0.2),
            nn.ReLU(),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        return self.network(x)
