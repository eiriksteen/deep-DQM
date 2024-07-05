import torch.nn as nn


class LinearRegressor(nn.Module):

    def __init__(self, in_dim, num_classes):
        super().__init__()

        self.model = nn.Linear(in_dim, num_classes)

    def forward(self, x):
        return self.model(x)
