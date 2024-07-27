import torch.nn as nn


class LinearRegressor(nn.Module):

    def __init__(self, in_dim):
        super().__init__()

        self.model = nn.Linear(in_dim, 1)

    def forward(self, x):
        return self.model(x)
