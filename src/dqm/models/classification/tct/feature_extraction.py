import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights
from ...blocks import MultiHeadAttention, MLP


class ResidualBlock(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1),
        )

        self.rs = nn.Conv2d(in_channels, out_channels, 1,
                            1) if in_channels != out_channels else nn.Identity()

        self.norm = nn.BatchNorm2d(out_channels)

    def forward(self, x):

        return self.norm(self.block(x) + self.rs(x))


class ResNet(nn.Module):

    def __init__(self, in_dim, out_dim, in_channels=3):
        super().__init__()

        self.network = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, 1, 1),
            ResidualBlock(64, 64),
            nn.Conv2d(64, 64, 3, 2, 1),
            ResidualBlock(64, 64),
            nn.Conv2d(64, 128, 3, 2, 1),
            ResidualBlock(128, 128),
            nn.Conv2d(128, 128, 3, 2, 1),
            ResidualBlock(128, 128),
            nn.Conv2d(128, 256, 3, 2, 1),
            ResidualBlock(256, 256),
            nn.Conv2d(256, 256, 3, 2, 1),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, 2, 1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(256 * (in_dim//(2**7))**2, out_dim)
        )

    def forward(self, x):

        return self.network(x)


class ResNet50(nn.Module):

    def __init__(self, freeze: bool = True):
        super().__init__()

        self.resnet = resnet50(weights=ResNet50_Weights.DEFAULT)
        self.resnet.fc = nn.Identity()

        if freeze:
            for param in self.resnet.parameters():
                param.requires_grad = False

    def forward(self, x):
        return self.resnet(x)


class HistTran(nn.Module):

    def __init__(self, n_vars: int, in_dim: int, hidden_dim: int):
        super().__init__()

        self.n_vars = n_vars
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim

        self.pos_embed = nn.Embedding(n_vars, in_dim)
        self.embed = nn.Linear(in_dim, hidden_dim)
        self.mha = MultiHeadAttention(hidden_dim)
        self.mlp = MLP(hidden_dim)

    def forward(self, x):

        e = self.embed(x + self.pos_embed.weight)
        attn_logits, _ = self.mha(e)
        logits = self.mlp(attn_logits)

        return logits
