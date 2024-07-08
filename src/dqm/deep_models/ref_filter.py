import torch
import torch.nn as nn
import torch.nn.functional as F


class RefFilter(nn.Module):

    def __init__(self, in_dim, hidden_dim):
        super().__init__()

        self.in_dim = in_dim

        self.proj = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            # nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.nominal_embedding = nn.Embedding(1, hidden_dim)
        self.anomaly_embedding = nn.Embedding(1, hidden_dim)

    def forward(self, x):

        p = self.proj(x)
        sim1 = F.cosine_similarity(p, self.nominal_embedding.weight)
        sim2 = F.cosine_similarity(p, self.anomaly_embedding.weight)
        sim = torch.stack([sim1, sim2], dim=-1)
        out = F.softmax(sim, dim=-1)

        return out
