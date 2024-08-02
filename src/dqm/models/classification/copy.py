import torch
import torch.nn as nn
import torch.nn.functional as F


class CopyModel(nn.Module):

    def __init__(self):
        super().__init__()

        self.prev_label = None

    def forward(self, x):

        if self.prev_label is None:
            out = torch.randint(0, 1, (x.shape[0], 1)).float().to(x.device)
        else:
            out = self.prev_label

        return {"logits": out}

    def update(self, x):
        self.prev_label = x
