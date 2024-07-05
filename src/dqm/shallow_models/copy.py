import torch
import torch.nn as nn
import torch.nn.functional as F


class CopyModel(nn.Module):

    def __init__(self, num_classes):
        super().__init__()

        self.num_classes = num_classes
        self.prev_label = None

    def forward(self, x):

        if self.prev_label is None:
            return F.one_hot(torch.randint(
                self.num_classes,
                [x.shape[0]]).long().to(x.device), num_classes=self.num_classes).float()
        else:
            return self.prev_label

    def update(self, x):
        self.prev_label = x
