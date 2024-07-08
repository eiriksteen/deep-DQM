import numpy as np
import torch
from .torch_datasets import LHCb2018SequentialDataset


class ReplayBuffer:

    def __init__(
        self,
        dataset: LHCb2018SequentialDataset,
        buffer_size: int,
        pos_ratio: float = 0.5
    ):

        self.dataset = dataset
        self.buffer_size = buffer_size
        self.pos_ratio = pos_ratio
        self.pos_buffer_size = int(buffer_size * pos_ratio)
        self.neg_buffer_size = buffer_size - self.pos_buffer_size
        self.cur_idx = 0
        self.num_features = dataset.num_features
        self.num_classes = dataset.num_classes
        self.pos_idx, self.neg_idx = self.dataset.get_pos_neg_idx()

    def get_rand_past_samples(self):

        idx_neg = self.neg_idx[:np.searchsorted(self.neg_idx, self.cur_idx)]
        idx_pos = self.pos_idx[:np.searchsorted(self.pos_idx, self.cur_idx)]
        hists, labels = [], []

        if idx_neg and self.neg_buffer_size > 0:

            replace_neg = len(idx_neg) < self.neg_buffer_size

            rand_idx_neg = np.random.choice(
                idx_neg, self.neg_buffer_size, replace=replace_neg)

            neg_samples = [self.dataset[i] for i in rand_idx_neg]

            hists.append(torch.stack([s["histogram"] for s in neg_samples]))
            labels.append(torch.stack([s["is_anomaly"] for s in neg_samples]))

        if idx_pos and self.pos_buffer_size > 0:

            replace_pos = len(idx_pos) < self.pos_buffer_size

            rand_idx_pos = np.random.choice(
                idx_pos, self.pos_buffer_size, replace=replace_pos)

            pos_samples = [self.dataset[i] for i in rand_idx_pos]

            hists.append(torch.stack([s["histogram"] for s in pos_samples]))
            labels.append(torch.stack([s["is_anomaly"] for s in pos_samples]))

        if not hists:
            raise ValueError("No samples in the replay buffer")
        else:
            return torch.cat(hists, dim=0), torch.cat(labels, dim=0)

    def update(self, num_steps: int):
        self.cur_idx += num_steps

    def __call__(self, hist: torch.Tensor, labels: torch.Tensor):

        try:
            rand_hist, rand_labels = self.get_rand_past_samples()
        except ValueError:
            return hist, labels
        else:
            rand_hist = rand_hist.to(hist.device)
            rand_labels = rand_labels.to(labels.device)
            augmented_hist = torch.cat([hist, rand_hist], dim=0)
            augmented_labels = torch.cat([labels, rand_labels], dim=0)
            rand_idx = torch.randperm(len(augmented_hist))
            augmented_hist, augmented_labels = augmented_hist[rand_idx], augmented_labels[rand_idx]

            return augmented_hist, augmented_labels
