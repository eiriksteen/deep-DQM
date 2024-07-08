import numpy as np
import pandas as pd
import torch
import pathlib as Path
import torch.nn.functional as F
from torch.utils.data import Dataset


class LHCb2018SequentialDataset(Dataset):

    def __init__(
            self,
            data_path: Path,
            center_and_normalize: bool = True,
            running_center_and_normalize: bool = True,
            to_torch: bool = True):

        super().__init__()

        self.to_torch = to_torch

        self.df = pd.read_csv(data_path)
        self.data = self.df[[
            c for c in self.df.columns if "var" in c and "err" not in c]].to_numpy()
        self.labels = 1 - self.df["all_OK"].to_numpy()

        self.size, self.num_features = self.data.shape
        self.num_classes = self.labels.max() + 1
        self.num_pos = len(self.labels[self.labels == 1])
        self.num_neg = len(self.labels[self.labels == 0])

        if center_and_normalize:
            self.whiten()

        if running_center_and_normalize:
            self.whiten_running()

    def __len__(self):
        return self.size

    def __getitem__(self, idx):

        histogram = self.data[idx]
        is_anomaly = self.labels[idx]

        if self.to_torch:
            histogram = torch.tensor(histogram).float()
            is_anomaly = F.one_hot(torch.tensor(is_anomaly).long(),
                                   num_classes=self.num_classes).float()

        sample = {
            "histogram": histogram,
            "is_anomaly": is_anomaly
        }

        return sample

    def whiten(self):

        mu = self.data.mean(axis=1, keepdims=True)
        std = self.data.std(axis=1, keepdims=True)
        std = np.where(std < 1e-06, 1e-06, std)
        self.data = (self.data - mu) / std

    def whiten_running(self):

        data_cusum = np.cumsum(self.data, axis=0)
        running_mean = data_cusum / np.arange(1, self.size + 1)[:, None]

        running_std = np.zeros_like(self.data)
        for t in range(1, self.size):
            running_std[t] = np.sqrt(
                (t * running_std[t-1]**2 + (self.data[t] - running_mean[t])**2) / (t+1))

        running_std = np.where(running_std < 1e-06, 1e-06, running_std)
        self.data = (self.data - running_mean) / running_std

        # TODO: Implement EMA normalization

    def get_pos_neg_idx(self):

        pos_idx = np.where(self.labels == 1)[0].tolist()
        neg_idx = np.where(self.labels == 0)[0].tolist()

        return pos_idx, neg_idx


class LHCb2018TempSplitDataset(Dataset):

    def __init__(
        self,
        data_path: Path,
        split: str,
        train_frac: float = 0.8,
        seed: int = 1,
        center_and_normalize: bool = True,
        to_torch: bool = True,
        upsample_positive: bool = False
    ):
        super().__init__()

        np.random.seed(seed)
        torch.manual_seed(seed)

        self.split = split
        self.train_frac = train_frac
        self.to_torch = to_torch

        self.df = pd.read_csv(data_path)
        self.complete_data = self.df[[
            c for c in self.df.columns if "var" in c and "err" not in c]].to_numpy()
        self.complete_labels = 1 - self.df["all_OK"].to_numpy()

        self.train_idx, self.val_idx, self.test_idx = self.split_data()

        if self.split == "train":
            self.idx = self.train_idx
        elif self.split == "val":
            self.idx = self.val_idx
        elif self.split == "test":
            self.idx = self.test_idx

        self.data = self.complete_data[self.idx]
        self.labels = self.complete_labels[self.idx]

        if center_and_normalize:
            self.preprocess()

        if upsample_positive:
            self.upsample_positive()

        self.size, self.num_features = self.data.shape
        self.num_classes = self.labels.max() + 1
        self.num_pos = len(self.labels[self.labels == 1])
        self.num_neg = len(self.labels[self.labels == 0])

    def __len__(self):
        return self.size

    def __getitem__(self, idx):

        histogram = self.data[idx]
        is_anomaly = self.labels[idx]

        if self.to_torch:
            histogram = torch.tensor(histogram).float()
            is_anomaly = F.one_hot(torch.tensor(is_anomaly).long(),
                                   num_classes=self.num_classes).float()

        sample = {
            "histogram": histogram,
            "is_anomaly": is_anomaly
        }

        return sample

    def split_data(self):

        train_size = int(self.train_frac * self.complete_data.shape[0])
        val_size = (self.complete_data.shape[0] - train_size) // 2
        test_size = self.complete_data.shape[0] - train_size - val_size

        train_idx = np.arange(train_size)
        val_idx = np.arange(train_size, train_size + val_size)
        test_idx = np.arange(train_size + val_size,
                             train_size + val_size + test_size)

        return train_idx, val_idx, test_idx

    def preprocess(self, eps=1e-6):

        mu = self.complete_data[self.train_idx].mean(axis=0)
        std = self.complete_data[self.train_idx].std(axis=0)
        std = np.where(std < eps, eps, std)

        self.data = (self.data - mu) / std

    def upsample_positive(self):

        nonzero_idx = self.idx[np.nonzero(self.labels)[0]]
        pos_count = len(self.labels[self.labels == 1])
        neg_count = len(self.labels[self.labels == 0])
        pos_idx_tiled = np.tile(nonzero_idx, neg_count//pos_count)
        idx_upsampled = np.concatenate((self.idx, pos_idx_tiled), axis=0)

        self.idx = idx_upsampled
        self.data = self.complete_data[self.idx]
        self.labels = self.complete_labels[self.idx]
