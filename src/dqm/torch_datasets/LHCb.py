import numpy as np
import pandas as pd
import torch
import pathlib as Path
import torch.nn.functional as F
from torch.utils.data import Dataset
from ..utils import rebin
from ..settings import HISTO_NBINS_DICT_2018, HISTO_NBINS_DICT_2023


class LHCbDataset(Dataset):

    def __init__(
        self,
        data_path: Path,
        year: int,
        num_bins: int = 100,
        whiten: bool = True,
        whiten_running: bool = True,
        to_torch: bool = True,
        undo_concat: bool = False
    ):

        super().__init__()

        if year == 2018:
            self.histo_nbins_dict = HISTO_NBINS_DICT_2018
        elif year == 2023:
            self.histo_nbins_dict = HISTO_NBINS_DICT_2023
        else:
            raise ValueError("Year must be either 2018 or 2023")

        self.histo_nbins_dict = {k: v for k,
                                 v in self.histo_nbins_dict.items() if v != 0}

        self.to_torch = to_torch

        self.df = pd.read_csv(data_path)
        self.data = self.df[[
            c for c in self.df.columns if "var" in c and "err" not in c]].to_numpy()
        self.labels = 1 - self.df["all_OK"].to_numpy()

        self.size, self.num_features = self.data.shape
        self.num_classes = self.labels.max() + 1
        self.num_pos = len(self.labels[self.labels == 1])
        self.num_neg = len(self.labels[self.labels == 0])
        self.num_bins = num_bins

        if undo_concat:
            self.undo_concat()

        if whiten:
            self.whiten()

        if whiten_running:
            self.whiten_running()

    def __len__(self):
        return self.size

    def __getitem__(self, idx):

        histogram = self.data[idx]
        is_anomaly = self.labels[idx]

        if self.to_torch:
            histogram = torch.tensor(histogram).float()
            is_anomaly = torch.tensor([is_anomaly]).float()

        sample = {
            "histogram": histogram,
            "is_anomaly": is_anomaly
        }

        return sample

    def whiten(self):

        # Try normalizing only the non-zeros
        mu = self.data.mean(axis=-1, keepdims=True)
        std = self.data.std(axis=-1, keepdims=True)
        std = np.where(std < 1e-06, 1e-06, std)
        self.data = (self.data - mu) / std

    def whiten_running(self):

        data_cusum = np.cumsum(self.data, axis=0)
        cu_slice = np.s_[:, None] if self.data.ndim == 2 else np.s_[
            :, None, None]
        running_mean = data_cusum / np.arange(1, self.size + 1)[cu_slice]

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

    def undo_concat(self):

        histo_nbins = [v for v in self.histo_nbins_dict.values() if v != 0]
        rebinned_data = np.zeros((self.size, len(histo_nbins), self.num_bins))

        prev_idx = 0
        for bin_num, size in enumerate(histo_nbins):
            bin = self.data[:, prev_idx:prev_idx + size]

            if size > self.num_bins:
                # Should parallelize this
                for b in range(self.size):
                    rebinned_data[b, bin_num] = rebin(
                        bin[b], new_bin_count=self.num_bins)
            elif size < self.num_bins:
                padding = np.zeros((self.size, self.num_bins - size))
                rebinned_data[:, bin_num] = np.concatenate(
                    (bin, padding), axis=-1)
            else:
                rebinned_data[:, bin_num] = bin

            prev_idx += size

        self.data = rebinned_data
        self.num_features = len(histo_nbins)

        print(f"Data shape after undoing concat: {self.data.shape}")

    def get_histogram_names(self) -> list[str]:
        return list(self.histo_nbins_dict.keys())
