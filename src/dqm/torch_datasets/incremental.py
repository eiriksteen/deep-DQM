from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import torch
import pathlib as Path
import torch.nn.functional as F
from torch.utils.data import Dataset
from ..utils import rebin
from ..settings import HISTO_NBINS_DICT_2018, HISTO_NBINS_DICT_2023


class IncrementalDataset(Dataset):

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

        if self.anomaly_idx is not None:
            anomaly_idx = self.anomaly_idx[idx]
            if self.to_torch:
                anomaly_idx = torch.tensor(anomaly_idx).float()
            sample["anomaly_idx"] = anomaly_idx

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


class LHCbDataset(IncrementalDataset):

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
        self.anomaly_idx = None
        self.num_pos = len(self.labels[self.labels == 1])
        self.num_neg = len(self.labels[self.labels == 0])
        self.num_bins = num_bins

        if undo_concat:
            self.undo_concat()

        if whiten:
            self.whiten()

        if whiten_running:
            self.whiten_running()

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


class SyntheticDataset(IncrementalDataset):
    def __init__(
            self,
            size,
            num_variables,
            num_bins,
            nominal_fraction=0.975,
            whiten: bool = True,
            whiten_running: bool = True,
            to_torch: bool = True
    ):

        self.size = size
        self.to_torch = to_torch
        self.num_bins = num_bins
        self.num_features = num_variables
        self.num_classes = 2

        self.nominal_params = [
            {"mu": 0.0, "sigma": 1.0},
            {"mu": 0.0, "sigma": 2.0},
            {"mu": 1.0, "sigma": 1.0},
            {"mu": 1.0, "sigma": 2.0},
            {"mu": 2.0, "sigma": 1.0},
            {"mu": 2.0, "sigma": 2.0},
        ]

        self.anomaly_params = [
            {"mu": 0.0, "sigma": 3.5},
            {"mu": 0.5, "sigma": 4.5},
            {"mu": 1.5, "sigma": 6.0},
            {"mu": 2.5, "sigma": 4.0},
            {"mu": 0.0, "sigma": 5.5},
        ]

        self.transforms = [
            lambda x: x,
            lambda x: x**2,
            # lambda x: x**3,
            lambda x: np.sin(x),
            # lambda x: np.cos(x),
            # lambda x: np.exp(x),
            # lambda x: np.log(x),
            # lambda x: np.sqrt(x),
            lambda x: np.abs(x),
            # lambda x: np.sin(x)*np.sin(x)
        ]

        self.data, self.labels, self.anomaly_idx = self.generate_data(
            size, num_variables, num_bins, nominal_fraction)

        if whiten:
            self.whiten()

        if whiten_running:
            self.whiten_running()

        self.num_pos = len(self.labels[self.labels == 1])
        self.num_neg = len(self.labels[self.labels == 0])

    def generate_data(
        self,
        size,
        num_variables,
        num_bins,
        total_nominal_fraction=0.99,
        change_conditions_every: int = 100,
        num_samples_per_hist_nominal: int = 1000,
        num_samples_per_hist_anomaly: int = 1000
    ):

        nominal_p = np.random.choice(
            self.nominal_params,
            size=size//change_conditions_every,
            replace=True).tolist()
        anomaly_p = np.random.choice(
            self.anomaly_params,
            size=size//change_conditions_every,
            replace=True).tolist()
        variable_transforms = np.random.choice(
            self.transforms,
            size=num_variables,
            replace=True).tolist()

        data, labels, anomaly_idx = [], [], []
        prev_is_anomaly = False
        for i in range(size):

            if prev_is_anomaly:
                is_anomaly = np.random.rand() < 0.8
            else:
                is_anomaly = not np.random.rand() < total_nominal_fraction

            if i % change_conditions_every == 0:
                cur_np = nominal_p.pop()
                cur_ap = anomaly_p.pop()

            a_idx_one_hot = np.zeros(num_variables)
            if is_anomaly:
                label = 1
                a_idx = np.random.choice(
                    num_variables, np.random.randint(1, num_variables//2))
                a_idx_one_hot[a_idx] = 1
            else:
                label = 0

            sample = []
            for idx in range(num_variables):
                if is_anomaly and idx in a_idx:
                    p = cur_ap
                    ns = num_samples_per_hist_anomaly
                else:
                    p = cur_np
                    ns = num_samples_per_hist_nominal

                s = np.random.normal(p["mu"], p["sigma"], ns)
                # s = np.random.normal(p["mu"], p["sigma"], num_bins)
                transform = variable_transforms[idx]
                s = transform(s)
                hist = np.histogram(s, bins=num_bins, density=True)[0]
                sample.append(hist)
                # sample.append(s)

            data.append(sample)
            labels.append(label)
            anomaly_idx.append(a_idx_one_hot)
            prev_is_anomaly = is_anomaly

        return np.array(data), np.array(labels), np.array(anomaly_idx)

    def get_histogram_names(self) -> list[str]:
        return [f"var_{i}" for i in range(self.num_features)]
