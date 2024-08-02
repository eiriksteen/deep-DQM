import numpy as np
import pandas as pd
import torch
import pathlib as Path
from torch.utils.data import Dataset
from ..utils import rebin


np.random.seed(42)


class DataStream(Dataset):

    def __init__(
        self,
        data: np.ndarray,
        labels: np.ndarray,
        histo_nbins_dict: dict[str, int],
        source_labels: np.ndarray | None = None,
        whiten: bool = True,
        to_torch: bool = True
    ):

        super().__init__()

        self.histo_nbins_dict = histo_nbins_dict
        self.histo_nbins_dict = {k: v for k,
                                 v in self.histo_nbins_dict.items() if v != 0}

        self.data = data
        self.labels = labels

        self.size, self.num_features, self.num_bins = self.data.shape
        self.num_classes = self.labels.max() + 1
        self.source_labels = source_labels
        self.num_pos = len(self.labels[self.labels == 1])
        self.num_neg = len(self.labels[self.labels == 0])
        self.to_torch = to_torch

        if whiten:
            self.whiten()

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

        if self.source_labels is not None:
            source_labels = self.source_labels[idx]
            if self.to_torch:
                source_labels = torch.tensor(source_labels).float()
            sample["source_labels"] = source_labels

        return sample

    def whiten(self):

        mu = self.data.mean(axis=-1, keepdims=True)
        std = self.data.std(axis=-1, keepdims=True)
        self.data = (self.data - mu) / (std + 1e-06)

    def get_pos_neg_idx(self):

        pos_idx = np.where(self.labels == 1)[0].tolist()
        neg_idx = np.where(self.labels == 0)[0].tolist()

        return pos_idx, neg_idx

    def __str__(self) -> str:

        out = "#"*10
        out += "\nDATASET STATISTICS:"
        out += f"\nNumber of features: {self.num_features}"
        out += f"\nNumber of bins: {self.num_bins}"
        out += f"\nNumber of classes: {self.num_classes}"
        out += f"\nNumber of samples: {self.size}"
        out += f"\nNumber of positive samples: {self.num_pos}"
        out += f"\nNumber of negative samples: {self.num_neg}"
        out += "\n"+"#"*10

        return out


class LHCbDataset(DataStream):

    def __init__(
        self,
        data_path: Path,
        histo_nbins_dict: dict[str, int],
        num_bins: int = 100,
        whiten: bool = True,
        to_torch: bool = True
    ):

        self.df = pd.read_csv(data_path)
        self.data = self.df[[
            c for c in self.df.columns if "var" in c and "err" not in c]].to_numpy()
        self.labels = 1 - self.df["all_OK"].to_numpy()
        self.num_bins = num_bins
        self.histo_nbins_dict = histo_nbins_dict

        self.undo_concat()

        super().__init__(data=self.data,
                         labels=self.labels,
                         histo_nbins_dict=histo_nbins_dict,
                         whiten=whiten,
                         to_torch=to_torch)

    def undo_concat(self):

        histo_nbins = [v for v in self.histo_nbins_dict.values() if v != 0]
        rebinned_data = np.zeros(
            (len(self.data), len(histo_nbins), self.num_bins))

        prev_idx = 0
        for bin_num, size in enumerate(histo_nbins):
            bin = self.data[:, prev_idx:prev_idx + size]

            if size > self.num_bins:
                # Should parallelize this
                for b in range(len(self.data)):
                    rebinned_data[b, bin_num] = rebin(
                        bin[b],
                        new_bin_count=self.num_bins
                    )

            elif size < self.num_bins:
                padding = np.zeros((len(self.data), self.num_bins - size))
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


class SyntheticDataset(DataStream):

    def __init__(
            self,
            size,
            num_variables,
            num_bins,
            nominal_fraction: float = 0.975,
            whiten: bool = True,
            to_torch: bool = True
    ):

        self.nominal_params = [
            {"mu": 5.0, "sigma": 5.0},
            {"mu": -5.0, "sigma": 3.0},
            {"mu": -3.0, "sigma": 2.0},
            {"mu": -2.0, "sigma": 1.0},
            {"mu": -4.0, "sigma": 2.0},
        ]

        self.anomaly_params = [
            {"mu": -3.5, "sigma": 1.5},
            {"mu": -0.5, "sigma": 5},
            {"mu": 0.5, "sigma": 2.5},
            {"mu": 2.5, "sigma": 5.5},
            {"mu": 2.5, "sigma": 2.0},
        ]

        self.transforms = [
            lambda x: x,
            lambda x: np.sin(x)*np.log(x)**2,
            lambda x: np.log(x)*np.exp(x),
            lambda x: np.sqrt(x)*np.log(x) + np.log(x),
            lambda x: np.abs(x)*np.log(x),
        ]

        self.data, self.labels, self.source_labels = self.generate_data(
            size, num_variables, num_bins, nominal_fraction)

        super().__init__(
            data=self.data,
            labels=self.labels,
            histo_nbins_dict={
                f"var_{i}": num_bins for i in range(num_variables)},
            source_labels=self.source_labels,
            whiten=whiten,
            to_torch=to_torch

        )

    def generate_data(
        self,
        size,
        num_variables,
        num_bins,
        total_nominal_fraction,
        change_conditions_every: int = 100,
        num_samples_per_hist_nominal: int = 10000,
        num_samples_per_hist_anomaly: int = 10000
    ):

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
                cur_np = [np.random.choice(self.nominal_params)
                          for _ in range(num_variables)]
                cur_ap = [np.random.choice(self.anomaly_params)
                          for _ in range(num_variables)]

            a_idx_one_hot = np.zeros(num_variables)
            if is_anomaly:
                label = 1
                if num_variables > 1:
                    a_idx = np.random.choice(
                        num_variables, np.random.randint(1, num_variables//2))
                else:
                    a_idx = [0]

                a_idx_one_hot[a_idx] = 1
            else:
                label = 0

            sample = []
            for idx in range(num_variables):
                if is_anomaly and idx in a_idx:
                    p = cur_ap[idx]
                    ns = num_samples_per_hist_anomaly
                else:
                    p = cur_np[idx]
                    ns = num_samples_per_hist_nominal

                s = np.random.normal(p["mu"], p["sigma"], ns)
                transform = variable_transforms[idx]
                s = transform(s)
                hist, _ = np.histogram(
                    s,
                    bins=num_bins,
                    density=False,
                    range=(-15, 15)
                )

                sample.append(hist)

            data.append(sample)
            labels.append(label)
            anomaly_idx.append(a_idx_one_hot)
            prev_is_anomaly = is_anomaly

        return np.array(data), np.array(labels), np.array(anomaly_idx)

    def get_histogram_names(self) -> list[str]:
        return [f"var_{i}" for i in range(self.num_features)]
