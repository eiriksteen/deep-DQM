import torch
from torch.utils.data import Dataset
from pathlib import Path


class DummyDataset(Dataset):
    def __init__(self, data_path: Path):

        self.data_path = data_path
        # get classes from directory names
        self.class_dirs = [x for x in data_path.rglob("*") if x.is_dir()]

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]

    def compute_num_samples_per_class(self):
        self.num_samples_per_class = [
            len(list(x.glob("*"))) for x in self.class_dirs]
        self.num_samples = sum(self.num_samples_per_class)
