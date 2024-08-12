from matplotlib import pyplot as plt
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from pathlib import Path
from ..settings import MVTEC_DIR


class MVTECDataset(Dataset):

    def __init__(
            self,
            resolution: int = 256,
            to_tensor: bool = True,
            normalize: bool = True,
            path: Path = MVTEC_DIR,
    ):

        self.resolution = resolution
        self.to_tensor = to_tensor
        self.normalize = normalize
        self.path = path

        self.classes_dir = [d for d in self.path.iterdir() if d.is_dir()]
        self.file_dirs = self.get_file_paths_per_class()
        self.size_per_class = [len(files) for files in self.file_dirs]
        self.size = sum(self.size_per_class)
        self.size_per_class_cusum = torch.cumsum(
            torch.tensor(self.size_per_class), 0)

        self.labels = np.array([0 if "good" in str(
            file) else 1 for files in self.file_dirs for file in files])

    def __len__(self):
        return self.size

    def __getitem__(self, idx):

        class_idx = np.searchsorted(self.size_per_class_cusum, idx)
        file_idx = idx - self.size_per_class_cusum[class_idx]
        file_path = self.file_dirs[class_idx][file_idx]
        image = Image.open(file_path)
        image = np.array(image).astype(np.float32)

        is_anomaly = self.labels[idx]

        if self.normalize:
            image = image / 255

        if self.to_tensor:
            image = image[:, :, None] if len(image.shape) == 2 else image
            image = torch.tensor(image).permute(2, 0, 1)
            is_anomaly = torch.tensor([is_anomaly]).float()
            class_idx = torch.tensor([class_idx]).float()

            image = transforms.Resize(self.resolution)(image)

        return {"inp": image, "is_anomaly": is_anomaly, "class_idx": class_idx}

    def get_file_paths_per_class(self) -> list[Path]:

        file_dirs = [[
            str(file)
            for subdir in ["train", "test"]
            for file in (class_dir / subdir).rglob("*")
            if file.is_file() and file.suffix in [".png", ".jpg"]
        ] for class_dir in self.classes_dir]

        for fdir in file_dirs:
            np.random.shuffle(fdir)

        return file_dirs

    def get_pos_neg_idx(self):
        pos_idx = np.where(self.labels == 1)[0].tolist()
        neg_idx = np.where(self.labels == 0)[0].tolist()

        return pos_idx, neg_idx

    def __str__(self) -> str:
        out = "MVTEC Dataset\n" + \
              "Size: {}\n".format(self.size) + \
              "Classes: {}\n".format(len(self.classes_dir)) + \
              "Resolution: {}\n".format(self.resolution) + \
              "Normalize: {}\n".format(self.normalize) + \
              "Num Pos: {}\n".format(len(np.where(self.labels == 1)[0])) + \
              "Num Neg: {}\n".format(len(np.where(self.labels == 0)[0]))

        return out
