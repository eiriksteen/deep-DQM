import numpy as np
import torch
from .torch_datasets import LHCbDataset


class ReplayBuffer:

    def __init__(
            self,
            dataset: LHCbDataset,
            store_ref: bool = True,
            classes: str = "both"
    ):

        self.dataset = dataset
        self.store_ref = store_ref
        self.classes = classes
        self.cur_idx = 0
        self.pos_idx, self.neg_idx = self.dataset.get_pos_neg_idx()
        self.refs = [] if store_ref else None

        if classes not in ["both", "pos", "neg"]:
            raise ValueError(
                "Invalid value for classes. Must be one of 'both', 'pos', 'neg'"
            )

    def get_rand_past_samples(self, num_neg_to_sample: int, num_pos_to_sample: int):

        cur_idx_neg = self.neg_idx[:np.searchsorted(
            self.neg_idx, self.cur_idx)]
        cur_idx_pos = self.pos_idx[:np.searchsorted(
            self.pos_idx, self.cur_idx)]
        hists, labels, refs = [], [], []

        if cur_idx_neg and num_neg_to_sample > 0:

            dist = np.arange(1, len(cur_idx_neg)+1)
            dist = dist / dist.sum()

            rand_idx_neg = np.random.choice(
                cur_idx_neg,
                size=num_neg_to_sample,
                p=dist,
                replace=num_neg_to_sample > len(cur_idx_neg)
            )

            neg_samples = [self.dataset[i] for i in rand_idx_neg]

            hists.append(torch.stack([s["histogram"] for s in neg_samples]))
            labels.append(torch.stack([s["is_anomaly"] for s in neg_samples]))

            if self.store_ref:
                ref_samples = [self.refs[i] for i in rand_idx_neg]
                refs.append(torch.stack(ref_samples))

        if cur_idx_pos and num_pos_to_sample > 0:

            dist = np.arange(1, len(cur_idx_pos)+1)
            dist = dist / dist.sum()

            rand_idx_pos = np.random.choice(
                cur_idx_pos,
                size=num_pos_to_sample,
                p=dist,
                replace=num_pos_to_sample > len(cur_idx_pos)
            )

            pos_samples = [self.dataset[i] for i in rand_idx_pos]

            hists.append(torch.stack([s["histogram"] for s in pos_samples]))
            labels.append(torch.stack([s["is_anomaly"] for s in pos_samples]))

            if self.store_ref:
                ref_samples = [self.refs[i] for i in rand_idx_pos]
                refs.append(torch.stack(ref_samples))

        if not hists:
            raise ValueError("No samples in the replay buffer")
        else:
            return {
                "hist": torch.cat(hists, dim=0),
                "labels": torch.cat(labels, dim=0),
                "refs": torch.cat(refs, dim=0) if self.store_ref else None
            }

    def update(self, num_steps: int):
        self.cur_idx += num_steps

    def __call__(
            self,
            hist: torch.Tensor,
            labels: torch.Tensor | None = None,
            refs: torch.Tensor | None = None):

        num_pos = int(labels[labels == 1].sum().item())

        if self.classes == "both":
            num_neg_to_sample = num_pos
            num_pos_to_sample = len(labels) - num_pos
        elif self.classes == "pos":
            num_neg_to_sample = 0
            num_pos_to_sample = len(labels)
        else:
            num_neg_to_sample = len(labels)
            num_pos_to_sample = 0

        try:
            res = self.get_rand_past_samples(
                num_neg_to_sample, num_pos_to_sample
            )

            rand_hist = res["hist"]
            rand_labels = res["labels"]
            rand_refs = res["refs"]

        except ValueError:
            return hist, labels, refs
        else:
            rand_hist = rand_hist.to(hist.device)
            rand_labels = rand_labels.to(labels.device)
            augmented_hist = torch.cat([hist, rand_hist], dim=0)
            augmented_labels = torch.cat([labels, rand_labels], dim=0)

            rand_idx = torch.randperm(len(augmented_hist))
            out_hist = augmented_hist[rand_idx]
            out_labels = augmented_labels[rand_idx]

            if self.store_ref:
                augmented_refs = torch.cat([refs, rand_refs], dim=0)
                out_refs = augmented_refs[rand_idx]
            else:
                out_refs = None

            return out_hist, out_labels, out_refs

    def add_ref(self, ref: torch.Tensor):
        self.refs.append(ref)
