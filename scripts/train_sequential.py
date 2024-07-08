import argparse
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    balanced_accuracy_score
)
from dqm.deep_models import (
    MLP,
    ResNet1D,
    CNN1D,
    RefFilter
)
from dqm.shallow_models import LinearRegressor, CopyModel
from dqm.torch_datasets import LHCb2018SequentialDataset
from dqm.replay_buffer import ReplayBuffer
from dqm.settings import DATA_DIR, DEVICE
from dqm.utils import compute_results_summary, plot_metrics_per_step


sns.set_style("white")

plt.rc("figure", figsize=(20, 10))
plt.rc("font", size=13)


def train(model, data, args):

    requires_grad = not isinstance(model, CopyModel)
    model = model.to(DEVICE)

    if requires_grad:
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    loss_fn = nn.CrossEntropyLoss()
    loader = DataLoader(data, batch_size=args.batch_size, shuffle=False)

    replay_buffer = ReplayBuffer(
        data,
        int(args.batch_size * args.replay_ratio),
        pos_ratio=args.replay_pos_ratio)

    total_probs, total_preds, total_labels = [], [], []
    loss_total = 0

    for batch_num, sample in enumerate(tqdm(loader)):

        model.eval()

        histogram = sample["histogram"].to(DEVICE)
        is_anomaly = sample["is_anomaly"].to(DEVICE)

        # Run the model on the current batch
        logits = model(histogram)[:len(is_anomaly)]
        loss = loss_fn(logits, is_anomaly)
        loss_total += loss.item()

        # Don't evaluate the model before it has seen anything
        if batch_num > 0:
            labels = is_anomaly.argmax(dim=-1)
            preds = logits.argmax(dim=-1)
            probs = F.softmax(logits, dim=-1)[:, -1]

            total_labels += labels.detach().cpu().tolist()
            total_preds += preds.detach().cpu().tolist()
            total_probs += probs.detach().cpu().tolist()

        if requires_grad:

            # Train model on current batch
            model.train()
            for _ in range(args.steps_per_batch):

                # Resample from replay buffer each step (simulate epochs)
                histogram_resampled, is_anomaly_resampled = replay_buffer(
                    histogram, is_anomaly)

                logits = model(histogram_resampled)
                loss = loss_fn(logits, is_anomaly_resampled)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            replay_buffer.update(args.batch_size)

        else:
            model.update(is_anomaly)

    return total_probs, total_preds, total_labels


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--steps_per_batch", type=int, default=8)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--n_runs", type=int, default=5)
    parser.add_argument("--replay_pos_ratio", type=float, default=1.0)
    # replay ratio as a fraction of the batch size
    # (0.5 means there are 0.5 * batch_size replayed samples,
    # while there are batch_size new samples)
    parser.add_argument("--replay_ratio", type=float, default=1.5)
    parser.add_argument("--model", type=str, default="mlp")
    args = parser.parse_args()

    models = ["mlp", "resnet1d", "linear", "cnn1d", "ref_filter", "copy"]
    if args.model not in models:
        raise ValueError(
            f"Model {args.model} not supported. Choose from {models}")

    out_dir = Path(f"./sequential_train_results_{args.model}")
    out_dir.mkdir(exist_ok=True)

    with open(out_dir / "config.json", "w") as f:
        json.dump(vars(args), f)

    data = LHCb2018SequentialDataset(
        DATA_DIR / "formatted_dataset_2018.csv",
        # Key to normalize both row and column-wise
        center_and_normalize=True,
        running_center_and_normalize=True,
        to_torch=True
    )

    print("*" * 10)
    print("DATASET:")
    print(f"Number of features: {data.num_features}")
    print(f"Number of classes: {data.num_classes}")
    print(f"Number of samples (train): {len(data)}")
    print(f"Number of positive samples (train): {data.num_pos}")
    print(f"Number of negative samples (train): {data.num_neg}")
    print("*" * 10)

    total_probs, total_preds, total_labels = [], [], []
    for run in range(args.n_runs):

        print(f"RUN {run + 1}/{args.n_runs}")

        if args.model == "mlp":
            model = MLP(data.num_features, data.num_classes)
        elif args.model == "resnet1d":
            model = ResNet1D(1, 1, data.num_classes)
        elif args.model == "linear":
            model = LinearRegressor(data.num_features, data.num_classes)
        elif args.model == "cnn1d":
            model = CNN1D(1, 1, data.num_classes)
        elif args.model == "ref_filter":
            model = RefFilter(data.num_features, 512)
        elif args.model == "copy":
            model = CopyModel(data.num_classes)
        else:
            raise ValueError(f"Model {args.model} not supported")

        print(f"MODEL SIZE: {sum(p.numel() for p in model.parameters())}")

        probs, preds, labels = train(
            model,
            data,
            args
        )

        print(f"BALANCED ACCURACY: {balanced_accuracy_score(labels, preds)}")
        print(f"ROC AUC: {roc_auc_score(labels, probs)}")
        print(f"AP: {average_precision_score(labels, probs)}")

        total_probs.append(probs)
        total_preds.append(preds)
        total_labels.append(labels)

    total_probs = np.array(total_probs)
    total_preds = np.array(total_preds)
    total_labels = np.array(total_labels)

    print("*" * 10)
    print("FINAL RESULTS")
    results_summary = compute_results_summary(
        total_probs, total_preds, total_labels)
    print(results_summary)
    print("*" * 10)

    with open(out_dir / "results.txt", "w") as f:
        f.write(results_summary)

    plot_metrics_per_step(total_probs, total_preds, total_labels, out_dir)
