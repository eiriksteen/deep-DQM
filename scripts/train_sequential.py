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
from pprint import pprint
from tqdm import tqdm
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score
)
from dqm.models import (
    MLP,
    ResNet1D,
    LinearRegressor,
    CNN,
    RefFilter
)
from dqm.torch_datasets import LHCbSequentialDataset
from dqm.settings import DATA_DIR, DEVICE
from dqm.utils import compute_results_summary


sns.set_style("white")

plt.rc("figure", figsize=(20, 10))
plt.rc("font", size=13)


def train(
        model,
        data,
        steps_per_batch=1,
        lr=0.001,
        batch_size=1):

    model = model.to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    loader = DataLoader(data, batch_size=batch_size, shuffle=False)
    total_labels, total_preds, total_probs = [], [], []
    auroc_per_step, auprc_per_step = [], []
    loss_total = 0

    for sample in tqdm(loader):

        histogram = sample["histogram"].to(DEVICE)
        is_anomaly = sample["is_anomaly"].to(DEVICE)
        histogram = histogram.unsqueeze(1) if isinstance(
            model, ResNet1D) or isinstance(model, CNN) else histogram

        # Run the model on the current batch
        logits = model(histogram)
        loss = loss_fn(logits, is_anomaly)

        loss_total += loss.item()
        total_labels += is_anomaly.argmax(
            dim=-1).detach().cpu().tolist()
        total_preds += logits.argmax(
            dim=-1).detach().cpu().tolist()
        total_probs += F.softmax(logits,
                                 dim=-1)[:, -1].detach().cpu().tolist()

        if 0 in total_labels and 1 in total_labels:
            auroc_per_step.append(roc_auc_score(total_labels, total_probs))
            auprc_per_step.append(
                average_precision_score(total_labels, total_probs))

        # Train model parameters on current batch
        for _ in range(steps_per_batch):
            logits = model(histogram)
            loss = loss_fn(logits, is_anomaly)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    metrics = {
        "accuracy": accuracy_score(total_labels, total_preds),
        "precision": precision_score(total_labels, total_preds),
        "recall": recall_score(total_labels, total_preds),
        "f1": f1_score(total_labels, total_preds),
        "auroc": roc_auc_score(total_labels, total_probs),
        "auprc": average_precision_score(total_labels, total_probs)
    }

    return metrics, auroc_per_step, auprc_per_step


def plot_stepwise_results(metric_per_step, metric_name, out_dir):

    metric_per_step = np.array(metric_per_step)
    mu = metric_per_step.mean(axis=0)
    std = metric_per_step.std(axis=0)

    plt.title(f"{metric_name} per step")
    plt.plot(mu)
    plt.fill_between(
        np.arange(len(mu)),
        mu - 1.96 * std,
        mu + 1.96 * std,
        alpha=0.3
    )
    plt.savefig(out_dir / f"{metric_name}.png")
    plt.close()


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--steps_per_batch", type=int, default=1)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--n_runs", type=int, default=5)
    parser.add_argument("--model", type=str, default="mlp")
    args = parser.parse_args()

    models = ["mlp", "resnet1d", "linear", "cnn", "ref_filter"]
    if args.model not in models:
        raise ValueError(
            f"Model {args.model} not supported. Choose from {models}")

    out_dir = Path(f"./sequential_train_results_{args.model}")
    out_dir.mkdir(exist_ok=True)

    with open(out_dir / "config.json", "w") as f:
        json.dump(vars(args), f)

    data = LHCbSequentialDataset(
        DATA_DIR / "formatted_dataset_2018.csv",
        center_and_normalize=True,
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

    total_metrics, total_auroc_per_step, total_auprc_per_step = [], [], []
    for run in range(args.n_runs):

        print(f"RUN {run + 1}/{args.n_runs}")

        if args.model == "mlp":
            model = MLP(data.num_features, data.num_classes)
        elif args.model == "resnet1d":
            model = ResNet1D(1, 1, data.num_classes)
        elif args.model == "linear":
            model = LinearRegressor(data.num_features, data.num_classes)
        elif args.model == "cnn":
            model = CNN(1, 1, data.num_classes)
        elif args.model == "ref_filter":
            model = RefFilter(data.num_features, 512)
        else:
            raise ValueError(f"Model {model} not supported")

        print(f"MODEL SIZE: {sum(p.numel() for p in model.parameters())}")

        metrics, auroc_per_step, auprc_per_step = train(
            model,
            data,
            steps_per_batch=args.steps_per_batch,
            lr=args.lr,
            batch_size=args.batch_size
        )

        total_metrics.append(metrics)
        total_auroc_per_step.append(auroc_per_step)
        total_auprc_per_step.append(auprc_per_step)

        pprint(metrics)

    print("*" * 10)
    print("FINAL RESULTS")
    results_summary = compute_results_summary(total_metrics)
    print(results_summary)
    print("*" * 10)

    with open(out_dir / "results_summary.txt", "w") as f:
        f.write(results_summary)

    plot_stepwise_results(total_auroc_per_step, "auroc", out_dir)
    plot_stepwise_results(total_auprc_per_step, "auprc", out_dir)
