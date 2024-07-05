import argparse
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
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
from dqm.models import MLP, ResNet1D
from dqm.torch_datasets import LHCbTempSplitDataset
from dqm.settings import DATA_DIR, DEVICE
from dqm.utils import compute_results_summary


def train(
        model,
        train_data,
        val_data,
        test_data,
        epochs=10,
        lr=0.001,
        batch_size=32):

    model = model.to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

    min_val_loss = float("inf")
    for epoch in range(epochs):

        print(f"RUNNING EPOCH {epoch+1}")
        print("TRAINING")
        train_loss = 0
        for sample in tqdm(train_loader):

            histogram = sample["histogram"].to(DEVICE)
            is_anomaly = sample["is_anomaly"].to(DEVICE)
            histogram = histogram.unsqueeze(1) if isinstance(
                model, ResNet1D) else histogram

            optimizer.zero_grad()
            logits = model(histogram)
            loss = loss_fn(logits, is_anomaly)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        val_loss = 0
        total_labels, total_preds, total_probs = [], [], []
        print("VALIDATING")
        for sample in tqdm(val_loader):

            histogram = sample["histogram"].to(DEVICE)
            is_anomaly = sample["is_anomaly"].to(DEVICE)
            histogram = histogram.unsqueeze(1) if isinstance(
                model, ResNet1D) else histogram

            logits = model(histogram)
            loss = loss_fn(logits, is_anomaly)
            val_loss += loss.item()

            total_labels += is_anomaly.argmax(
                dim=-1).detach().cpu().tolist()
            total_preds += logits.argmax(
                dim=-1).detach().cpu().tolist()
            total_probs += F.softmax(logits, dim=-
                                     1)[:, 1].detach().cpu().tolist()

        metrics = {
            "train_loss": train_loss / len(train_loader),
            "val_loss": val_loss / len(val_loader),
            "val_accuracy": accuracy_score(total_labels, total_preds),
            "val_precision": precision_score(total_labels, total_preds),
            "val_recall": recall_score(total_labels, total_preds),
            "val_f1": f1_score(total_labels, total_preds),
            "val_auroc": roc_auc_score(total_labels, total_probs),
            "val_auprc": average_precision_score(total_labels, total_probs)
        }

        pprint(metrics)

        if metrics["val_loss"] < min_val_loss:
            print("New min loss, saving model...")
            min_val_loss = metrics["val_loss"]
            torch.save(model.state_dict(), out_dir / "model")

    model.load_state_dict(torch.load("model"))
    test_loss = 0
    total_labels, total_preds, total_probs = [], [], []
    print("TESTING")
    for sample in tqdm(test_loader):

        histogram = sample["histogram"].to(DEVICE)
        is_anomaly = sample["is_anomaly"].to(DEVICE)
        histogram = histogram.unsqueeze(1) if isinstance(
            model, ResNet1D) else histogram

        logits = model(histogram)
        loss = loss_fn(logits, is_anomaly)
        test_loss += loss.item()

        total_labels += is_anomaly.argmax(
            dim=-1).detach().cpu().tolist()
        total_preds += logits.argmax(
            dim=-1).detach().cpu().tolist()
        total_probs += F.softmax(logits, dim=-1)[:, -1].detach().cpu().tolist()

    test_metrics = {
        "test_loss": test_loss / len(test_loader),
        "test_accuracy": accuracy_score(total_labels, total_preds),
        "test_precision": precision_score(total_labels, total_preds),
        "test_recall": recall_score(total_labels, total_preds),
        "test_f1": f1_score(total_labels, total_preds),
        "test_auroc": roc_auc_score(total_labels, total_probs),
        "test_auprc": average_precision_score(total_labels, total_probs)
    }

    return test_metrics


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--n_runs", type=int, default=5)
    parser.add_argument("--train_frac", type=float, default=0.9)
    parser.add_argument("--model", type=str, default="mlp")
    args = parser.parse_args()

    models = ["mlp", "resnet1d"]
    if args.model not in models:
        raise ValueError(
            f"Model {args.model} not supported. Choose from {models}")

    out_dir = Path(f"./temp_split_results_{args.model}")
    out_dir.mkdir(exist_ok=True)

    with open(out_dir / "config.json", "w") as f:
        json.dump(vars(args), f)

    train_data = LHCbTempSplitDataset(
        DATA_DIR / "formatted_dataset_2018.csv",
        "train",
        train_frac=args.train_frac,
        upsample_positive=True)
    val_data = LHCbTempSplitDataset(
        DATA_DIR / "formatted_dataset_2018.csv",
        "val",
        train_frac=args.train_frac,
    )
    test_data = LHCbTempSplitDataset(
        DATA_DIR / "formatted_dataset_2018.csv",
        "test",
        train_frac=args.train_frac,
    )

    print("*" * 10)
    print("DATASET:")
    print(f"Number of features: {train_data.num_features}")
    print(f"Number of classes: {train_data.num_classes}")
    print(f"Number of samples (train): {len(train_data)}")
    print(f"Number of samples (val): {len(val_data)}")
    print(f"Number of samples (test): {len(test_data)}")
    print(f"Number of positive samples (train): {train_data.num_pos}")
    print(f"Number of negative samples (train): {train_data.num_neg}")
    print(f"Number of positive samples (val): {val_data.num_pos}")
    print(f"Number of negative samples (val): {val_data.num_neg}")
    print(f"Number of positive samples (test): {test_data.num_pos}")
    print(f"Number of negative samples (test): {test_data.num_neg}")
    print("*" * 10)

    total_test_metrics = []
    for run in range(args.n_runs):

        if args.model == "mlp":
            model = MLP(train_data.num_features, 2)
        elif args.model == "resnet1d":
            model = ResNet1D(1, 1, train_data.num_classes)
        else:
            raise ValueError(f"Model {model} not supported")

        test_metrics = train(
            model,
            train_data,
            val_data,
            test_data,
            epochs=args.epochs,
            lr=args.lr,
            batch_size=args.batch_size
        )

        total_test_metrics.append(test_metrics)

    print("*" * 10)
    print("FINAL RESULTS")
    results_summary = compute_results_summary(total_test_metrics)
    print(results_summary)
    print("*" * 10)

    with open(out_dir / "results_summary.txt", "w") as f:
        f.write(results_summary)
