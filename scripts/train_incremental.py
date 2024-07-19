import argparse
import json
from pprint import pprint
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from matplotlib import pyplot as plt
from pathlib import Path
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    accuracy_score,
    f1_score,
    precision_recall_curve,
    roc_curve,
    RocCurveDisplay
)
from dqm.deep_models import (
    MLP,
    ResNet1D,
    Transformer,
    CNN1D,
    CNN2D,
    ContextMLP
)
from dqm.shallow_models import LinearRegressor, CopyModel
from dqm.torch_datasets import LHCbDataset, SyntheticDataset
from dqm.replay_buffer import ReplayBuffer
from dqm.settings import DATA_DIR, DEVICE
from dqm.utils import (
    compute_results_summary,
    plot_metrics_per_step,
    plot_attn_weights,
    plot_scores,
    filter_flips
)


np.random.seed(42)
torch.manual_seed(42)


def train(
    model: nn.Module,
    data: LHCbDataset,
    args: argparse.Namespace,
    plot: bool = False,
    thresh_every: int = 5
):

    requires_grad = not isinstance(model, CopyModel)
    include_var_preds = isinstance(data, SyntheticDataset)
    model = model.to(DEVICE)

    if requires_grad:

        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

        replay_buffer = ReplayBuffer(
            data,
            int(args.batch_size * args.replay_ratio),
            pos_ratio=args.replay_pos_ratio
        )

    loss_fn = nn.BCEWithLogitsLoss()
    loader = DataLoader(data, batch_size=args.batch_size, shuffle=False)

    total_probs, total_labels, total_preds = [], [], []
    if include_var_preds:
        total_var_probs, total_var_labels, total_var_preds = [], [], []

    loss_total = 0
    optimal_thresh = 0.5

    c = 0
    t = 0
    for batch_num, sample in enumerate(tqdm(loader)):

        model.eval()

        histogram = sample["histogram"].to(DEVICE)
        is_anomaly = sample["is_anomaly"].to(DEVICE)

        if include_var_preds:
            var_labels = sample["anomaly_idx"].to(DEVICE)

        # Run the model on the current batch
        out = model(histogram)

        logits = out["logits"][:len(is_anomaly)]

        loss = loss_fn(logits, is_anomaly)
        loss_total += loss.item()

        if batch_num > 0:
            # Compute outputs for evaluation
            probs = F.sigmoid(logits)
            preds = (probs > optimal_thresh).float()

            total_labels += is_anomaly.detach().flatten().cpu().tolist()
            total_probs += probs.detach().flatten().cpu().tolist()
            total_preds += preds.detach().flatten().cpu().tolist()

            if include_var_preds:

                if preds.count_nonzero():

                    var_probs = out["prob"]
                    var_preds = (var_probs > optimal_thresh).float()

                    total_var_labels += var_labels.detach().cpu().tolist()
                    total_var_probs += var_probs.detach().squeeze(-1).cpu().tolist()
                    total_var_preds += var_preds.detach().squeeze(-1).cpu().tolist()

            # if batch_num % thresh_every == 0:
            #     if 0 in total_labels and 1 in total_labels:
            #         fpr, tpr, thresholds = roc_curve(total_labels, total_probs)
            #         optimal_thresh = thresholds[np.argmax(tpr - fpr)]

        if requires_grad:

            # Train model on current batch
            model.train()
            for _ in range(args.steps_per_batch):

                # Resample from replay buffer each step (simulate epochs)
                histogram_resampled, is_anomaly_resampled = replay_buffer(
                    histogram, is_anomaly)

                logits = model(histogram_resampled)["logits"]
                loss = loss_fn(logits, is_anomaly_resampled)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            replay_buffer.update(args.batch_size)

        else:
            model.update(is_anomaly)

        if plot and batch_num > 0:

            if "attn_weights" in out.keys():
                attn_weight_dir = out_dir / "attention_weights"
                attn_weight_dir.mkdir(exist_ok=True)
                attn_weights = out["attn_weights"].detach(
                ).cpu().numpy().mean(1)

                if preds.count_nonzero():
                    plot_attn_weights(attn_weights, histogram, is_anomaly,
                                      preds, attn_weight_dir / f"{batch_num}.png")

            if "prob" in out.keys():
                scores_dir = out_dir / "probs"
                scores_dir.mkdir(exist_ok=True)

                categories = data.get_histogram_names()

                if preds.count_nonzero():
                    plot_scores(out["prob"], categories, histogram, is_anomaly,
                                preds, scores_dir / f"{batch_num}.png",
                                true_scores=var_labels if include_var_preds else None)

    fpr, tpr, _ = roc_curve(total_labels, total_probs)
    RocCurveDisplay(fpr=fpr, tpr=tpr).plot()
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.savefig(out_dir / "roc_curve.png")
    plt.close()

    precision, recall, _ = precision_recall_curve(
        total_labels, total_probs)
    plt.plot(recall, precision)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.savefig(out_dir / "precision_recall_curve.png")
    plt.close()

    torch.save(model, out_dir / "model")

    np.save(out_dir / "probs.npy", total_probs)
    np.save(out_dir / "preds.npy", total_preds)
    np.save(out_dir / "labels.npy", total_labels)

    res = {
        "probs": total_probs,
        "preds": total_preds,
        "labels": total_labels
    }

    if include_var_preds:
        np.save(out_dir / "var_probs.npy", total_var_probs)
        np.save(out_dir / "var_preds.npy", total_var_preds)
        np.save(out_dir / "var_labels.npy", total_var_labels)

        res.update({
            "var_probs": total_var_probs,
            "var_preds": total_var_preds,
            "var_labels": total_var_labels
        })

    res = {k: np.array(v) for k, v in res.items()}

    return res


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--year", type=int, default=2018)
    parser.add_argument("--steps_per_batch", type=int, default=2)
    parser.add_argument("--num_bins", type=int, default=100)
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--n_runs", type=int, default=5)
    parser.add_argument("--replay_pos_ratio", type=float, default=1.0)
    # replay ratio as a fraction of the batch size
    # (0.5 means there are 0.5 * batch_size replayed samples,
    # while there are batch_size new samples)
    parser.add_argument("--replay_ratio", type=float, default=1.0)
    parser.add_argument("--model", type=str, default="cmlp")
    parser.add_argument("--dataset", type=str, default="lhcb")
    parser.add_argument("--plot", action=argparse.BooleanOptionalAction)
    parser.add_argument(
        "--sigmoid_attn", action=argparse.BooleanOptionalAction)
    parser.add_argument("--pretrained_path", type=str, default=None)
    args = parser.parse_args()

    models = ["mlp", "tran", "cmlp", "resnet1d", "linear",
              "cnn1d", "cnn2d", "ref_filter", "copy"]
    if args.model not in models:
        raise ValueError(
            f"Model {args.model} not supported. Choose from {models}")

    years = [2018, 2023]
    if args.year not in years:
        raise ValueError(
            f"Year {args.year} not supported. Choose from {years}")

    datasets = ["lhcb", "synthetic"]
    if args.dataset not in datasets:
        raise ValueError(
            f"Dataset {args.dataset} not supported. Choose from {datasets}")

    undo_concat = args.model in ["resnet1d", "cnn1d", "cnn2d",
                                 "tran", "cmlp"]

    if args.sigmoid_attn:
        assert args.model == "tran", "Sigmoid attention only supported for Transformer model"

    out_dir = Path(
        f"./t{args.model}{'sig' if args.sigmoid_attn else ''}_inc_results_{args.year if args.dataset == "lhcb" else "synthetic"}")
    out_dir.mkdir(exist_ok=True)

    with open(out_dir / "config.json", "w") as f:
        json.dump(vars(args), f)

    include_var_preds = args.dataset == "synthetic"
    if args.dataset == "lhcb":
        data_file = f"formatted_dataset_{args.year}.csv"
        data = LHCbDataset(
            DATA_DIR / data_file,
            # Should center and norm both row and column-wise
            year=args.year,
            num_bins=args.num_bins,
            whiten=True,
            whiten_running=True,
            to_torch=True,
            undo_concat=undo_concat
        )
    else:
        data = SyntheticDataset(
            size=1000,
            num_variables=100,
            num_bins=100,
            whiten=False,
            whiten_running=True
        )

    print("#"*10)
    print("DATASET STATISTICS:")
    print(f"Number of features: {data.num_features}")
    print(f"Number of classes: {data.num_classes}")
    print(f"Number of samples (train): {len(data)}")
    print(f"Number of positive samples (train): {data.num_pos}")
    print(f"Number of negative samples (train): {data.num_neg}")
    print("#"*10)

    total_probs, total_preds, total_labels = [], [], []
    if include_var_preds:
        total_var_probs, total_var_preds, total_var_labels = [], [], []
    else:
        total_var_probs = total_var_preds = total_var_labels = None
    for run in range(args.n_runs):

        print(f"RUN {run + 1}/{args.n_runs}")

        if args.model == "mlp":
            model = MLP(data.num_features)
        elif args.model == "tran":
            model = Transformer(
                data.num_bins, data.num_features, 128, sigmoid_attn=args.sigmoid_attn)
        elif args.model == "cmlp":
            model = ContextMLP(data.num_bins, data.num_features, 128)
        elif args.model == "resnet1d":
            model = ResNet1D(data.num_features, 32)
        elif args.model == "linear":
            model = LinearRegressor(data.num_features)
        elif args.model == "cnn1d":
            model = CNN1D(data.num_features, 32)
        elif args.model == "cnn2d":
            model = CNN2D(1, 32)
        elif args.model == "copy":
            model = CopyModel()
        else:
            raise ValueError(f"Model {args.model} not supported")

        if args.pretrained_path and isinstance(model, ContextMLP):
            pretrained = torch.load(args.pretrained_path)
            model.network = pretrained.network
            model.head = pretrained.head

        print(f"MODEL SIZE: {sum(p.numel() for p in model.parameters())}")

        res = train(model, data, args, plot=args.plot)
        probs, preds, labels = res["probs"], res["preds"], res["labels"]
        _, flip_preds, flip_labels = filter_flips(probs, preds, labels)

        print(f"BALANCED ACCURACY: {balanced_accuracy_score(labels, preds)}")
        print(f"ACCURACY FLIPS: {accuracy_score(flip_labels, flip_preds)}")
        print(f"AP: {average_precision_score(labels, probs)}")
        print(f"F1: {f1_score(labels, preds)}")

        total_probs.append(probs)
        total_preds.append(preds)
        total_labels.append(labels)

        if include_var_preds:
            var_probs, var_preds, var_labels = res["var_probs"], res["var_preds"], res["var_labels"]
            total_var_probs.append(var_probs)
            total_var_preds.append(var_preds)
            total_var_labels.append(var_labels)

    print("*" * 10)
    print("FINAL RESULTS")
    results_summary = compute_results_summary(
        total_probs, total_preds, total_labels,
        total_var_probs, total_var_preds, total_var_labels
    )
    print(results_summary)
    print("*" * 10)

    with open(out_dir / "results.txt", "w") as f:
        f.write(results_summary)

    plot_metrics_per_step(total_probs, total_preds, total_labels, out_dir)
