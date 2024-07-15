import argparse
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    f1_score
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
from dqm.torch_datasets import LHCbDataset
from dqm.replay_buffer import ReplayBuffer
from dqm.settings import DATA_DIR, DEVICE
from dqm.utils import (
    compute_results_summary,
    plot_metrics_per_step,
    plot_attn_weights,
    plot_scores
)


torch.manual_seed(0)


def train(
        model: nn.Module,
        data: LHCbDataset,
        args: argparse.Namespace,
        plot: bool = False):

    requires_grad = not isinstance(model, CopyModel)
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

    total_probs, total_preds, total_labels = [], [], []
    loss_total = 0

    for batch_num, sample in enumerate(tqdm(loader)):

        model.eval()

        histogram = sample["histogram"].to(DEVICE)
        is_anomaly = sample["is_anomaly"].to(DEVICE)

        # Run the model on the current batch
        out = model(histogram)
        logits = out["logits"][:len(is_anomaly)]

        loss = loss_fn(logits, is_anomaly)
        loss_total += loss.item()

        # Don't evaluate the model before it has seen anything
        if batch_num > 0:
            probs = F.sigmoid(logits)
            preds = torch.where(logits > 0.5, 1, 0)

            total_labels += is_anomaly.detach().cpu().tolist()
            total_preds += preds.detach().cpu().tolist()
            total_probs += probs.detach().cpu().tolist()

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

        if plot and batch_num > 0 and batch_num % (5 if args.year == 2018 else 2) == 0:

            if "attn_weights" in out.keys():
                attn_weight_dir = out_dir / "attention_weights"
                attn_weight_dir.mkdir(exist_ok=True)
                attn_weights = out["attn_weights"].detach(
                ).cpu().numpy().mean(1)
                plot_attn_weights(attn_weights, histogram, is_anomaly,
                                  preds, attn_weight_dir / f"{batch_num}.png")

            elif "scores" in out.keys():
                scores_dir = out_dir / "scores"
                scores_dir.mkdir(exist_ok=True)
                scores = out["scores"].detach().cpu().numpy()

                categories = data.get_histogram_names()
                plot_scores(scores, categories, histogram, is_anomaly,
                            preds, scores_dir / f"{batch_num}.png")

    return total_probs, total_preds, total_labels


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--year", type=int, default=2018)
    parser.add_argument("--steps_per_batch", type=int, default=16)
    parser.add_argument("--num_bins", type=int, default=100)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--n_runs", type=int, default=5)
    parser.add_argument("--replay_pos_ratio", type=float, default=1.0)
    # replay ratio as a fraction of the batch size
    # (0.5 means there are 0.5 * batch_size replayed samples,
    # while there are batch_size new samples)
    parser.add_argument("--replay_ratio", type=float, default=1.0)
    parser.add_argument("--model", type=str, default="tran")
    parser.add_argument("--plot", action=argparse.BooleanOptionalAction)
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

    undo_concat = args.model in ["resnet1d", "cnn1d", "cnn2d",
                                 "tran", "cmlp"]

    out_dir = Path(f"./{args.model}_inc_results_{args.year}")
    out_dir.mkdir(exist_ok=True)

    with open(out_dir / "config.json", "w") as f:
        json.dump(vars(args), f)

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
            model = MLP(data.num_features)
        elif args.model == "tran":
            model = Transformer(
                data.num_bins, data.num_features, 128)
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

        print(f"MODEL SIZE: {sum(p.numel() for p in model.parameters())}")

        probs, preds, labels = train(model, data, args, plot=args.plot)

        print(f"BALANCED ACCURACY: {balanced_accuracy_score(labels, preds)}")
        print(f"AP: {average_precision_score(labels, probs)}")
        print(f"F1: {f1_score(labels, preds)}")

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
