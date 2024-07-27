import argparse
import json
from matplotlib.lines import Line2D
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
    precision_recall_curve,
    roc_curve,
    roc_auc_score,
    RocCurveDisplay
)
from dqm.deep_models import (
    AlphaNet,
    ContextMLP,
    Transformer
)
from dqm.reference_builder import RefBuilder
from dqm.torch_datasets import LHCbDataset, SyntheticDataset, SequentialDataset
from dqm.replay_buffer import ReplayBuffer
from dqm.settings import DATA_DIR, DEVICE
from dqm.utils import (
    compute_results_summary,
    plot_scores,
)
from dqm.settings import HISTO_NBINS_DICT_2018, HISTO_NBINS_DICT_2023


np.random.seed(42)
torch.manual_seed(42)


def warmup_synthetic(
        classifier: nn.Module,
        data: SequentialDataset,
        args: argparse.Namespace,
        alpha: float = 0.01
):

    data = SyntheticDataset(
        size=2000,
        num_variables=data.num_features,
        num_bins=data.num_bins,
        whiten=False,
        whiten_running=False,
        nominal_fraction=0.8
    )

    print("WARMUP DATASET:")
    print(data)

    classifier = classifier.to(DEVICE)
    replay_buffer = ReplayBuffer(data)

    ref_builder = RefBuilder(data.num_features, data.num_bins, device=DEVICE)
    optimizer = torch.optim.Adam(classifier.parameters(), lr=args.lr)
    loss_fn = nn.BCEWithLogitsLoss()
    loader = DataLoader(data, batch_size=args.batch_size, shuffle=False)

    for sample in tqdm(loader):

        histogram = sample["histogram"].to(DEVICE)
        is_anomaly = sample["is_anomaly"].to(DEVICE)

        out = classifier(histogram, ref_builder.mu)
        logits = out["logits"]

        loss = loss_fn(logits, is_anomaly)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        replay_buffer.update(args.batch_size)

        ref_builder.update_reference(
            histogram[(is_anomaly == 0).squeeze(-1)],
            alpha
        )

    return model


def train(
    classifier: nn.Module,
    data: SequentialDataset,
    args: argparse.Namespace,
    alpha: float = 0.01
):

    classifier = classifier.to(DEVICE)
    replay_buffer = ReplayBuffer(data)

    ref_builder = RefBuilder(data.num_features, data.num_bins, device=DEVICE)
    cls_optimizer = torch.optim.Adam(classifier.parameters(), lr=args.lr)

    loss_fn = nn.BCEWithLogitsLoss()
    loader = DataLoader(data, batch_size=args.batch_size, shuffle=False)

    total_probs, total_labels = [], []
    loss_per_step = []
    num_steps = args.steps_per_batch

    for batch_num, sample in enumerate(tqdm(loader)):

        histogram = sample["histogram"].to(DEVICE)
        is_anomaly = sample["is_anomaly"].to(DEVICE)

        out = classifier(histogram, ref_builder.mu)
        logits = out["logits"]

        loss = loss_fn(logits, is_anomaly)
        loss_per_step.append(loss.item())

        if batch_num == int(args.warmup_frac * len(data)):
            print(f"WARMUP FINISHED. LOSS: {np.mean(loss_per_step)}")

        if batch_num > 0:

            probs = F.sigmoid(logits)

            # if is_anomaly.item() != preds.item():
            #     num_steps = 2 * args.steps_per_batch
            #     # cls_optimizer.param_groups[0]["lr"] = 10 * args.lr
            # else:
            #     num_steps = args.steps_per_batch
            # cls_optimizer.param_groups[0]["lr"] = args.lr

            total_labels += is_anomaly.detach().flatten().cpu().tolist()
            total_probs += probs.detach().flatten().cpu().tolist()

            if args.plot:

                # print(sample["anomaly_idx"])
                aidx = sample["anomaly_idx"].argmax(dim=-1).item()
                _, ax = plt.subplots(nrows=3, figsize=(10, 12))
                pdir = out_dir / "preds"
                pdir.mkdir(exist_ok=True)
                ax[0].plot(ref_builder.mu[aidx].detach().cpu().numpy())
                ax[1].plot(histogram[0, aidx].detach().cpu().numpy())
                diff_arr = (histogram[0, aidx] - ref_builder.mu[aidx]).abs()
                diff = diff_arr.mean().item()
                ax[1].set_title(f"Pred={probs.item()} Label={
                                is_anomaly.item()} Diff={diff:.2f}")
                ax[2].plot(diff_arr.detach().cpu().numpy())
                plt.savefig(pdir / f"{batch_num}.png")
                plt.close()

                if "prob" in out.keys():
                    scores_dir = out_dir / "probs"
                    scores_dir.mkdir(exist_ok=True)
                    scores = out["prob"]

                    categories = data.get_histogram_names()
                    plot_scores(scores, categories, histogram, is_anomaly,
                                scores_dir / f"{batch_num}.png", reference=ref_builder.mu)

        # Train model on current batch
        for _ in range(num_steps):

            # Resample from replay buffer each step (simulate epochs)
            histogram_resampled, is_anomaly_resampled, ref_resampled = replay_buffer(
                histogram, is_anomaly, ref_builder.mu.unsqueeze(0))

            logits = classifier(histogram_resampled,
                                ref_resampled)["logits"]

            loss = loss_fn(logits, is_anomaly_resampled)
            cls_optimizer.zero_grad()
            loss.backward()
            cls_optimizer.step()

        replay_buffer.update(args.batch_size)

        for _ in range(args.batch_size):
            replay_buffer.add_ref(ref_builder.mu)

        ref_builder.update_reference(
            histogram[(is_anomaly == 0).squeeze(-1)],
            alpha
        )

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

    plt.plot(loss_per_step)
    plt.xlabel("Batch Number")
    plt.ylabel("Loss")
    plt.savefig(out_dir / "loss.png")
    plt.close()

    colors = ["green" if l == 0 else "red" for l in total_labels]

    start_idx = int(args.warmup_frac * len(data))

    plt.scatter(range(len(total_probs)), total_probs, c=colors, alpha=0.6)
    tline = plt.axhline(y=0.5, color="blue", linestyle="--",
                        label="Threshold: 0.5")
    wline = plt.axvline(x=start_idx, color="orange",
                        linestyle="--", label="Warmup End")

    legend_elements = [
        Line2D([0], [0], marker="o", color="w", label="Nominal",
               markerfacecolor="green", markersize=10),
        Line2D([0], [0], marker="o", color="w", label="Anomaly",
               markerfacecolor="red", markersize=10)
    ]

    # Add the line objects directly, no need for additional Line2D objects
    legend_elements += [tline, wline]

    plt.title("Probability Scatter Plot with Optimal Threshold")
    plt.xlabel("Sample Index")
    plt.ylabel("Probability")
    plt.ylim(-0.05, 1.05)

    # Only call legend once, with all the elements
    plt.legend(handles=legend_elements, loc='best')

    plt.tight_layout()
    plt.savefig(out_dir / "prob_scatter.png")
    plt.close()

    np.save(out_dir / "probs.npy", total_probs)
    np.save(out_dir / "labels.npy", total_labels)

    torch.save(classifier.state_dict(), out_dir / "model")

    return {
        "model": classifier,
        "probs": total_probs,
        "labels": total_labels
    }


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--year", type=int, default=2023)
    parser.add_argument("--steps_per_batch", type=int, default=2)
    parser.add_argument("--num_bins", type=int, default=100)
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--n_runs", type=int, default=5)
    parser.add_argument("--model", type=str, default="tran")
    parser.add_argument("--dataset", type=str, default="lhcb")
    parser.add_argument("--warmup_frac", type=float, default=0.2)
    parser.add_argument("--warmup_synthetic",
                        action=argparse.BooleanOptionalAction)
    # replay ratio as a fraction of the batch size
    # (0.5 means there are 0.5 * batch_size replayed samples,
    # while there are batch_size new samples)
    parser.add_argument("--plot", action=argparse.BooleanOptionalAction)
    args = parser.parse_args()

    years = [2018, 2023]
    if args.year not in years:
        raise ValueError(
            f"Year {args.year} not supported. Choose from {years}")

    models = ["tran", "cmlp"]
    if args.model not in models:
        raise ValueError(
            f"Model {args.model} not supported. Choose from {models}")

    out_dir = Path(
        f"./{args.model}refb_results_{args.year if args.dataset == "lhcb" else "synthetic"}")
    out_dir.mkdir(exist_ok=True)

    with open(out_dir / "config.json", "w") as f:
        json.dump(vars(args), f)

    if args.dataset == "lhcb":
        data_file = f"formatted_dataset_{args.year}.csv"
        histo_nbins_dict = HISTO_NBINS_DICT_2018 if args.year == 2018 else HISTO_NBINS_DICT_2023
        data = LHCbDataset(
            DATA_DIR / data_file,
            histo_nbins_dict=histo_nbins_dict,
            # Should center and norm both row and column-wise
            num_bins=args.num_bins,
            whiten=False,
            whiten_running=False,
            to_torch=True
        )
    else:
        data = SyntheticDataset(
            size=2000,
            num_variables=100,
            num_bins=100,
            whiten=False,
            whiten_running=False
        )

    print(data)

    total_probs, total_labels = [], []
    for run in range(args.n_runs):

        print(f"RUN {run + 1}/{args.n_runs}")

        if args.model == "cmlp":
            model = ContextMLP(
                data.num_bins, data.num_features, 128, use_ref=True)
        elif args.model == "tran":
            model = Transformer(
                data.num_bins,
                data.num_features,
                128,
                sigmoid_attn=False,
                use_ref=True)
        else:
            raise ValueError("Model not supported")

        print(f"MODEL SIZE: {sum(p.numel() for p in model.parameters())}")

        if args.warmup_synthetic:
            print("WARMING UP ON SYNTHETIC DATA...")
            model = warmup_synthetic(model, data, args)

        print("TRAINING...")
        out = train(model, data, args)

        start_idx = int(args.warmup_frac * len(data))
        probs, labels = out["probs"][start_idx:], out["labels"][start_idx:]

        print(f"AP: {average_precision_score(labels, probs)}")
        print(f"AUROC: {roc_auc_score(labels, probs)}")

        total_probs.append(probs)
        total_labels.append(labels)

    total_probs = np.array(total_probs)
    total_labels = np.array(total_labels)

    print("="*100)
    results_summary = compute_results_summary(
        total_probs, total_labels)
    print(results_summary)
    print("="*100)

    with open(out_dir / "results.txt", "w") as f:
        f.write(results_summary)

    # plot_metrics_per_step(total_probs, total_labels, out_dir)
