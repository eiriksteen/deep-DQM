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
from dqm.models.reconstruction import VAE
from dqm.torch_datasets import LHCbDataset, SyntheticDataset, DataStream
from dqm.replay_buffer import ReplayBuffer
from dqm.settings import DATA_DIR, DEVICE
from dqm.utils import (
    compute_results_summary,
    plot_metrics_per_step
)
from dqm.settings import HISTO_NBINS_DICT_2018, HISTO_NBINS_DICT_2023


np.random.seed(42)
torch.manual_seed(42)


def warmup_synthetic(
        autoencoder: VAE,
        data: LHCbDataset | SyntheticDataset,
        args: argparse.Namespace
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

    autoencoder = autoencoder.to(DEVICE)
    optimizer = torch.optim.Adam(autoencoder.parameters(), lr=args.lr)
    loader = DataLoader(data, batch_size=args.batch_size, shuffle=False)
    loss_total = 0

    for sample in tqdm(loader):

        histogram = sample["histogram"].to(DEVICE)

        logits = autoencoder(histogram)
        loss = F.mse_loss(logits, histogram)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_total += loss.item()

    return autoencoder


def train(
    autoencoder: nn.Module,
    data: DataStream,
    args: argparse.Namespace
):

    autoencoder = autoencoder.to(DEVICE)
    replay_buffer = ReplayBuffer(data, classes="neg")
    optimizer = torch.optim.Adam(autoencoder.parameters(), lr=args.lr)
    loader = DataLoader(data, batch_size=args.batch_size, shuffle=False)

    total_scores, total_labels = [], []
    loss_per_step = []

    for batch_num, sample in enumerate(tqdm(loader)):

        with torch.no_grad():

            histogram = sample["histogram"].to(DEVICE)
            is_anomaly = sample["is_anomaly"].to(DEVICE)

            logits, _, _ = autoencoder(histogram)
            loss = F.mse_loss(logits, histogram)
            loss_per_step.append(loss.item())

        if batch_num == int(args.warmup_frac * len(data)):
            print(f"WARMUP FINISHED. LOSS: {np.mean(loss_per_step)}")

        if batch_num > 0:

            errs = (logits - histogram).abs().mean(-1)
            scores = errs.max(-1).values  # errs.max(-1).values
            # aidx = torch.argmax(errs, dim=-1)

            if args.plot:

                hist_cpu = histogram.detach().cpu().numpy()
                logits_cpu = logits.detach().cpu().numpy()

                recon_dir = out_dir / "reconstructions"
                recon_dir.mkdir(exist_ok=True)
                full_size = len(hist_cpu[0].flatten())
                fig, ax = plt.subplots(nrows=4)
                fig.suptitle(f"Is anomaly: {is_anomaly[0].item()}\nScore: {
                             scores[0].item()}")
                ax[0].plot(hist_cpu[0].flatten()[:full_size//4])
                ax[0].plot(logits_cpu[0].flatten()[:full_size//4], alpha=0.3)
                ax[0].legend(["Original", "Reconstructed"])
                ax[1].plot(hist_cpu[0].flatten()[full_size//4:2*full_size//4])
                ax[1].plot(logits_cpu[0].flatten()[
                           full_size//4:2*full_size//4], alpha=0.3)
                ax[1].legend(["Original", "Reconstructed"])
                ax[2].plot(hist_cpu[0].flatten()[
                           2*full_size//4:3*full_size//4])
                ax[2].plot(logits_cpu[0].flatten()[
                           2*full_size//4:3*full_size//4], alpha=0.3)
                ax[2].legend(["Original", "Reconstructed"])
                ax[3].plot(hist_cpu[0].flatten()[3*full_size//4:])
                ax[3].plot(logits_cpu[0].flatten()[3*full_size//4:], alpha=0.3)
                ax[3].legend(["Original", "Reconstructed"])

                plt.savefig(recon_dir / f"recon_{batch_num}.png")
                plt.close()

            total_labels += is_anomaly.detach().flatten().cpu().tolist()
            total_scores += scores.detach().flatten().cpu().tolist()

        neg_idx = (is_anomaly == 0).nonzero(as_tuple=True)[0]
        is_anomaly = is_anomaly[neg_idx]
        histogram = histogram[neg_idx]

        if len(is_anomaly) > 0:

            # Train model on current batch

            model.train()

            for _ in range(args.steps_per_batch):

                # Resample from replay buffer each step(simulate epochs)
                histogram_resampled, _, _ = replay_buffer(
                    histogram, is_anomaly)

                logits, mu, logvar = autoencoder(histogram_resampled)

                mse_loss = ((logits - histogram_resampled)**2).sum(-1).mean()
                kl_loss = -0.5*(1+logvar-mu**2-torch.e**logvar).sum(-1).mean()
                loss = mse_loss + kl_loss
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        replay_buffer.update(args.batch_size)

    fpr, tpr, thresholds = roc_curve(total_labels, total_scores)
    RocCurveDisplay(fpr=fpr, tpr=tpr).plot()
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.savefig(out_dir / "roc_curve.png")
    plt.close()

    precision, recall, _ = precision_recall_curve(
        total_labels, total_scores)
    plt.plot(recall, precision)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.savefig(out_dir / "precision_recall_curve.png")
    plt.close()

    plt.plot(loss_per_step)
    plt.xlabel("Batch Number")
    plt.ylabel("MSE Loss")
    plt.savefig(out_dir / "loss.png")
    plt.close()

    optimal_threshold = thresholds[np.argmax(tpr - fpr)]

    colors = ["green" if l == 0 else "red" for l in total_labels]

    start_idx = int(args.warmup_frac * len(data))

    plt.scatter(range(len(total_scores)), total_scores, c=colors, alpha=0.6)
    tline = plt.axhline(y=optimal_threshold, color="blue", linestyle="--",
                        label=f"Threshold: {optimal_threshold:.2f}")
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

    plt.title("Score Scatter Plot with Optimal Threshold")
    plt.xlabel("Sample Index")
    plt.ylabel("Score")
    plt.ylim(min(total_scores), max(total_scores))

    # Only call legend once, with all the elements
    plt.legend(handles=legend_elements, loc='best')

    plt.tight_layout()
    plt.savefig(out_dir / "score_scatter.png")
    plt.close()

    np.save(out_dir / "scores.npy", total_scores)
    np.save(out_dir / "labels.npy", total_labels)

    torch.save(autoencoder.state_dict(), out_dir / "model")

    return {
        "model": autoencoder,
        "probs": total_scores,
        "labels": total_labels
    }


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--year", type=int, default=2018)
    parser.add_argument("--steps_per_batch", type=int, default=4)
    parser.add_argument("--num_bins", type=int, default=100)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--n_runs", type=int, default=5)
    parser.add_argument("--model", type=str, default="vae")
    parser.add_argument("--dataset", type=str, default="lhcb")
    parser.add_argument("--warmup_frac", type=float, default=0.1)
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

    models = ["vae"]
    if args.model not in models:
        raise ValueError(
            f"Model {args.model} not supported. Choose from {models}")

    out_dir = Path(
        f"./{args.model}_rec_results_{args.year if args.dataset == "lhcb" else "synthetic"}")
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
            whiten=True,
            to_torch=True
        )
    else:
        data = SyntheticDataset(
            size=2000,
            num_variables=100,
            num_bins=100,
            whiten=True,
            to_torch=True
        )

    print(data)

    total_scores, total_labels = [], []
    for run in range(args.n_runs):

        print(f"RUN {run + 1}/{args.n_runs}")

        if args.model == "vae":
            model = VAE(data.num_bins, data.num_features, 32)
        else:
            raise ValueError("Model not supported")

        print(f"MODEL SIZE: {sum(p.numel() for p in model.parameters())}")

        if args.warmup_synthetic:
            print("WARMING UP ON SYNTHETIC DATA...")
            model = warmup_synthetic(model, data, args)

        print("TRAINING...")
        out = train(model, data, args)

        start_idx = int(args.warmup_frac * len(data))
        scores, labels = out["probs"][start_idx:], out["labels"][start_idx:]

        print(f"AP: {average_precision_score(labels, scores)}")
        print(f"AUROC: {roc_auc_score(labels, scores)}")

        total_scores.append(scores)
        total_labels.append(labels)

    total_scores = np.array(total_scores)
    total_labels = np.array(total_labels)

    print("="*100)
    results_summary = compute_results_summary(total_scores, total_labels)
    print(results_summary)
    print("="*100)

    with open(out_dir / "results.txt", "w") as f:
        f.write(results_summary)

    plot_metrics_per_step(total_scores, total_labels, out_dir)
