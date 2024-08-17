import argparse
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from pathlib import Path
from torch.utils.data import DataLoader
from tqdm import tqdm
from pprint import pprint
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_curve,
    roc_auc_score,
    RocCurveDisplay
)
from dqm.models.classification import TemporalContinualTransformer, ResNet50, ResNet, HistTran
from dqm.torch_datasets import LHCbDataset, SyntheticDataset, MVTECDataset
from dqm.replay_buffer import ReplayBuffer
from dqm.settings import DATA_DIR, DEVICE
from dqm.utils import (
    compute_results_summary,
    plot_source_preds,
    compute_source_preds_results_summary,
    plot_metrics_per_step
)
from dqm.settings import HISTO_NBINS_DICT_2018, HISTO_NBINS_DICT_2023


np.random.seed(42)
torch.manual_seed(42)


def train(
    classifier: nn.Module,
    data: LHCbDataset | SyntheticDataset,
    args: argparse.Namespace
):

    classifier = classifier.to(DEVICE)
    replay_buffer = ReplayBuffer(data, args.k_past)
    cls_optimizer = torch.optim.Adam(classifier.parameters(), lr=args.lr)

    loss_fn = nn.BCEWithLogitsLoss()
    loader = DataLoader(data, batch_size=args.batch_size, shuffle=False)

    total_scores, total_labels = [], []
    total_source_preds, total_source_labels = [], []
    loss_per_step = []
    num_steps = args.steps_per_batch
    warmup = True

    for batch_num, sample in enumerate(tqdm(loader)):

        with torch.no_grad():

            inp = sample["inp"].to(DEVICE)
            is_anomaly = sample["is_anomaly"].to(DEVICE)

            pastk = torch.stack([
                replay_buffer.get_neg_pastk_samples(i) for i in range(batch_num, batch_num+args.batch_size)
            ]).to(DEVICE)

            out = classifier(inp, pastk[:len(inp)])
            logits = out["logits"]

            loss = loss_fn(logits, is_anomaly)
            loss_per_step.append(loss.item())

        if batch_num == int(args.warmup_frac * len(data)):
            print(f"WARMUP FINISHED. LOSS: {np.mean(loss_per_step)}")
            warmup = False

        if batch_num > 0:

            scores = F.sigmoid(logits)

            if not (is_anomaly == (scores > 0.5)).all():
                num_steps = 2 * args.steps_per_batch
            else:
                num_steps = args.steps_per_batch

            total_labels += is_anomaly.detach().flatten().cpu().tolist()
            total_scores += scores.detach().flatten().cpu().tolist()

            if "source_labels" in sample.keys() and (scores > 0.5).count_nonzero() and not warmup:
                source_preds = out["source_preds"].detach().cpu()
                source_labels = sample["source_labels"]
                pos_idx = torch.where(scores > 0.5)[0].detach().cpu()

                total_source_preds += source_preds[pos_idx].tolist()
                total_source_labels += source_labels[pos_idx].tolist()

            # if batch_num % 50 == 0:
            #     if 0 in total_labels and 1 in total_labels:
            #         print(f"auprc = {average_precision_score(
            #             total_labels, total_scores)}")
            #         print(f"roc_auc = {roc_auc_score(
            #             total_labels, total_scores)}")

            if args.plot:

                hist_cpu = inp.detach().cpu().numpy()
                preds_dir = out_dir / "preds"
                preds_dir.mkdir(exist_ok=True)
                full_size = len(hist_cpu[0].flatten())
                fig, ax = plt.subplots(nrows=4)
                fig.suptitle(f"Is anomaly: {is_anomaly[0].item()}\nScore: {
                             scores[0].item()}")
                ax[0].plot(hist_cpu[0].flatten()[:full_size//4])
                ax[1].plot(hist_cpu[0].flatten()[full_size//4:2*full_size//4])
                ax[2].plot(hist_cpu[0].flatten()[
                           2*full_size//4:3*full_size//4])
                ax[3].plot(hist_cpu[0].flatten()[3*full_size//4:])

                plt.savefig(preds_dir / f"pred_{batch_num}.png")
                plt.close()

                if "source_preds" in out.keys() and is_anomaly.count_nonzero() > 0:
                    source_preds_dir = out_dir / "source_preds"
                    source_preds_dir.mkdir(exist_ok=True)
                    source_preds = out["source_preds"]
                    anomaly_idx = sample["source_labels"] if "source_labels" in sample.keys(
                    ) else None

                    # categories = data.get_histogram_names()
                    plot_source_preds(inp, anomaly_idx,
                                      source_preds, source_preds_dir / f"{batch_num}.png")

        # Train model on current batch
        model.train()
        for _ in range(num_steps):

            # Resample from replay buffer each step (simulate epochs)
            histogram_resampled, is_anomaly_resampled, pastk_resampled = replay_buffer(
                inp, is_anomaly, pastk)

            logits = classifier(histogram_resampled,
                                pastk_resampled)["logits"]

            loss = loss_fn(logits, is_anomaly_resampled)
            cls_optimizer.zero_grad()
            loss.backward()
            cls_optimizer.step()

        replay_buffer.update(args.batch_size)

    fpr, tpr, _ = roc_curve(total_labels, total_scores)
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
    plt.ylabel("Loss")
    plt.savefig(out_dir / "loss.png")
    plt.close()

    colors = ["green" if l == 0 else "red" for l in total_labels]

    start_idx = int(args.warmup_frac * len(data))

    plt.scatter(range(len(total_scores)), total_scores, c=colors, alpha=0.6)
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

    legend_elements += [tline, wline]

    plt.title("Probability Scatter Plot with Optimal Threshold")
    plt.xlabel("Sample Index")
    plt.ylabel("Probability")
    plt.ylim(-0.05, 1.05)

    plt.legend(handles=legend_elements, loc='best')

    plt.tight_layout()
    plt.savefig(out_dir / "prob_scatter.png")
    plt.close()

    np.save(out_dir / "probs.npy", total_scores)
    np.save(out_dir / "labels.npy", total_labels)

    torch.save(classifier.state_dict(), out_dir / "model")

    return {
        "model": classifier,
        "probs": total_scores,
        "labels": total_labels,
        "source_preds": total_source_preds,
        "source_labels": total_source_labels
    }


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--steps_per_batch", type=int, default=2)
    parser.add_argument("--num_bins", type=int, default=100)
    parser.add_argument("--k_past", type=int, default=5)
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--n_runs", type=int, default=5)
    parser.add_argument("--dataset", type=str, default="lhcb")
    parser.add_argument("--year", type=int, default=2018)
    parser.add_argument("--warmup_frac", type=float, default=0.1)
    parser.add_argument("--warmup_synthetic",
                        action=argparse.BooleanOptionalAction)
    parser.add_argument("--plot", action=argparse.BooleanOptionalAction)
    args = parser.parse_args()

    years = [2018, 2023]
    if args.year not in years:
        raise ValueError(
            f"Year {args.year} not supported. Choose from {years}")

    out_dir = Path(
        f"./tct_inc_results_{args.dataset}{f'_{args.year}' if args.dataset == "lhcb" else ""}")
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
    elif args.dataset == "synthetic":
        data = SyntheticDataset(
            size=2000,
            num_variables=100,
            num_bins=args.num_bins,
            whiten=True
        )

    elif args.dataset == "mvtec":
        data = MVTECDataset(
            resolution=256,
            to_tensor=True,
            normalize=True
        )

    print("="*50)
    print("CONFIG:")
    pprint(args.__dict__)

    print("="*50)
    print(data)

    total_probs, total_labels = [], []
    total_source_preds, total_source_labels = [], []
    for run in range(args.n_runs):

        print(f"RUN {run + 1}/{args.n_runs}")

        if args.dataset == "mvtec":
            in_dim = 2048
            n_vars = 1
            backbone = ResNet50()
            # backbone = ResNet(256, 100, 3)
            # in_dim = 100
        else:
            n_vars = data.num_features
            backbone = HistTran(n_vars, data.num_bins, 100)

        model = TemporalContinualTransformer(
            backbone,
            n_vars,
            100,
            k_past=args.k_past
        )

        print(f"MODEL SIZE: {sum(p.numel() for p in model.parameters())}")

        print("TRAINING...")
        out = train(model, data, args)

        start_idx = int(args.warmup_frac * len(data))
        probs, labels = out["probs"], out["labels"]
        probs, labels = probs[start_idx:], labels[start_idx:]

        if isinstance(data, SyntheticDataset):
            source_preds, source_labels = out["source_preds"], out["source_labels"]
            total_source_preds.append(source_preds)
            total_source_labels.append(source_labels)

        print(f"AP: {average_precision_score(labels, probs)}")
        print(f"AUROC: {roc_auc_score(labels, probs)}")

        total_probs.append(probs)
        total_labels.append(labels)

    total_probs = np.array(total_probs)
    total_labels = np.array(total_labels)

    print("="*50)
    results_summary = compute_results_summary(
        total_probs, total_labels)
    print(results_summary)
    print("="*50)

    with open(out_dir / "results.txt", "w") as f:
        f.write(results_summary)

    if isinstance(data, SyntheticDataset):

        source_preds_results_summary = compute_source_preds_results_summary(
            total_source_preds, total_source_labels)
        print(source_preds_results_summary)
        print("="*50)
        with open(out_dir / "source_preds_results.txt", "w") as f:
            f.write(source_preds_results_summary)

    plot_metrics_per_step(total_probs, total_labels, out_dir)
