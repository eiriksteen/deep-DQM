from matplotlib.lines import Line2D
import pandas as pd
import argparse
from pathlib import Path
from matplotlib import pyplot as plt
import numpy as np
from sklearn.metrics import (
    RocCurveDisplay,
    f1_score,
    average_precision_score,
    balanced_accuracy_score,
    precision_recall_curve,
    roc_curve
)
from dqm.models.classification import ChiSquareModel
from dqm.settings import DATA_DIR
from dqm.torch_datasets import SyntheticDataset
from dqm.utils import compute_results_summary, plot_metrics_per_step
from dqm.settings import HISTO_NBINS_DICT_2018, HISTO_NBINS_DICT_2023

np.random.seed(42)


def prepare_lhcb_data(data_path: Path):
    df = pd.read_csv(data_path, header=0)
    input_var_cols = [
        c for c in df.columns if 'var' in c and not 'err' in c]
    histograms = df[input_var_cols].to_numpy()
    is_anomaly = 1 - df['all_OK'].to_numpy()

    return histograms, is_anomaly


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--n_runs", type=int, default=1)
    parser.add_argument("--dataset", type=str, default="lhcb")
    parser.add_argument("--year", type=int, default=2018)
    parser.add_argument("--warmup_frac", type=float, default=0.0)
    parser.add_argument("--optimize_hyperparams",
                        action=argparse.BooleanOptionalAction)
    args = parser.parse_args()

    if args.year not in [2018, 2023]:
        raise ValueError("Year must be 2018 or 2023")

    dsets = ["lhcb", "synthetic"]
    if args.dataset not in dsets:
        raise ValueError(f"Dataset must be one of {dsets}")

    if args.dataset == "lhcb":
        data_path = DATA_DIR / f"formatted_dataset_{args.year}.csv"
        histograms, is_anomaly = prepare_lhcb_data(data_path)

        if args.year == 2018:
            histo_nbins_dict = HISTO_NBINS_DICT_2018
        else:
            histo_nbins_dict = HISTO_NBINS_DICT_2023

    else:
        data = SyntheticDataset(
            size=2000,
            num_variables=100,
            num_bins=100,
            whiten=False
        )
        histograms, is_anomaly = data.data, data.labels
        histograms = histograms.reshape(len(histograms), -1)
        histo_nbins_dict = {
            f"var{i}": data.num_bins for i in range(data.num_features)}

        print(data)

    start_idx = int(args.warmup_frac * len(histograms))
    out_dir = Path(f"chi_square_results_{args.dataset}{'_'+str(
                   args.year) if args.dataset == 'lhcb' else ''}")
    out_dir.mkdir(exist_ok=True)
    total_scores, total_preds, total_labels = [], [], []
    for run in range(args.n_runs):

        print(f"RUN {run + 1}/{args.n_runs}")
        if args.dataset == "synthetic":
            alpha = 0.00011437481734488664
        elif args.year == 2018:
            alpha = 0.5290345799521846
        else:
            alpha = 0.02885998

        model = ChiSquareModel(
            histograms=histograms,
            is_anomaly=is_anomaly,
            histo_nbins_dict=histo_nbins_dict,
            alpha=alpha,
            optimise_hyperparameters=args.optimize_hyperparams
        )

        scores, preds, labels = model.fit()
        scores, preds, labels = (z[start_idx:]
                                 for z in (scores, preds, labels))

        np.save(out_dir / "scores.npy", scores)
        np.save(out_dir / "preds.npy", preds)
        np.save(out_dir / "labels.npy", labels)

        print(f"BALANCED ACCURACY: {balanced_accuracy_score(labels, preds)}")
        print(f"AP: {average_precision_score(labels, scores)}")
        print(f"F1: {f1_score(labels, preds)}")

        total_scores.append(scores)
        total_preds.append(preds)
        total_labels.append(labels)

        fpr, tpr, thresholds = roc_curve(labels, scores)
        RocCurveDisplay(fpr=fpr, tpr=tpr).plot()
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.savefig(out_dir / "roc_curve.png")
        plt.close()

        precision, recall, thresholds = precision_recall_curve(labels, scores)
        plt.plot(recall, precision)
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.savefig(out_dir / "precision_recall_curve.png")
        plt.close()

        optimal_threshold = 0

        colors = ["green" if l == 0 else "red" for l in labels]

        plt.scatter(range(len(scores)), scores, c=colors, alpha=0.6)
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
        plt.ylim(min(scores), max(scores))

        # Only call legend once, with all the elements
        plt.legend(handles=legend_elements, loc='best')

        plt.tight_layout()
        plt.savefig(out_dir / "score_scatter.png")
        plt.close()

    total_scores = np.array(total_scores)
    total_labels = np.array(total_labels)

    print("*" * 10)
    print("FINAL RESULTS")

    results_summary = compute_results_summary(
        total_scores,
        total_labels
    )

    print(results_summary)
    print("*" * 10)

    with open(out_dir / "results.txt", "w") as f:
        f.write(results_summary)

    plot_metrics_per_step(total_scores, total_labels, out_dir)
