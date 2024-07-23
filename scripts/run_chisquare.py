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
from dqm.shallow_models import ChiSquareModel
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
    parser.add_argument("--n_runs", type=int, default=5)
    parser.add_argument("--dataset", type=str, default="lhcb")
    parser.add_argument("--year", type=int, default=2018)
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
            size=500,
            num_variables=100,
            num_bins=100,
            whiten=False,
            whiten_running=False
        )
        histograms, is_anomaly = data.data, data.labels
        histograms = histograms.reshape(len(histograms), -1)
        histo_nbins_dict = {
            f"var{i}": data.num_bins for i in range(data.num_features)}

        print(data)

    out_dir = Path(f"chi_square_results_{args.dataset}{'_'+str(
                   args.year) if args.dataset == 'lhcb' else ''}")
    out_dir.mkdir(exist_ok=True)
    total_scores, total_preds, total_labels = [], [], []
    for run in range(args.n_runs):

        print(f"RUN {run + 1}/{args.n_runs}")
        if args.dataset == "synthetic":
            alpha = 0.232
        elif args.year == 2018:
            alpha = 0.6769
        else:
            alpha = 0.640

        model = ChiSquareModel(
            histograms=histograms,
            is_anomaly=is_anomaly,
            histo_nbins_dict=histo_nbins_dict,
            alpha=alpha,
            optimise_hyperparameters=False
        )

        scores, preds, labels = model.fit()

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

    total_scores = np.array(total_scores)
    total_preds = np.array(total_preds)
    total_labels = np.array(total_labels)

    print("*" * 10)
    print("FINAL RESULTS")
    results_summary = compute_results_summary(
        total_scores, total_preds, total_labels)
    print(results_summary)
    print("*" * 10)

    with open(out_dir / "results.txt", "w") as f:
        f.write(results_summary)

    plot_metrics_per_step(total_scores, total_preds, total_labels, out_dir)
