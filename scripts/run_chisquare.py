import argparse
from pathlib import Path
import numpy as np
from sklearn.metrics import (
    f1_score,
    average_precision_score,
    balanced_accuracy_score
)
from dqm.shallow_models import ChiSquareModel
from dqm.settings import DATA_DIR
from dqm.utils import compute_results_summary, plot_metrics_per_step


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--n_runs", type=int, default=5)
    parser.add_argument("--year", type=int, default=2018)
    args = parser.parse_args()

    if args.year not in [2018, 2023]:
        raise ValueError("Year must be 2018 or 2023")

    out_dir = Path(f"chi_square_results_{args.year}")
    out_dir.mkdir(exist_ok=True)
    total_scores, total_preds, total_labels = [], [], []
    for run in range(args.n_runs):

        print(f"RUN {run + 1}/{args.n_runs}")

        alpha = 0.1075 if args.year == 2023 else 0.6769

        model = ChiSquareModel(
            DATA_DIR / f"formatted_dataset_{args.year}.csv",
            year=args.year,
            alpha=alpha,
            optimise_hyperparameters=False
        )

        scores, preds, labels = model.fit()

        print(f"BALANCED ACCURACY: {balanced_accuracy_score(labels, preds)}")
        print(f"AP: {average_precision_score(labels, scores)}")
        print(f"F1: {f1_score(labels, preds)}")

        total_scores.append(scores)
        total_preds.append(preds)
        total_labels.append(labels)

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
