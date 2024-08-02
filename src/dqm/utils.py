import torch
import warnings
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    balanced_accuracy_score
)
from pathlib import Path


sns.set_style("darkgrid")

plt.rc("figure", figsize=(20, 10))
plt.rc("font", size=13)

warnings.filterwarnings("ignore")


def rebin(old_hist, new_bin_count) -> np.ndarray:

    old_hist = np.array(old_hist)
    old_bin_count = len(old_hist)
    old_edges = np.linspace(0, old_bin_count, old_bin_count + 1)
    old_widths = np.diff(old_edges)
    old_centers = (old_edges[:-1] + old_edges[1:]) / 2
    new_edges = np.linspace(0, old_bin_count, new_bin_count + 1)
    new_hist = np.zeros(new_bin_count)

    for i, (left, right) in enumerate(zip(new_edges[:-1], new_edges[1:])):
        mask = (old_centers >= left) & (old_centers < right)

        if np.any(mask):
            left_fractions = np.maximum(0, np.minimum(
                right, old_edges[1:]) - np.maximum(left, old_edges[:-1]))
            right_fractions = np.maximum(0, np.minimum(
                right, old_edges[1:]) - np.maximum(left, old_edges[:-1]))
            fractions = (left_fractions + right_fractions) / 2 / old_widths

            new_hist[i] = np.max(old_hist[mask] * fractions[mask])

    return new_hist


def compute_flip_idx(labels: np.ndarray):
    return np.where(labels[:-1] != labels[1:])[0] + 1


def plot_source_preds(
    histograms: torch.Tensor,
    true_source: torch.Tensor | None,
    predicted_source: torch.Tensor,
    out_path: Path
):
    histograms = histograms.cpu().numpy()
    true_source = true_source.cpu().numpy() if true_source is not None else None
    predicted_source = predicted_source.cpu().numpy()

    batch_size, num_histograms, num_bins = histograms.shape
    ncols = 3 if true_source is not None else 2

    _, axes = plt.subplots(
        batch_size, ncols, figsize=(6 * ncols, 6 * batch_size))
    if batch_size == 1:
        axes = axes.reshape(1, -1)

    colors = plt.cm.viridis(np.linspace(0, 1, num_histograms))

    for i in range(batch_size):
        highest_score_idx = predicted_source[i].argmax()

        highest_score_hist = histograms[i, highest_score_idx]
        axes[i, 0].bar(range(num_bins), highest_score_hist,
                       color='lightgray', edgecolor='black')
        axes[i, 0].set_title(f"Sample {i}: Histogram with Highest Score (Index {
                             highest_score_idx})", fontsize=14)
        axes[i, 0].set_xlabel("Bin", fontsize=12)
        axes[i, 0].set_ylabel("Count", fontsize=12)
        axes[i, 0].grid(axis='y', linestyle='--', alpha=0.7)

        for j in range(num_histograms):
            axes[i, 1].bar(j, predicted_source[i, j],
                           color=colors[j], edgecolor='black')
        axes[i, 1].set_title(
            f"Sample {i}: Predicted Anomaly Scores", fontsize=14)
        axes[i, 1].set_xlabel("Histogram Index", fontsize=12)
        axes[i, 1].set_ylabel("Score", fontsize=12)
        axes[i, 1].set_ylim(0, 1)
        axes[i, 1].set_xticks(range(num_histograms))
        axes[i, 1].grid(axis='y', linestyle='--', alpha=0.7)

        if true_source is not None:
            for j in range(num_histograms):
                axes[i, 2].bar(j, true_source[i, j],
                               color=colors[j], edgecolor='black')
            axes[i, 2].set_title(
                f"Sample {i}: True Anomaly Labels", fontsize=14)
            axes[i, 2].set_xlabel("Histogram Index", fontsize=12)
            axes[i, 2].set_ylabel("Label", fontsize=12)
            axes[i, 2].set_ylim(0, 1)
            axes[i, 2].set_xticks(range(num_histograms))
            axes[i, 2].grid(axis='y', linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def optimal_balanced_accuracy(scores, labels):

    scores = np.array(scores)
    labels = np.array(labels)

    thresholds = np.linspace(scores.min(), scores.max(), 1000)

    predictions = (scores[:, np.newaxis] >= thresholds).astype(int)

    balanced_accuracies = np.array(
        [balanced_accuracy_score(labels, pred) for pred in predictions.T])

    best_threshold_index = np.argmax(balanced_accuracies)
    best_threshold = thresholds[best_threshold_index]
    highest_balanced_accuracy = balanced_accuracies[best_threshold_index]

    return best_threshold, highest_balanced_accuracy


def compute_metrics(
        scores: np.ndarray,
        preds: np.ndarray,
        labels: np.ndarray,
):

    metrics = {
        "accuracy": accuracy_score(labels, preds),
        "balanced_accuracy": balanced_accuracy_score(labels, preds),
        "precision": precision_score(labels, preds),
        "recall": recall_score(labels, preds),
        "f1": f1_score(labels, preds),
        "auroc": roc_auc_score(labels, scores),
        "auprc": average_precision_score(labels, scores)
    }

    return metrics


def compute_results_summary(total_scores: np.ndarray, total_labels: np.ndarray):

    all_metrics_standard = []
    all_metrics_flips = []

    for run in range(len(total_scores)):

        scores = total_scores[run]
        is_anomaly = total_labels[run]
        threshold, _ = optimal_balanced_accuracy(scores, is_anomaly)
        preds = (scores >= threshold).astype(int)

        all_metrics_standard.append(compute_metrics(
            scores,
            preds,
            is_anomaly))

        flip_idx = compute_flip_idx(is_anomaly)
        flip_scores, flip_labels, flip_preds = [
            z[flip_idx] for z in [scores, is_anomaly, preds]]

        all_metrics_flips.append(compute_metrics(
            flip_scores,
            flip_preds,
            flip_labels)
        )

    metrics_standard_dict = {key: [d[key] for d in all_metrics_standard]
                             for key in all_metrics_standard[0]}
    metrics_flips_dict = {key: [d[key] for d in all_metrics_flips]
                          for key in all_metrics_flips[0]}

    results_standard = {}
    for metric, values in metrics_standard_dict.items():
        results_standard[metric] = {
            "mean": np.mean(values),
            "std": np.std(values)
        }

    results_flips = {}
    for metric, values in metrics_flips_dict.items():
        results_flips[metric] = {
            "mean": np.mean(values),
            "std": np.std(values)
        }

    res = ["Metrics Overall:"]
    for metric, stats in results_standard.items():
        res.append(f"     {metric} = {
                   stats['mean']:.4f} +- {stats['std']:.4f}")

    res.append("Metrics Flips:")
    for metric, stats in results_flips.items():
        res.append(f"     {metric} = {
                   stats['mean']:.4f} +- {stats['std']:.4f}")

    return "\n".join(res)


def compute_source_preds_results_summary(predictions, labels):

    n_runs = len(predictions)

    def safe_metric(metric_func, *args, **kwargs):
        try:
            return metric_func(*args, **kwargs)
        except ValueError:
            return np.nan

    metrics = {
        "accuracy": [],
        "f1": [],
        "roc_auc": [],
        "average_precision": []
    }

    max_metrics = {
        "accuracy": [],
        "f1": [],
        "roc_auc": [],
        "average_precision": []
    }

    for i in range(n_runs):
        preds = np.array(predictions[i])
        lbls = np.array(labels[i])

        # Calculate overall metrics
        metrics["accuracy"].append(accuracy_score(lbls, preds > 0.5))
        metrics["f1"].append(safe_metric(
            f1_score, lbls, preds > 0.5, average="macro"))
        metrics["roc_auc"].append(safe_metric(
            roc_auc_score, lbls, preds, average="macro", multi_class="ovr"))
        metrics["average_precision"].append(safe_metric(
            average_precision_score, lbls, preds, average="macro"))

        # Calculate maximum metrics
        max_pred_idx = np.argmax(preds, axis=1)
        max_lbls = lbls[np.arange(lbls.shape[0]), max_pred_idx]
        max_preds = preds[np.arange(preds.shape[0]), max_pred_idx]

        max_metrics["accuracy"].append(
            accuracy_score(max_lbls, max_preds > 0.5))
        max_metrics["f1"].append(safe_metric(
            f1_score, max_lbls, max_preds > 0.5, average="binary"))
        max_metrics["roc_auc"].append(
            safe_metric(roc_auc_score, max_lbls, max_preds))
        max_metrics["average_precision"].append(
            safe_metric(average_precision_score, max_lbls, max_preds))

    def mean_and_std(values):
        mean = np.nanmean(values)
        std = np.nanstd(values)
        return mean, std

    overall_metrics = {metric: mean_and_std(
        values) for metric, values in metrics.items()}
    overall_max_metrics = {metric: mean_and_std(
        values) for metric, values in max_metrics.items()}

    overall_str = "Source Pred Metrics Overall:\n" + "\n".join([
        f"     {metric} = {mean:.4f} +- {std:.4f}"
        for metric, (mean, std) in overall_metrics.items()
    ])

    max_pred_str = "Source Pred Metrics for Maximum:\n" + "\n".join([
        f"     {metric} = {mean:.4f} +- {std:.4f}"
        for metric, (mean, std) in overall_max_metrics.items()
    ])

    return overall_str + "\n" + max_pred_str


def plot_metrics_per_step(
        total_scores: np.ndarray,
        total_labels: np.ndarray,
        out_dir: Path):

    metrics_per_step = {
        "accuracy": [[] for _ in range(len(total_scores))],
        "balanced_accuracy": [[] for _ in range(len(total_scores))],
        "balanced_accuracy_p50": [[] for _ in range(len(total_scores))],
        "precision": [[] for _ in range(len(total_scores))],
        "recall": [[] for _ in range(len(total_scores))],
        "f1": [[] for _ in range(len(total_scores))],
        "auroc": [[] for _ in range(len(total_scores))],
        "auprc": [[] for _ in range(len(total_scores))]
    }

    total_scores = np.array(total_scores)
    total_labels = np.array(total_labels)

    thresholds = [optimal_balanced_accuracy(
        s, l)[0] for s, l in zip(total_scores, total_labels)]
    total_preds = np.array([(s >= t).astype(int)
                            for s, t in zip(total_scores, thresholds)])

    for run in range(len(total_scores)):
        for step in range(len(total_scores[run])):
            scores = total_scores[run, :step]
            preds = total_preds[run, :step]
            is_anomaly = total_labels[run, :step]
            first_label = is_anomaly[0] if step > 0 else is_anomaly

            if not np.all(is_anomaly == first_label):

                accuracy = accuracy_score(is_anomaly, preds)
                balanced_accuracy = balanced_accuracy_score(is_anomaly, preds)
                balanced_accuracy_p10 = balanced_accuracy_score(
                    is_anomaly[-50:], preds[-50:])
                precision = precision_score(is_anomaly, preds)
                recall = recall_score(is_anomaly, preds)
                f1 = f1_score(is_anomaly, preds)
                auroc = roc_auc_score(is_anomaly, scores)
                auprc = average_precision_score(is_anomaly, scores)

                metrics_per_step["accuracy"][run].append(accuracy)
                metrics_per_step["balanced_accuracy"][run].append(
                    balanced_accuracy)
                metrics_per_step["balanced_accuracy_p50"][run].append(
                    balanced_accuracy_p10)
                metrics_per_step["precision"][run].append(precision)
                metrics_per_step["recall"][run].append(recall)
                metrics_per_step["f1"][run].append(f1)
                metrics_per_step["auroc"][run].append(auroc)
                metrics_per_step["auprc"][run].append(auprc)

    for metric_name, metric_values in metrics_per_step.items():
        metric_values = np.array(metric_values)
        mu = metric_values.mean(axis=0)
        std = metric_values.std(axis=0)

        plt.title(f"{metric_name} per step")
        plt.plot(mu)
        plt.fill_between(
            np.arange(len(mu)),
            mu - std,
            mu + std,
            alpha=0.3
        )
        plt.tight_layout()
        plt.savefig(out_dir / f"{metric_name}.png")
        plt.close()
