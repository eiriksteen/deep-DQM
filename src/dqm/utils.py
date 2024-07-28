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


def plot_attn_weights(
        attn_weights: torch.Tensor,
        histogram: torch.Tensor,
        labels: torch.Tensor,
        scores: torch.Tensor,
        save_path: Path,
        categories: list,
        true_scores: torch.Tensor | None = None
):

    num_samples = len(attn_weights)
    ncols = 2 + (1 if true_scores is not None else 0)
    _, axes = plt.subplots(
        nrows=num_samples, ncols=ncols, figsize=(10, 5 * num_samples))
    axes = np.atleast_2d(axes)
    color_palette = sns.color_palette("viridis", len(categories))
    bar_width = 0.6

    for i, ax in enumerate(axes):
        label_value = labels[i].item()
        score_value = scores[i].item()
        top_act_idx = attn_weights[i].mean(0).argmax().item()
        hist = histogram[i, top_act_idx].detach().cpu().numpy()
        true_scr = true_scores[i].detach().cpu().numpy(
        ) if true_scores is not None else None

        sns.heatmap(attn_weights[i], ax=ax[0],
                    cmap="YlGnBu", cbar=True)
        ax[0].set_title(f"Attention Weights\nLabel: {
                        label_value}, Pred: {score_value}")
        ax[0].set_xlabel("Target")
        ax[0].set_ylabel("Source")
        ax[1].plot(hist)
        ax[1].set_title(f"Histogram with Strongest Activation\n{
            categories[top_act_idx][:10]}")

        if true_scores is not None:

            ax[2].bar(
                range(len(categories)),
                true_scr.squeeze().tolist(),
                color=color_palette,
                width=bar_width,
                edgecolor='black',
                linewidth=1
            )
            ax[2].set_title(f"True Scores\nLabel: {label_value}")
            ax[2].set_xlabel("Histogram number")
            ax[2].set_ylabel("Anomaly Probability")

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_scores(
        scores: torch.Tensor,
        categories: list,
        histogram: torch.Tensor,
        labels: torch.Tensor,
        save_path: Path,
        reference: torch.Tensor | None = None,
        true_scores: torch.Tensor | None = None):

    num_samples = len(scores)
    ncols = 2 + (1 if reference is not None else 0) + \
        (1 if true_scores is not None else 0)
    _, axes = plt.subplots(
        nrows=num_samples, ncols=ncols, figsize=(10, 5 * num_samples))
    axes = np.atleast_2d(axes)

    for i, ax in enumerate(axes):
        label_value = labels[i].item()
        score_value = scores[i].item()
        top_act_idx = scores[i].argmax().item()
        hist = histogram[i, top_act_idx].detach().cpu().numpy()
        scr = scores.detach().cpu().numpy()
        true_scr = true_scores[i].detach().cpu().numpy(
        ) if true_scores is not None else None

        color_palette = sns.color_palette("viridis", len(categories))
        bar_width = 0.6

        ax[0].bar(
            list(range(len(categories))),
            scr.squeeze(),
            color=color_palette,
            width=bar_width,
            edgecolor='black',
            linewidth=1
        )
        ax[0].set_title(f"Scores\nLabel: {label_value}, Score: {score_value}")
        ax[0].set_xlabel("Histogram number")
        ax[0].set_ylabel("Anomaly Probability")
        ax[1].plot(hist)
        ax[1].set_title(f"Histogram with Strongest Activation\n{
                        categories[top_act_idx][:10]}")
        if true_scores is not None:

            ax[2].bar(
                range(len(categories)),
                true_scr.squeeze().tolist(),
                color=color_palette,
                width=bar_width,
                edgecolor='black',
                linewidth=1
            )
            ax[2].set_title(f"True Scores\nLabel: {label_value}")
            ax[2].set_xlabel("Histogram number")
            ax[2].set_ylabel("Anomaly Probability")
        if reference is not None:
            ax[2].plot(reference[top_act_idx].detach().cpu().numpy())
            ax[2].set_title(f"Reference Histogram")

    plt.tight_layout()
    plt.savefig(save_path)
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


def compute_results_summary(total_scores: np.ndarray,
                            total_labels: np.ndarray,
                            total_scores_per_var: np.ndarray | None = None,
                            total_labels_per_var: np.ndarray | None = None):

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
    total_preds = np.array(total_preds)
    total_labels = np.array(total_labels)

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
        plt.savefig(out_dir / f"{metric_name}.png")
        plt.close()
