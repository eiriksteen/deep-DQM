from pathlib import Path
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

sns.set_style("white")

plt.rc("figure", figsize=(20, 10))
plt.rc("font", size=13)


def filter_flips(scores: np.ndarray,
                 preds: np.ndarray,
                 labels: np.ndarray):

    flip_idx = np.where(labels[:-1] != labels[1:])[0] + 1
    flip_scores = scores[flip_idx]
    flip_preds = preds[flip_idx]
    flip_labels = labels[flip_idx]

    return flip_scores, flip_preds, flip_labels


def compute_results_summary(total_scores: np.ndarray,
                            total_preds: np.ndarray,
                            total_labels: np.ndarray,):

    metrics = {
        "total": {
            "accuracy": [],
            "balanced_accuracy": [],
            "precision": [],
            "recall": [],
            "f1": [],
            "auroc": [],
            "auprc": []
        },
        "flips": {
            "accuracy": [],
            "balanced_accuracy": [],
            "precision": [],
            "recall": [],
            "f1": [],
            "auroc": [],
            "auprc": []
        }
    }

    for run in range(len(total_scores)):
        scores = total_scores[run]
        preds = total_preds[run]
        is_anomaly = total_labels[run]

        metrics["total"]["accuracy"].append(accuracy_score(is_anomaly, preds))
        metrics["total"]["balanced_accuracy"].append(
            balanced_accuracy_score(is_anomaly, preds))
        metrics["total"]["precision"].append(
            precision_score(is_anomaly, preds))
        metrics["total"]["recall"].append(recall_score(is_anomaly, preds))
        metrics["total"]["f1"].append(f1_score(is_anomaly, preds))
        metrics["total"]["auroc"].append(roc_auc_score(is_anomaly, scores))
        metrics["total"]["auprc"].append(
            average_precision_score(is_anomaly, scores))

        flip_scores, flip_preds, flip_labels = filter_flips(
            scores, preds, is_anomaly)

        metrics["flips"]["accuracy"].append(
            accuracy_score(flip_labels, flip_preds))
        metrics["flips"]["balanced_accuracy"].append(
            balanced_accuracy_score(flip_labels, flip_preds))
        metrics["flips"]["precision"].append(
            precision_score(flip_labels, flip_preds))
        metrics["flips"]["recall"].append(
            recall_score(flip_labels, flip_preds))
        metrics["flips"]["f1"].append(f1_score(flip_labels, flip_preds))
        metrics["flips"]["auroc"].append(
            roc_auc_score(flip_labels, flip_scores))
        metrics["flips"]["auprc"].append(
            average_precision_score(flip_labels, flip_scores))

    res = []

    for total_or_flips, metrics_dict in metrics.items():
        res += [f"{total_or_flips.upper()}"]
        for metric_name, metric_values in metrics_dict.items():
            mean = np.mean(metric_values)
            std = np.std(metric_values)
            res.append(f"{' '*5}{metric_name} = {mean:.3f} +- {std:.3f}")

    return "\n".join(res)


def plot_metrics_per_step(
        total_scores: np.ndarray,
        total_preds: np.ndarray,
        total_labels: np.ndarray,
        out_dir: Path):

    metrics_per_step = {
        "accuracy": [[] for _ in range(len(total_scores))],
        "balanced_accuracy": [[] for _ in range(len(total_scores))],
        "precision": [[] for _ in range(len(total_scores))],
        "recall": [[] for _ in range(len(total_scores))],
        "f1": [[] for _ in range(len(total_scores))],
        "auroc": [[] for _ in range(len(total_scores))],
        "auprc": [[] for _ in range(len(total_scores))]
    }

    for run in range(len(total_scores)):
        for step in range(len(total_scores[run])):
            scores = total_scores[run, :step]
            preds = total_preds[run, :step]
            is_anomaly = total_labels[run, :step]
            first_label = is_anomaly[0] if step > 0 else is_anomaly

            if not np.all(is_anomaly == first_label):

                accuracy = accuracy_score(is_anomaly, preds)
                balanced_accuracy = balanced_accuracy_score(is_anomaly, preds)
                precision = precision_score(is_anomaly, preds)
                recall = recall_score(is_anomaly, preds)
                f1 = f1_score(is_anomaly, preds)
                auroc = roc_auc_score(is_anomaly, scores)
                auprc = average_precision_score(is_anomaly, scores)

                metrics_per_step["accuracy"][run].append(accuracy)
                metrics_per_step["balanced_accuracy"][run].append(
                    balanced_accuracy)
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
