import numpy as np


def compute_results_summary(metrics: list[dict]):

    metrics_summary_dict = {}

    for run_metrics in metrics:
        for metric_name, metric_val in run_metrics.items():
            if metric_name != "support":
                try:
                    metrics_summary_dict[metric_name].append(metric_val)
                except KeyError:
                    metrics_summary_dict[metric_name] = [metric_val]

    res = []
    for metric_name, metric_values in metrics_summary_dict.items():
        mean = np.mean(metric_values)
        std = np.std(metric_values)
        res.append(f"{metric_name} = {mean:.3f} +- {std:.3f}")

    return "\n".join(res)
