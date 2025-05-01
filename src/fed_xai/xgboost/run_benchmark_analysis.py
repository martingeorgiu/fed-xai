# ruff: noqa: E712
import os
import shutil

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

StatisticalMetric = tuple[str, str]
StatisticalMetrics = list[StatisticalMetric]

working_dir = "output/benchmark-1745483011/"
analysis_dir = f"{working_dir}analysis/"


def path_save_name(name: str) -> str:
    return name.replace(" ", "_").replace("+", "").lower()


def process_plt(name: str) -> None:
    plt.tight_layout()
    save_name = path_save_name(name)
    plt.savefig(f"{analysis_dir}{save_name}.pdf", format="pdf", bbox_inches="tight")
    # Uncomment to show each plot
    # plt.show()


def drop_create_analysis_dir() -> None:
    """
    Drops the analysis directory if it exists and creates a new one.
    """
    if os.path.exists(analysis_dir):
        shutil.rmtree(analysis_dir)
    os.makedirs(analysis_dir, exist_ok=True)


def summary_statistics(avg: pd.DataFrame, metrics: StatisticalMetrics) -> None:
    """
    Prints summary statistics (mean, median, std, min, max) of ACC and AUC
    for averaged rows, grouped by Clients and RuleCOSI.
    """
    for metric, metric_name in metrics:
        summary_with_clients = (
            avg.groupby(["Clients", "RuleCOSI"])[metric]
            .agg(["mean", "median", "std", "min", "max"])
            .reset_index()
        )
        summary_without_clients = (
            avg.groupby(["RuleCOSI"])[metric]
            .agg(["mean", "median", "std", "min", "max"])
            .reset_index()
        )

        save_name = path_save_name(metric_name)
        summary_with_clients.drop(columns="RuleCOSI").to_latex(
            f"{analysis_dir}summary_{save_name}_client_number_split.tex",
            index=False,
            float_format="%.3f",
        )
        summary_without_clients.drop(columns="RuleCOSI").to_latex(
            f"{analysis_dir}summary_{save_name}.tex",
            index=False,
            float_format="%.3f",
        )
        print(f"Summary of {metric_name}:")
        print(summary_with_clients.to_string(index=False))
        print(f"General summary of {metric_name}:")
        print(summary_without_clients.to_string(index=False))


def plot_instance_variability(raw: pd.DataFrame, metrics: StatisticalMetrics) -> None:
    """
    Plots the variability (standard deviation) of ACC and AUC
    across runs (Averaged == False), comparing XGBoost vs XGBoost + RuleCOSI.
    """
    for metric, metric_name in metrics:
        variability = (
            raw.groupby(["Clients", "Server rounds", "Local rounds", "RuleCOSI"])[metric]
            .std()
            .reset_index(name="STD")
            .dropna(subset=["STD"])
        )

        fig, ax = plt.subplots(figsize=(6, 4))
        data_xgb = variability[variability["RuleCOSI"] == False]["STD"]
        data_rc = variability[variability["RuleCOSI"] == True]["STD"]

        ax.boxplot([data_xgb, data_rc], patch_artist=True)
        ax.set_xticks([1, 2])
        ax.set_xticklabels(["XGBoost", "XGBoost + RuleCOSI+"])
        ax.set_ylabel(f"STD of {metric_name}")
        ax.set_title(f"Instance Variability of {metric_name}")
        process_plt(f"instance_variability_{metric_name}")


def plot_global_vs_aggregated(avg: pd.DataFrame) -> None:
    """
    Compares Global vs Aggregated metrics (ACC and AUC) on averaged rows,
    side-by-side for XGBoost vs XGBoost + RuleCOSI.
    """
    metrics = [
        ("ACC Global", "ACC Aggregated", "Accuracy"),
        ("AUC Global", "AUC Aggregated", "ROC AUC"),
    ]
    for global_col, agg_col, name in metrics:
        fig, ax = plt.subplots(figsize=(8, 4))
        xgb = avg[avg["RuleCOSI"] == False][[global_col, agg_col]]
        rc = avg[avg["RuleCOSI"] == True][[global_col, agg_col]]
        data = [xgb[global_col], xgb[agg_col], rc[global_col], rc[agg_col]]
        ax.boxplot(data, patch_artist=True)
        ax.set_xticks([1, 2, 3, 4])
        ax.set_xticklabels(
            [
                f"XGBoost\nGlobal {name}",
                f"XGBoost\nAggregated {name}",
                f"XGBoost + RuleCOSI+\nGlobal {name}",
                f"XGBoost + RuleCOSI+\nAggregated {name}",
            ],
            rotation=45,
            ha="right",
        )
        ax.set_title(f"Global vs Aggregated {name}")
        ax.set_ylabel(name)
        process_plt(f"global_vs_aggregated_{name}")


def plot_metrics(avg: pd.DataFrame, metrics: StatisticalMetrics) -> None:
    """
    Plots boxplots of ACC and AUC for averaged rows,
    comparing XGBoost vs XGBoost + RuleCOSI.
    """
    for metric, metric_name in metrics:
        plt.figure(figsize=(6, 4))
        data_xgb = avg[avg["RuleCOSI"] == False][metric]
        data_rc = avg[avg["RuleCOSI"] == True][metric]
        plt.boxplot([data_xgb, data_rc], patch_artist=True)
        plt.xticks([1, 2], ["XGBoost", "XGBoost + RuleCOSI+"])
        plt.xlabel("Method")
        plt.ylabel(metric_name)
        plt.title(f"Boxplot of {metric_name}")
        process_plt(f"boxplot_{metric_name}")


def plot_heatmaps(avg: pd.DataFrame, metrics: StatisticalMetrics) -> None:
    """
    Plots heatmaps of ACC and AUC,
    comparing XGBoost vs XGBoost + RuleCOSI across Clients and Local rounds.
    Only uses averaged rows.
    """

    clients_vals = sorted(avg["Clients"].unique())
    local_vals = sorted(avg["Local rounds"].unique())

    for metric, metric_name in metrics:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        for ax, rule, title in zip(
            axes, [False, True], ["XGBoost", "XGBoost + RuleCOSI+"], strict=False
        ):
            df_rule = avg[avg["RuleCOSI"] == rule]
            pivot = df_rule.pivot_table(
                index="Clients", columns="Local rounds", values=metric, aggfunc="mean"
            ).reindex(index=clients_vals, columns=local_vals)

            im = ax.imshow(pivot.values, aspect="auto", origin="lower")
            ax.set_xticks(range(len(local_vals)))
            ax.set_xticklabels(local_vals)
            ax.set_yticks(range(len(clients_vals)))
            ax.set_yticklabels(clients_vals)
            ax.set_xlabel("Local rounds")
            ax.set_ylabel("Clients")
            ax.set_title(f"{title}\n{metric_name}")
            fig.colorbar(im, ax=ax, orientation="vertical", label=metric_name)

        fig.suptitle(f"Heatmaps of {metric_name}")
        process_plt(f"heatmap_{metric_name}")


def plot_pareto_frontier(
    datasets: list[tuple[pd.DataFrame, str]], x: StatisticalMetric, y: StatisticalMetric
) -> None:
    """
    Plots the Pareto frontier of ACC vs AUC for
    averaged rows. The Pareto frontier shows the
    set of points where no other configuration is strictly better in both
    metrics.
    """
    # Sort by ACC descending

    x_column, x_label = x
    y_column, y_label = y

    for dataset, dataset_name in datasets:
        sorted_df = dataset.sort_values(x_column, ascending=False)
        pareto = []
        max_auc = -np.inf
        for _, row in sorted_df.iterrows():
            auc = row[y_column]
            if auc > max_auc:
                pareto.append(row)
                max_auc = auc
        pareto_df = pd.DataFrame(pareto)
        print(f"Pareto Frontier of {dataset_name}:")
        print(pareto_df)
        # Plot all points and frontier
        plt.figure(figsize=(6, 5))
        plt.scatter(dataset[x_column], dataset[y_column], alpha=0.4, label="Configs")
        plt.scatter(
            pareto_df[x_column],
            pareto_df[y_column],
            color="red",
            label="Pareto Frontier",
        )
        plt.title(f"Pareto Frontier of {dataset_name}")
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.legend()
        process_plt(f"pareto_frontier_{path_save_name(dataset_name)}_{x_label}_{y_label}")


def run_benchmark_analysis(df: pd.DataFrame) -> None:
    """
    Full analysis on the provided DataFrame:
      1. Summary statistics of ACC and AUC
      2. Instance variability plots from raw rows
      3. Boxplot of ACC and AUC
      4. Heatmaps of ACC & AUC
      5. Pareto frontier
    """
    # Split averaged vs raw
    avg = df[df["Averaged"] == True]
    raw = df[df["Averaged"] == False]

    data_xgb = raw[raw["RuleCOSI"] == False]
    data_rc = raw[raw["RuleCOSI"] == True]
    # datasets = [(avg, "Averaged Data"), (raw, "All Data")]
    datasets = [(data_xgb, "XGBoost"), (data_rc, "XGBoost + RuleCOSI+")]

    relevant_metrics = [("ACC Global", "Accuracy"), ("AUC Global", "ROC AUC")]

    x = relevant_metrics[0]
    y = relevant_metrics[1]

    drop_create_analysis_dir()

    summary_statistics(avg, relevant_metrics)

    plot_instance_variability(raw, relevant_metrics)
    plot_global_vs_aggregated(avg)
    plot_metrics(avg, relevant_metrics)

    plot_heatmaps(avg, relevant_metrics)

    plot_pareto_frontier(datasets, x, y)


# Example usage:
if __name__ == "__main__":
    df = pd.read_csv(f"{working_dir}benchmark-results-to-copy.tsv", sep="\t")
    run_benchmark_analysis(df)
