import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import f_oneway, wilcoxon
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split


def plot_heatmaps(df: pd.DataFrame) -> None:
    """
    Plots heatmaps of ACC Aggregated and AUC Aggregated for Server rounds = 5,
    comparing XGBoost vs XGBoost + RuleCOSI across Clients and Local rounds.
    """
    # Filter to server rounds = 5
    df5 = df[df["Server rounds"] == 5]

    # Sorted unique values for consistent axis ordering
    clients_vals = sorted(df5["Clients"].unique())
    local_vals = sorted(df5["Local rounds"].unique())

    metrics = [("ACC Aggregated", "Aggregated Accuracy"), ("AUC Aggregated", "Aggregated AUC")]

    for metric_col, metric_name in metrics:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        for ax, rule, title in zip(
            axes, [False, True], ["XGBoost", "XGBoost + RuleCOSI"], strict=True
        ):
            df_rule = df5[df5["RuleCOSI"] == rule]
            # Pivot to Clients x Local rounds
            pivot = df_rule.pivot_table(
                index="Clients", columns="Local rounds", values=metric_col, aggfunc="mean"
            ).reindex(index=clients_vals, columns=local_vals)

            # Plot heatmap
            im = ax.imshow(pivot.values, aspect="auto", origin="lower")
            ax.set_xticks(range(len(local_vals)))
            ax.set_xticklabels(local_vals)
            ax.set_yticks(range(len(clients_vals)))
            ax.set_yticklabels(clients_vals)
            ax.set_xlabel("Local rounds")
            ax.set_ylabel("Clients")
            ax.set_title(f"{title}\n{metric_name}")
            fig.colorbar(im, ax=ax, orientation="vertical", label=metric_name)

        fig.suptitle(f"Heatmaps of {metric_name} (Server rounds = 5)")
        plt.tight_layout(rect=(0, 0.03, 1, 0.95))
        plt.show()


# ---- Main analysis pipeline ----
def run_benchmark_analysis(df: pd.DataFrame) -> None:
    # ---- Descriptive summaries ----
    summary = (
        df.groupby(["Clients", "RuleCOSI"])
        .agg(
            acc_mean=("ACC Aggregated", "mean"),
            acc_std=("ACC Aggregated", "std"),
            auc_mean=("AUC Aggregated", "mean"),
            auc_std=("AUC Aggregated", "std"),
        )
        .reset_index()
    )
    print("Summary statistics by Clients and RuleCOSI:")
    print(summary)

    # ---- Boxplots by RuleCOSI ----
    metrics = ["ACC Aggregated", "AUC Aggregated"]
    for m in metrics:
        plt.figure()
        data_true = df[df["RuleCOSI"]][m]
        data_false = df[~df["RuleCOSI"]][m]
        # Two boxes: first for XGBoost alone, second for XGBoost + RuleCOSI
        plt.boxplot([data_false, data_true], patch_artist=True)
        plt.xticks([1, 2], ["XGBoost", "XGBoost + RuleCOSI"])
        plt.title(f"Boxplot of {m} by Method")
        plt.ylabel(m)
        plt.xlabel("Method")
        plt.tight_layout()
        plt.show()

    # clients 2 acc server rounds 5

    plot_heatmaps(df)
    # ---- Heatmaps of Aggregated Accuracy ----
    # clients_vals = sorted(df["Clients"].unique())
    # for C in clients_vals:
    #     sub = df[df["Clients"] == C]
    #     if sub.empty:
    #         continue
    #     for flag in (False, True):
    #         sel = sub[sub["RuleCOSI"]] if flag else sub[~sub["RuleCOSI"]]
    #         label = "XGBoost + RuleCOSI" if flag else "XGBoost"
    #         pivot = sel.pivot(
    #             index="Local rounds", columns="Server rounds", values="ACC Aggregated"
    #         )
    #         plt.figure()
    #         plt.imshow(pivot, aspect="auto", origin="lower")
    #         plt.colorbar(label="ACC Aggregated")
    #         plt.title(f"Clients={C}, Method={label}: Accuracy heatmap")
    #         plt.xlabel("Server rounds")
    #         plt.ylabel("Local rounds")
    #         plt.xticks(ticks=np.arange(len(pivot.columns)), labels=pivot.columns)
    #         plt.yticks(ticks=np.arange(len(pivot.index)), labels=pivot.index)
    #         plt.tight_layout()
    #         plt.show()

    # ---- 5. Statistical tests ----
    paired = df.pivot_table(
        index=["Clients", "Server rounds", "Local rounds"],
        columns="RuleCOSI",
        values="ACC Aggregated",
    ).dropna()
    stat, p = wilcoxon(paired[False], paired[True])
    print(f"Wilcoxon test (ACC Aggregated) stat={stat:.3f}, p={p:.3e}")

    groups = [grp["ACC Aggregated"].values for _, grp in df.groupby("Server rounds")]
    f_stat, f_p = f_oneway(*groups)
    print(f"ANOVA across Server rounds: F={f_stat:.3f}, p={f_p:.3e}")

    # ---- 6. Feature importance via Random Forest ----
    X = df[["Clients", "Server rounds", "Local rounds"]].copy()
    X["RuleCOSI"] = df["RuleCOSI"].astype(int)
    y = df["ACC Aggregated"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    importances = rf.feature_importances_
    feat_imp = pd.Series(importances, index=X.columns).sort_values(ascending=False)
    print("Feature importances for predicting ACC Aggregated:")
    print(feat_imp)

    # ---- 7. Pareto frontier of Accuracy vs AUC Aggregated ----
    def pareto_frontier(df: pd.DataFrame, x_col: str, y_col: str) -> pd.DataFrame:
        """
        Compute the Pareto frontier (maximizing both x_col and y_col).

        Args:
            df: DataFrame containing at least x_col and y_col.
            x_col: Name of the column for the x-axis metric.
            y_col: Name of the column for the y-axis metric.

        Returns:
            DataFrame of the Pareto-optimal points.
        """
        sorted_df = df.sort_values(x_col, ascending=False)
        frontier_rows: list[pd.Series] = []
        max_y = -np.inf
        for _, row in sorted_df.iterrows():
            current_y = row[y_col]
            if current_y > max_y:
                frontier_rows.append(row)
                max_y = current_y
        return pd.DataFrame(frontier_rows)

    pf = pareto_frontier(df, "ACC Aggregated", "AUC Aggregated")
    plt.figure()
    plt.scatter(df["ACC Aggregated"], df["AUC Aggregated"], alpha=0.3)
    plt.scatter(pf["ACC Aggregated"], pf["AUC Aggregated"], label="Pareto frontier")
    plt.title("Pareto frontier (Accuracy vs AUC)")
    plt.xlabel("ACC Aggregated")
    plt.ylabel("AUC Aggregated")
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    df = pd.read_csv("output/benchmark-1745085664/benchmark-results-to-copy.tsv", sep="\t")
    run_benchmark_analysis(df)
