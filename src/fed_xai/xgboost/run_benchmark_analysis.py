from itertools import product

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import f_oneway, wilcoxon
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split


# ---- Main analysis pipeline ----
def run_benchmark_analysis(df: pd.DataFrame) -> None:
    # Ensure the boolean flag exists
    if "RuleCOSI" not in df.columns:
        df = add_rulecosi_flag(df)

    # ---- 1. Descriptive summaries ----
    summary = (
        df.groupby(["Clients", "RuleCOSI"])
        .agg(
            acc_mean=("Accuracy aggregated", "mean"),
            acc_std=("Accuracy aggregated", "std"),
            auc_mean=("AUC aggregated", "mean"),
            auc_std=("AUC aggregated", "std"),
        )
        .reset_index()
    )
    print("Summary statistics by Clients and RuleCOSI:")
    print(summary)

    # ---- 2. Boxplots by RuleCOSI ----
    metrics = ["Accuracy aggregated", "AUC aggregated"]
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

    # ---- 3. Line plots vs Server rounds ----
    clients_vals = sorted(df["Clients"].unique())
    local_vals = sorted(df["Local rounds"].unique())
    for C, L in product(clients_vals, local_vals):
        sub = df[(df["Clients"] == C) & (df["Local rounds"] == L)]
        if sub.empty:
            continue
        plt.figure()
        for flag in (False, True):
            grp = sub[sub["RuleCOSI"]] if flag else sub[~sub["RuleCOSI"]]
            label = "XGBoost + RuleCOSI" if flag else "XGBoost"
            plt.plot(grp["Server rounds"], grp["Accuracy aggregated"], marker="o", label=label)
        plt.title(f"Clients={C}, Local={L}: Accuracy vs Server rounds")
        plt.xlabel("Server rounds")
        plt.ylabel("Accuracy aggregated")
        plt.legend()
        plt.tight_layout()
        plt.show()

    # ---- 4. Heatmaps of aggregated Accuracy ----
    for C in clients_vals:
        sub = df[df["Clients"] == C]
        if sub.empty:
            continue
        for flag in (False, True):
            sel = sub[sub["RuleCOSI"]] if flag else sub[~sub["RuleCOSI"]]
            label = "XGBoost + RuleCOSI" if flag else "XGBoost"
            pivot = sel.pivot(
                index="Local rounds", columns="Server rounds", values="Accuracy aggregated"
            )
            plt.figure()
            plt.imshow(pivot, aspect="auto", origin="lower")
            plt.colorbar(label="Accuracy aggregated")
            plt.title(f"Clients={C}, Method={label}: Accuracy heatmap")
            plt.xlabel("Server rounds")
            plt.ylabel("Local rounds")
            plt.xticks(ticks=np.arange(len(pivot.columns)), labels=pivot.columns)
            plt.yticks(ticks=np.arange(len(pivot.index)), labels=pivot.index)
            plt.tight_layout()
            plt.show()

    # ---- 5. Statistical tests ----
    paired = df.pivot_table(
        index=["Clients", "Server rounds", "Local rounds"],
        columns="RuleCOSI",
        values="Accuracy aggregated",
    ).dropna()
    stat, p = wilcoxon(paired[False], paired[True])
    print(f"Wilcoxon test (Accuracy aggregated) stat={stat:.3f}, p={p:.3e}")

    groups = [grp["Accuracy aggregated"].values for _, grp in df.groupby("Server rounds")]
    f_stat, f_p = f_oneway(*groups)
    print(f"ANOVA across Server rounds: F={f_stat:.3f}, p={f_p:.3e}")

    # ---- 6. Feature importance via Random Forest ----
    X = df[["Clients", "Server rounds", "Local rounds"]].copy()
    X["RuleCOSI"] = df["RuleCOSI"].astype(int)
    y = df["Accuracy aggregated"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    importances = rf.feature_importances_
    feat_imp = pd.Series(importances, index=X.columns).sort_values(ascending=False)
    print("Feature importances for predicting Accuracy aggregated:")
    print(feat_imp)

    # ---- 7. Pareto frontier of Accuracy vs AUC aggregated ----
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

    pf = pareto_frontier(df, "Accuracy aggregated", "AUC aggregated")
    plt.figure()
    plt.scatter(df["Accuracy aggregated"], df["AUC aggregated"], alpha=0.3)
    plt.scatter(pf["Accuracy aggregated"], pf["AUC aggregated"], label="Pareto frontier")
    plt.title("Pareto frontier (Accuracy vs AUC)")
    plt.xlabel("Accuracy aggregated")
    plt.ylabel("AUC aggregated")
    plt.legend()
    plt.tight_layout()
    plt.show()


# ---- Utility: add RuleCOSI flag based on 'Round' column ----
def add_rulecosi_flag(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds a boolean 'RuleCOSI' column to the DataFrame based on the 'Round' column.

    If df['Round'] is the string 'RuleCosi', then RuleCOSI=True; otherwise False.

    Args:
        df: Original DataFrame containing a 'Round' column.

    Returns:
        DataFrame with an added 'RuleCOSI' boolean column.
    """
    df = df.copy()
    df["RuleCOSI"] = df["Round"].astype(str).eq("RuleCosi")
    return df


if __name__ == "__main__":
    df = pd.read_csv("output/benchmark-1744993691/benchmark-results-to-copy.tsv", sep="\t")
    run_benchmark_analysis(df)
