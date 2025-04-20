import random
import time
from typing import Any

import pandas as pd
from sklearn.model_selection import ParameterGrid

from fed_xai.xgboost.federation.run_xgb_simulation import run_xgb_simulation


def run_benchmark() -> None:
    unix_time = int(time.time())
    benchmark_name = f"benchmark-{unix_time}/"
    # param_grid = {
    #     "clients": [2, 3, 4, 5],
    #     "server_rounds": [10],
    #     "local_rounds": [1, 2, 3, 4],
    # }
    # single_param_rounds = 8
    single_param_rounds = 2

    # Test quick sanity benchmark
    param_grid = {
        "clients": [2, 3],
        "server_rounds": [10],
        "local_rounds": [2],
    }
    results = get_df()

    for params in ParameterGrid(param_grid):
        print(
            f"\n\n\n-----------------------------------------------------------------------------------------------------------\nRunning simulation with parameters: {params}\n-----------------------------------------------------------------------------------------------------------\n\n"  # noqa: E501
        )
        param_results = get_df()
        for _ in range(single_param_rounds):
            # Randomize the seed for each simulation
            random_state = random.randint(1, 10000)
            df = run_xgb_simulation(**params, random_state=random_state, path_prefix=benchmark_name)
            df["Clients"] = params["clients"]
            df["Server rounds"] = params["server_rounds"]
            df["Local rounds"] = params["local_rounds"]
            df["Averaged"] = False
            param_results = pd.concat([param_results, df], ignore_index=True)[param_results.columns]

        cols_to_avg = ["ACC Global", "ACC Aggregated", "AUC Global", "AUC Aggregated", "Round"]

        averages_xgb = param_results[param_results["RuleCOSI"] == False][cols_to_avg].mean()  # noqa: E712
        averages_rulecosi = param_results[param_results["RuleCOSI"] == True][cols_to_avg].mean()  # noqa: E712

        data = [
            get_averaged_data(params, averages_xgb, False),
            get_averaged_data(params, averages_rulecosi, True),
        ]
        averages = get_df(data)
        param_results = pd.concat([param_results, averages], ignore_index=True)[
            param_results.columns
        ]

        results = pd.concat([results, param_results], ignore_index=True)[results.columns]

        print(
            f"\n\n\n-----------------------------------------------------------------------------------------------------------\Simulation with parameters done: {params}\n{param_results}\n-----------------------------------------------------------------------------------------------------------\n\n"  # noqa: E501
        )

    print(results)
    with open(f"output/{benchmark_name}benchmark-results.txt", "w") as file:
        file.write(results.to_string(index=True))
    results.to_csv(
        f"output/{benchmark_name}benchmark-results-to-copy.tsv",
        sep="\t",
        index=False,
        encoding="utf-8",
    )


def get_averaged_data(params: dict[str, Any], averages: pd.Series, rulecosi: bool) -> list:
    return [
        params["clients"],
        params["server_rounds"],
        params["local_rounds"],
        round(averages["ACC Global"], 4),
        round(averages["ACC Aggregated"], 4),
        round(averages["AUC Global"], 4),
        round(averages["AUC Aggregated"], 4),
        rulecosi,
        round(averages["Round"], 2),
        True,
    ]


def get_df(data: Any | None = None) -> pd.DataFrame:  # noqa: ANN401
    return pd.DataFrame(
        data=data,
        columns=[
            "Clients",
            "Server rounds",
            "Local rounds",
            "ACC Global",
            "ACC Aggregated",
            "AUC Global",
            "AUC Aggregated",
            "RuleCOSI",
            "Round",
            "Averaged",
        ],
    )


if __name__ == "__main__":
    run_benchmark()
