import time

import pandas as pd
from sklearn.model_selection import ParameterGrid

from fed_xai.xgboost.federation.run_xgb_simulation import run_xgb_simulation


def run_benchmark() -> None:
    unix_time = int(time.time())
    benchmark_name = f"benchmark-{unix_time}/"
    param_grid = {
        "clients": [2, 3, 4, 5],
        "server_rounds": [5, 10],
        "local_rounds": [1, 2, 3, 4],
    }
    # Test quick sanity benchmark
    # param_grid = {
    #     "clients": [2],
    #     "server_rounds": [5, 10],
    #     "local_rounds": [2],
    # }
    results = pd.DataFrame(
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
        ]
    )

    for params in ParameterGrid(param_grid):
        print(
            f"\n\n\n-----------------------------------------------------------------------------------------------------------\nRunning simulation with parameters: {params}\n-----------------------------------------------------------------------------------------------------------\n\n"  # noqa: E501
        )
        df = run_xgb_simulation(**params, path_prefix=benchmark_name)
        df["Clients"] = params["clients"]
        df["Server rounds"] = params["server_rounds"]
        df["Local rounds"] = params["local_rounds"]
        results = pd.concat([results, df], ignore_index=True)[results.columns]

    print(results)
    with open(f"output/{benchmark_name}benchmark-results.txt", "w") as file:
        file.write(results.to_string(index=True))
    results.to_csv(
        f"output/{benchmark_name}benchmark-results-to-copy.tsv",
        sep="\t",
        index=False,
        encoding="utf-8",
    )


if __name__ == "__main__":
    run_benchmark()
