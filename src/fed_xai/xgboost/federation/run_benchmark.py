from sklearn.model_selection import ParameterGrid

from fed_xai.xgboost.federation.run_xgb_simulation import run_xgb_simulation


def run_benchmark() -> None:
    param_grid = {
        "clients": [2, 3, 4, 5],
        "server_rounds": [5, 10],
        "local_rounds": [1, 2, 3, 4],
    }

    for params in ParameterGrid(param_grid):
        # params is already a dict
        run_xgb_simulation(**params)


if __name__ == "__main__":
    run_benchmark()
