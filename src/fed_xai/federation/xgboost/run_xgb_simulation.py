import pyperclip
from flwr.client import ClientApp
from flwr.server import ServerApp
from flwr.simulation import run_simulation

from fed_xai.federation.xgboost.xgb_client_app import xgb_client_fn
from fed_xai.federation.xgboost.xgb_server_app import (
    acc_aggregates,
    acc_globals,
    auc_aggregates,
    auc_globals,
    xgb_server_fn,
)


def main() -> None:
    client_app = ClientApp(
        xgb_client_fn,
    )
    server_app = ServerApp(
        server_fn=xgb_server_fn,
    )

    run_simulation(
        server_app=server_app,
        client_app=client_app,
        num_supernodes=2,
    )

    print("\nSimulation finished")
    max_auc = max(auc_globals)
    max_auc_index = auc_globals.index(max_auc)
    # Round numbers are counted from 1
    max_auc_round = max_auc_index + 1
    results = f"{acc_globals[max_auc_index]}\t{acc_aggregates[max_auc_index]}\t{auc_globals[max_auc_index]}\t{auc_aggregates[max_auc_index]}\t{max_auc_round}"  # noqa: E501
    print("Accuracy,	Accuracy aggregated,	AUC,	AUC aggregated,    AUC - Round of max")
    print(results)

    print("\nResults copied to clipboard")
    pyperclip.copy(results)


if __name__ == "__main__":
    main()
