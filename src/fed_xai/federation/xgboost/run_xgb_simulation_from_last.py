import pandas as pd
import pyperclip
from flwr.client import ClientApp
from flwr.server import ServerApp
from flwr.simulation import run_simulation

from fed_xai.federation.xgboost.const import num_rounds
from fed_xai.federation.xgboost.xgb_client_app import xgb_client_fn
from fed_xai.federation.xgboost.xgb_server_app import (
    acc_aggregates,
    acc_globals,
    auc_aggregates,
    auc_globals,
    xgb_server_fn,
)
from fed_xai.helpers.cleanup_output import cleanup_output


def main() -> None:
    cleanup_output()

    client_app = ClientApp(
        xgb_client_fn,
    )
    server_app = ServerApp(
        server_fn=lambda ctx: xgb_server_fn(ctx, num_rounds, last_round_rulecosi=True),
    )

    run_simulation(
        server_app=server_app,
        client_app=client_app,
        num_supernodes=2,
    )

    print("\nSimulation finished")
    max_auc = max(auc_globals)
    rules_index = len(acc_aggregates) - 1
    max_auc_index = auc_globals.index(max_auc)
    # Round numbers are counted from 1
    max_auc_round = max_auc_index + 1
    rules_round = rules_index + 1
    results = f"{acc_globals[max_auc_index]}\t{acc_aggregates[max_auc_index]}\t{auc_globals[max_auc_index]}\t{auc_aggregates[max_auc_index]}\t{max_auc_round}"  # noqa: E501
    df = pd.DataFrame(
        data={
            "Accuracy": [acc_globals[max_auc_index], acc_globals[rules_index]],
            "Accuracy aggregated": [acc_aggregates[max_auc_index], acc_aggregates[rules_index]],
            "AUC": [auc_globals[max_auc_index], auc_globals[rules_index]],
            "AUC aggregated": [auc_aggregates[max_auc_index], auc_aggregates[rules_index]],
            "Round": [max_auc_round, f"RuleCosi ({rules_round})"],
        }
    )
    print(df.to_string(index=False))

    print("\nResults copied to clipboard")
    pyperclip.copy(results)


if __name__ == "__main__":
    main()
