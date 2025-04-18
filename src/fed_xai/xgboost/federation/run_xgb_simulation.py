import os
import time

import pandas as pd
from flwr.client import ClientApp
from flwr.server import ServerApp
from flwr.simulation import run_simulation

from fed_xai.helpers.booster_to_classifier import load_booster_from_bytes
from fed_xai.helpers.cleanup_output import model_path
from fed_xai.helpers.generate_xgb_visualization import generate_xgb_visualization
from fed_xai.helpers.rulecosi_helpers import bytes_to_ruleset
from fed_xai.xgboost.const import booster_params_from_hp, rules_suffix
from fed_xai.xgboost.federation.xgb_client_app import xgb_client_fn
from fed_xai.xgboost.federation.xgb_server_app import (
    acc_aggregates,
    acc_globals,
    auc_aggregates,
    auc_globals,
    xgb_server_fn,
)


def create_empty_result_table() -> pd.DataFrame:
    return pd.DataFrame(
        columns=["Accuracy", "Accuracy aggregated", "AUC", "AUC aggregated", "Round"]
    )


def run_xgb_simulation(clients: int, server_rounds: int, local_rounds: int) -> pd.DataFrame:
    unix_time = int(time.time())

    training_name = f"{clients}-{server_rounds}-{local_rounds}-{unix_time}"
    os.makedirs(f"output/{training_name}", exist_ok=True)

    # cleanup_output()

    client_app1 = ClientApp(
        client_fn=lambda ctx: xgb_client_fn(ctx, local_rounds),
    )
    server_app1 = ServerApp(
        server_fn=lambda ctx: xgb_server_fn(
            ctx, server_rounds, last_round_rulecosi=False, training_name=training_name
        ),
    )

    run_simulation(
        server_app=server_app1,
        client_app=client_app1,
        num_supernodes=clients,
    )

    max_auc = max(auc_globals)
    max_auc_index = auc_globals.index(max_auc)
    # Round numbers are counted from 1
    max_auc_round = max_auc_index + 1
    # results = f"{acc_globals[max_auc_index]}\t{acc_aggregates[max_auc_index]}\t{auc_globals[max_auc_index]}\t{auc_aggregates[max_auc_index]}\t{max_auc_round}"  # noqa: E501

    with open(model_path(str(max_auc_round), training_name), "rb") as file:
        best_xgb_model = file.read()

    generate_xgb_visualization(
        load_booster_from_bytes(booster_params_from_hp, best_xgb_model), training_name
    )

    client_app2 = ClientApp(
        client_fn=lambda ctx: xgb_client_fn(ctx, local_rounds),
    )
    server_app2 = ServerApp(
        server_fn=lambda ctx: xgb_server_fn(
            ctx,
            1,
            last_round_rulecosi=True,
            training_name=training_name,
            initial_data=best_xgb_model,
        ),
    )

    run_simulation(
        server_app=server_app2,
        client_app=client_app2,
        num_supernodes=2,
    )
    # Indexes are counted from 0  but rounds from 1, so the last round is num_rounds
    rulecosi_index = server_rounds

    results = pd.DataFrame(
        data={
            "Accuracy": [acc_globals[max_auc_index], acc_globals[rulecosi_index]],
            "Accuracy aggregated": [acc_aggregates[max_auc_index], acc_aggregates[rulecosi_index]],
            "AUC": [auc_globals[max_auc_index], auc_globals[rulecosi_index]],
            "AUC aggregated": [auc_aggregates[max_auc_index], auc_aggregates[rulecosi_index]],
            "Round": [max_auc_round, "RuleCosi"],
        }
    )

    with open(model_path(rules_suffix, training_name), "rb") as file:
        best_xgb_model = file.read()
    ruleset = bytes_to_ruleset(best_xgb_model)

    with open(f"output/{training_name}/ruleset.txt", "w") as file:
        file.write(str(ruleset))
    with open(f"output/{training_name}/benchmark.txt", "w") as file:
        file.write("XGBoost - Results:\n" + results.to_string(index=False))

    # preparation for testing and saving to clipboard
    # df_rules.to_clipboard(index=False, sep="\t", header=None)
    return results


if __name__ == "__main__":
    run_xgb_simulation(clients=2, server_rounds=10, local_rounds=2)
