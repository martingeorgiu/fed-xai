import pandas as pd
from flwr.client import ClientApp
from flwr.server import ServerApp
from flwr.simulation import run_simulation

from fed_xai.federation.xgboost.const import num_rounds, rules_suffix
from fed_xai.federation.xgboost.xgb_client_app import xgb_client_fn
from fed_xai.federation.xgboost.xgb_server_app import (
    acc_aggregates,
    acc_globals,
    auc_aggregates,
    auc_globals,
    xgb_server_fn,
)
from fed_xai.helpers.cleanup_output import cleanup_output, model_path
from fed_xai.helpers.generate_xgb_visualization import generate_xgb_visualization
from fed_xai.helpers.rulecosi_helpers import bytes_to_ruleset
from fed_xai.xgboost.booster_to_classifier import load_booster_from_bytes
from fed_xai.xgboost.const import booster_params_from_hp


def main() -> None:
    cleanup_output()

    client_app1 = ClientApp(
        xgb_client_fn,
    )
    server_app1 = ServerApp(
        server_fn=lambda ctx: xgb_server_fn(ctx, num_rounds, last_round_rulecosi=False),
    )

    run_simulation(
        server_app=server_app1,
        client_app=client_app1,
        num_supernodes=2,
    )

    max_auc = max(auc_globals)
    max_auc_index = auc_globals.index(max_auc)
    # Round numbers are counted from 1
    max_auc_round = max_auc_index + 1
    # results = f"{acc_globals[max_auc_index]}\t{acc_aggregates[max_auc_index]}\t{auc_globals[max_auc_index]}\t{auc_aggregates[max_auc_index]}\t{max_auc_round}"  # noqa: E501
    df_xgb = pd.DataFrame(
        data={
            "Accuracy": [acc_globals[max_auc_index]],
            "Accuracy aggregated": [acc_aggregates[max_auc_index]],
            "AUC": [auc_globals[max_auc_index]],
            "AUC aggregated": [auc_aggregates[max_auc_index]],
            "Round": [max_auc_round],
        }
    )
    print(df_xgb.to_string(index=False))

    with open(model_path(str(max_auc_round)), "rb") as file:
        best_xgb_model = file.read()

    generate_xgb_visualization(load_booster_from_bytes(booster_params_from_hp, best_xgb_model))

    client_app2 = ClientApp(
        xgb_client_fn,
    )
    server_app2 = ServerApp(
        server_fn=lambda ctx: xgb_server_fn(
            ctx, 1, last_round_rulecosi=True, initial_data=best_xgb_model
        ),
    )

    run_simulation(
        server_app=server_app2,
        client_app=client_app2,
        num_supernodes=2,
    )
    # Indexes are counted from 0  but rounds from 1, so the last round is num_rounds
    rulecosi_index = num_rounds
    df_rules = pd.DataFrame(
        data={
            "Accuracy": [acc_globals[rulecosi_index]],
            "Accuracy aggregated": [acc_aggregates[rulecosi_index]],
            "AUC": [auc_globals[rulecosi_index]],
            "AUC aggregated": [auc_aggregates[rulecosi_index]],
            "Round": ["RuleCosi"],
        }
    )

    with open(model_path(rules_suffix), "rb") as file:
        best_xgb_model = file.read()
    ruleset = bytes_to_ruleset(best_xgb_model)

    with open("output/ruleset.txt", "w") as file:
        file.write(str(ruleset))
    with open("output/benchmark.txt", "w") as file:
        file.write(
            "XGBoost - Results:\n"
            + df_xgb.to_string(index=False)
            + "\n\nRuleCOSI - Results:\n"
            + df_rules.to_string(index=False)
        )

    # preparation for testing and saving to clipboard
    df_rules.to_clipboard(index=False, sep="\t", header=None)


if __name__ == "__main__":
    main()
