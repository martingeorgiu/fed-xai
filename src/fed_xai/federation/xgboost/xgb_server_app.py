from flwr.common import Context, Parameters
from flwr.server import ServerAppComponents, ServerConfig

from fed_xai.federation.xgboost.xgb_save_model_strategy import XGBSaveModelStrategy


def evaluate_metrics_aggregation(eval_metrics):
    """Return an aggregated metric (AUC) for evaluation."""
    total_num = sum([num for num, _ in eval_metrics])
    auc_aggregated = sum([metrics["AUC"] * num for num, metrics in eval_metrics]) / total_num
    metrics_aggregated = {"AUC": auc_aggregated}
    return metrics_aggregated


def config_func(rnd: int) -> dict[str, str]:
    """Return a configuration with global epochs."""
    config = {
        "global_round": str(rnd),
    }
    return config


def xgb_server_fn(context: Context):
    # Read from config
    num_rounds = 10
    fraction_fit = 1.0
    fraction_evaluate = 1.0

    # Init an empty Parameter
    parameters = Parameters(tensor_type="", tensors=[])

    # Define strategy
    strategy = XGBSaveModelStrategy(
        shouldSave=False,
        fraction_fit=fraction_fit,
        fraction_evaluate=fraction_evaluate,
        evaluate_metrics_aggregation_fn=evaluate_metrics_aggregation,
        on_evaluate_config_fn=config_func,
        on_fit_config_fn=config_func,
        initial_parameters=parameters,
    )
    config = ServerConfig(num_rounds=num_rounds)

    return ServerAppComponents(strategy=strategy, config=config)
