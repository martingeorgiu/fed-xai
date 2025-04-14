from flwr.common import Context, Metrics, Parameters
from flwr.server import ServerAppComponents, ServerConfig

from fed_xai.federation.xgboost.xgb_save_model_strategy import XGBSaveModelStrategy


def evaluate_metrics_aggregation(
    eval_metrics: list[tuple[int, Metrics]],
) -> Metrics:
    """Return an aggregated metric (AUC) for evaluation."""
    total_num = sum([num for num, _ in eval_metrics])

    acc_list: list[float] = []
    auc_list: list[float] = []
    for num, metrics in eval_metrics:
        acc = metrics["acc"]
        auc = metrics["auc"]
        if not isinstance(acc, int | float) or not isinstance(auc, int | float):
            raise ValueError("ACC and AUC metric must be numeric (int or float)")
        acc_list.append(acc * num)
        auc_list.append(auc * num)

    # # Calculate average AUC accross all clients
    weighted_acc = sum(acc_list) / total_num
    weighted_auc = sum(auc_list) / total_num

    metrics_aggregated: Metrics = {
        "acc": weighted_acc,
        "auc": weighted_auc,
    }
    return metrics_aggregated


def config_func(rnd: int) -> dict[str, str]:
    """Return a configuration with global epochs."""
    config = {
        "global_round": str(rnd),
    }
    return config


def xgb_server_fn(context: Context) -> ServerAppComponents:
    # Read from config
    num_rounds = 2
    fraction_fit = 1.0
    fraction_evaluate = 1.0

    # Init an empty Parameter
    parameters = Parameters(tensor_type="", tensors=[])

    # Define strategy
    strategy = XGBSaveModelStrategy(
        shouldSave=True,
        # shouldSave=False,
        fraction_fit=fraction_fit,
        fraction_evaluate=fraction_evaluate,
        evaluate_metrics_aggregation_fn=evaluate_metrics_aggregation,
        on_evaluate_config_fn=config_func,
        on_fit_config_fn=config_func,
        initial_parameters=parameters,
    )
    config = ServerConfig(num_rounds=num_rounds)

    return ServerAppComponents(strategy=strategy, config=config)
