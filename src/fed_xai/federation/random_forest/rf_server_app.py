from flwr.common import Context, Metrics, Parameters, Scalar
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from flwr.server.strategy import FedAvg

# This technique was not used eventually


def evaluate_metrics_aggregation(eval_metrics: list[tuple[int, Metrics]]) -> Metrics:
    """Return an aggregated metric (AUC) for evaluation."""
    total_num = sum([num for num, _ in eval_metrics])
    auc_sum: float = 0.0
    for num, metrics in eval_metrics:
        auc_value = metrics.get("AUC")
        if not isinstance(auc_value, int | float):
            raise TypeError("Expected numeric AUC in metrics")
        auc_sum += float(auc_value) * num
    auc_aggregated: float = auc_sum / total_num
    metrics_aggregated: Metrics = {"AUC": auc_aggregated}
    return metrics_aggregated


def config_func(rnd: int) -> dict[str, Scalar]:
    """Return a configuration with global epochs."""
    config: dict[str, Scalar] = {
        "global_round": str(rnd),
    }
    return config


def server_fn(context: Context) -> ServerAppComponents:
    # Read from config
    num_rounds = context.run_config["num-server-rounds"]
    fraction_fit = context.run_config["fraction-fit"]
    fraction_evaluate = context.run_config["fraction-evaluate"]

    # Init an empty Parameter
    parameters = Parameters(tensor_type="", tensors=[])

    # Type checks and conversions
    if not isinstance(fraction_fit, float | int):
        raise ValueError("fraction_fit must be a number (float)")
    fraction_fit = float(fraction_fit)
    if not isinstance(fraction_evaluate, float | int):
        raise ValueError("fraction_evaluate must be a number (float)")
    fraction_evaluate = float(fraction_evaluate)
    if not isinstance(num_rounds, int):
        raise ValueError("num_rounds must be an integer")

    # Define strategy
    strategy = FedAvg(
        fraction_fit=fraction_fit,
        fraction_evaluate=fraction_evaluate,
        evaluate_metrics_aggregation_fn=evaluate_metrics_aggregation,
        on_evaluate_config_fn=config_func,
        on_fit_config_fn=config_func,
        initial_parameters=parameters,
    )
    config = ServerConfig(num_rounds=num_rounds)

    return ServerAppComponents(strategy=strategy, config=config)


# Create ServerApp
app = ServerApp(
    server_fn=server_fn,
)
