from flwr.common import FitRes, Parameters, Scalar
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedXgbBagging

from fed_xai.explainers.combining_rulecosi_explainer import combine_rulesets
from fed_xai.helpers.rulecosi_helpers import (
    bytes_to_ruleset,
    create_empty_rulecosi,
    ruleset_to_bytes,
)


class XGBSaveModelStrategy(FedXgbBagging):
    def __init__(
        self, num_rounds: int, last_round_rulecosi: bool, should_save: bool, *args, **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)
        self.num_rounds = num_rounds
        self.last_round_rulecosi = last_round_rulecosi
        self.should_save = should_save

    def aggregate_fit(
        self,
        server_round: int,
        results: list[tuple[ClientProxy, FitRes]],
        failures: list[tuple[ClientProxy, FitRes] | BaseException],
    ) -> tuple[Parameters | None, dict[str, Scalar]]:
        if self.last_round_rulecosi and server_round == self.num_rounds:
            results_bytes = list(map(lambda x: x[1].parameters.tensors[0], results))
            combined_rule = combine_rules(results_bytes)
            return (
                Parameters(tensor_type="", tensors=[combined_rule]),
                {},
            )

        res = super().aggregate_fit(
            server_round=server_round,
            results=results,
            failures=failures,
        )
        if res[0] is not None:
            bytes_model = res[0].tensors[0]
            if self.should_save:
                with open(f"output/output{server_round}.bin", "wb") as file:
                    file.write(bytes_model)
        return res


def combine_rules(input: list[bytes]) -> bytes:
    """
    Combine rules from multiple clients into a single model.
    """
    rc_combiner = create_empty_rulecosi()

    actual_rule = bytes_to_ruleset(input.pop(0))
    for i in input:
        actual_rule = combine_rulesets(
            rc_combiner,
            actual_rule,
            bytes_to_ruleset(i),
        )
    return ruleset_to_bytes(actual_rule)
