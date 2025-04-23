from flwr.common import FitRes, Parameters, Scalar
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedXgbBagging
from rulecosi import RuleSet

from fed_xai.explainers.combining_rulecosi_explainer import combine_rulesets
from fed_xai.helpers.cleanup_output import model_path, save_path
from fed_xai.helpers.rulecosi_helpers import (
    bytes_to_ruleset,
    create_empty_rulecosi,
    ruleset_to_bytes,
)
from fed_xai.xgboost.const import rules_suffix

AggregateRes = tuple[Parameters | None, dict[str, Scalar]]


class XGBSaveModelStrategy(FedXgbBagging):
    def __init__(
        self,
        num_rounds: int,
        last_round_rulecosi: bool,
        should_save: bool,
        random_state: int,
        training_name: str | None = None,
        *args,  # noqa: ANN002
        **kwargs,  # noqa: ANN003
    ) -> None:
        super().__init__(*args, **kwargs)
        self.num_rounds = num_rounds
        self.last_round_rulecosi = last_round_rulecosi
        self.should_save = should_save
        self.random_state = random_state
        self.training_name = training_name

    def aggregate_fit(
        self,
        server_round: int,
        results: list[tuple[ClientProxy, FitRes]],
        failures: list[tuple[ClientProxy, FitRes] | BaseException],
    ) -> AggregateRes:
        if self.last_round_rulecosi and server_round == self.num_rounds:
            # Save model from all clients

            results_bytes = list(map(lambda x: x[1].parameters.tensors[0], results))

            # save each RuleCOSI models from clients
            for idx, input_bytes in enumerate(results_bytes):
                self.save_model(rules_suffix + f"{idx}", bytes_to_res(input_bytes))
                self.save_ruleset(input_bytes, idx)

            combined_rules = combine_rules(results_bytes, self.random_state)
            self.save_ruleset(combined_rules)

            res_rules = bytes_to_res(combined_rules)
            self.save_model(rules_suffix, res_rules)
            return res_rules

        res_xgb = super().aggregate_fit(
            server_round=server_round,
            results=results,
            failures=failures,
        )
        self.save_model(str(server_round), res_xgb)
        return res_xgb

    def save_model(self, suffix: str, res: AggregateRes) -> None:
        if res[0] is not None:
            bytes_model = res[0].tensors[0]
            if self.should_save:
                with open(model_path(suffix, self.training_name), "wb") as file:
                    file.write(bytes_model)

    def save_ruleset(self, input_bytes: bytes, id: int | None = None) -> None:
        suffix = f"{id}" if id is not None else ""
        ruleset = bytes_to_ruleset(input_bytes)
        with open(save_path(self.training_name) + f"ruleset{suffix}.txt", "w") as file:
            file.write(str(ruleset))


def bytes_to_res(input: bytes) -> AggregateRes:
    return (
        Parameters(tensor_type="", tensors=[input]),
        {},
    )


def combine_rules(input: list[bytes], random_state: int) -> bytes:
    """
    Combine rules from multiple clients into a single model.
    """
    rc_combiner = create_empty_rulecosi(random_state)

    actual_rule = bytes_to_ruleset(input.pop(0))
    for i in input:
        actual_rule = combine_rulesets(
            rc_combiner,
            actual_rule,
            bytes_to_ruleset(i),
        )
    return ruleset_to_bytes(actual_rule)


# Not used, but kept for future reference
def combine_rules2(input: list[bytes], random_state: int) -> bytes:
    """
    Combine rules from multiple clients into a single model.
    """
    rc_combiner = create_empty_rulecosi(random_state)

    rules = []

    rule_sets = map(bytes_to_ruleset, input)
    for rule_set in rule_sets:
        # without the last 0 rule
        rules.extend(rule_set.rules[:-1])
    final_ruleset = RuleSet(rules=rules, classes=rc_combiner.classes_)
    # final_ruleset.prune_condition_map()
    return ruleset_to_bytes(final_ruleset)
