import json
from logging import WARNING
from typing import Any, Callable, Optional, Union, cast

from flwr.common import EvaluateRes, FitRes, Parameters, Scalar
from flwr.common.logger import log
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedXgbBagging


class SaveModelStrategy(FedXgbBagging):
    def __init__(self, shouldSave: bool):
        self.shouldSave = shouldSave

    def aggregate_fit(
        self,
        server_round: int,
        results: list[tuple[ClientProxy, FitRes]],
        failures: list[Union[tuple[ClientProxy, FitRes], BaseException]],
    ) -> tuple[Optional[Parameters], dict[str, Scalar]]:
        res = super().aggregate_fit(
            server_round=server_round,
            results=results,
            failures=failures,
        )
        bytes_model = res[0].tensors[0]
        if self.shouldSave:
            with open(f"output/output{server_round}.bin", "wb") as file:
                file.write(bytes_model)
        return res
