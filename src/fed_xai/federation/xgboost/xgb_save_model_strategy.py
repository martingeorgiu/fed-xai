from flwr.common import FitRes, Parameters, Scalar
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedXgbBagging


class XGBSaveModelStrategy(FedXgbBagging):
    def __init__(self, shouldSave: bool, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.shouldSave = shouldSave

    def aggregate_fit(
        self,
        server_round: int,
        results: list[tuple[ClientProxy, FitRes]],
        failures: list[tuple[ClientProxy, FitRes] | BaseException],
    ) -> tuple[Parameters | None, dict[str, Scalar]]:
        res = super().aggregate_fit(
            server_round=server_round,
            results=results,
            failures=failures,
        )
        if res[0] is not None:
            bytes_model = res[0].tensors[0]
            if self.shouldSave:
                with open(f"output/output{server_round}.bin", "wb") as file:
                    file.write(bytes_model)
        return res
