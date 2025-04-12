from flwr.client import ClientApp
from flwr.server import ServerApp
from flwr.simulation import run_simulation

from fed_xai.federation.xgboost.xgb_client_app import xgb_client_fn
from fed_xai.federation.xgboost.xgb_server_app import xgb_server_fn


def main() -> None:
    client_app = ClientApp(
        xgb_client_fn,
    )
    server_app = ServerApp(
        server_fn=xgb_server_fn,
    )

    run_simulation(
        server_app=server_app,
        client_app=client_app,
        num_supernodes=3,
    )


if __name__ == "__main__":
    main()
