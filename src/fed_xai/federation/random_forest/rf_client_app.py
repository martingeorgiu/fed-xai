from typing import Any

import numpy as np
from flwr.client import Client, ClientApp
from flwr.common import Code, EvaluateIns, EvaluateRes, FitIns, FitRes, Parameters, Status
from flwr.common.context import Context
from pandas import DataFrame
from sklearn.ensemble import RandomForestClassifier

from fed_xai.data_loaders.loader import load_data

# This technique was not used eventually


def get_params(model: RandomForestClassifier) -> list[np.ndarray]:
    params = [
        model.n_estimators,
        model.max_depth,
        model.min_samples_split,
        model.min_samples_leaf,
    ]
    return params


# Set the parameters in the RandomForestClassifier
def set_params(model: RandomForestClassifier, params: list[np.ndarray]) -> RandomForestClassifier:
    model.n_estimators = int(params[0])
    model.max_depth = int(params[1])
    model.min_samples_split = int(params[2])
    model.min_samples_leaf = int(params[3])
    return model


class RFFlowerClient(Client):
    def __init__(
        self,
        train_dmatrix: DataFrame,
        valid_dmatrix: DataFrame,
        num_train: int,
        num_val: int,
        num_local_round: int,
        params: dict[str, Any],
    ) -> None:
        self.train_dmatrix = train_dmatrix
        self.valid_dmatrix = valid_dmatrix
        self.num_train = num_train
        self.num_val = num_val
        self.num_local_round = num_local_round
        self.params = params

    # Train the local model, return the model parameters to the server
    def fit(self, ins: FitIns) -> FitRes:
        # global_round = int(ins.config["global_round"])
        model = RandomForestClassifier()
        # if global_round == 1:
        # model.set_params(ins.parameters.tensors[0])
        model.fit(self.train_dmatrix.get_data(), self.train_dmatrix.get_label())

        return FitRes(
            status=Status(
                code=Code.OK,
                message="OK",
            ),
            parameters=Parameters(tensor_type="", tensors=[model.get_params()]),
            num_examples=self.num_train,
            metrics={},
        )

    def evaluate(self, ins: EvaluateIns) -> EvaluateRes:
        # Load global model
        # model = RandomForestClassifier()
        # model.set_params(ins.parameters.tensors[0])

        # Run evaluation
        # model.predict(self.valid_dmatrix.get_data())
        # eval_results = model.eval_set(
        #     evals=[(self.valid_dmatrix, "valid")],
        #     iteration=bst.num_boosted_rounds() - 1,
        # )
        # auc = round(float(eval_results.split("\t")[1].split(":")[1]), 4)

        return EvaluateRes(
            status=Status(
                code=Code.OK,
                message="OK",
            ),
            loss=0.0,
            num_examples=self.num_val,
            metrics={"AUC": 0.9},
        )


def client_fn(context: Context) -> RFFlowerClient:
    # Load model and data
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    if not isinstance(partition_id, int) or not isinstance(num_partitions, int):
        raise TypeError("partition_id and num_partitions must be integers")
    train_dmatrix, valid_dmatrix, num_train, num_val = load_data(partition_id, num_partitions)

    num_local_round = 10

    return RFFlowerClient(
        train_dmatrix,
        valid_dmatrix,
        0,
        0,
        # num_train,
        # num_val,
        num_local_round,
        {},
    )


app = ClientApp(
    client_fn,
)
