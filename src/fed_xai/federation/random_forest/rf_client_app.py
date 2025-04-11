import warnings
from typing import List

import numpy as np
import xgboost as xgb
from flwr.client import Client, ClientApp
from flwr.common import (
    Code,
    EvaluateIns,
    EvaluateRes,
    FitIns,
    FitRes,
    Parameters,
    Status,
)
from flwr.common.config import unflatten_dict
from flwr.common.context import Context
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
)

from fed_xai.data_loaders import load_data, replace_keys


def get_params(model: RandomForestClassifier) -> List[np.ndarray]:
    params = [
        model.n_estimators,
        model.max_depth,
        model.min_samples_split,
        model.min_samples_leaf,
    ]
    return params


# Set the parameters in the RandomForestClassifier
def set_params(
    model: RandomForestClassifier, params: List[np.ndarray]
) -> RandomForestClassifier:
    model.n_estimators = int(params[0])
    model.max_depth = int(params[1])
    model.min_samples_split = int(params[2])
    model.min_samples_leaf = int(params[3])
    return model


class RFFlowerClient(Client):
    def __init__(
        self,
        train_dmatrix,
        valid_dmatrix,
        num_train,
        num_val,
        num_local_round,
        params,
    ):
        self.train_dmatrix = train_dmatrix
        self.valid_dmatrix = valid_dmatrix
        self.num_train = num_train
        self.num_val = num_val
        self.num_local_round = num_local_round
        self.params = params

    # Train the local model, return the model parameters to the server
    def fit(self, ins: FitIns) -> FitRes:
        global_round = int(ins.config["global_round"])
        model = RandomForestClassifier()
        if global_round == 1:
            model.set_params(ins.parameters.tensors[0])
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
        model = RandomForestClassifier()
        model.set_params(ins.parameters.tensors[0])

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


def client_fn(context: Context):
    # Load model and data
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    train_dmatrix, valid_dmatrix, num_train, num_val = load_data(
        partition_id, num_partitions
    )

    cfg = replace_keys(unflatten_dict(context.run_config))
    num_local_round = cfg["local_epochs"]

    return RFFlowerClient(
        train_dmatrix,
        valid_dmatrix,
        num_train,
        num_val,
        num_local_round,
        cfg["params"],
    )


app = ClientApp(
    client_fn,
)
