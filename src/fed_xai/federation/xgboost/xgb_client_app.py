from typing import Any

import xgboost as xgb
from flwr.client import Client
from flwr.common import Code, EvaluateIns, EvaluateRes, FitIns, FitRes, Parameters, Status
from flwr.common.context import Context
from sklearn.metrics import accuracy_score  # noqa: F401

from fed_xai.data_loaders.loader import load_data_for_xgb
from fed_xai.helpers.accuracy_score_with_threshold import accuracy_score_with_threshold
from fed_xai.xgboost.train_xgboost import selected_space

booster_params_from_hp = selected_space | {
    "objective": "binary:logistic",
    "tree_method": "hist",
    "eval_metric": "auc",
}


class XGBFlowerClient(Client):
    def __init__(
        self,
        train_dmatrix: xgb.DMatrix,
        valid_dmatrix: xgb.DMatrix,
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

    def _local_boost(self, bst_input: xgb.Booster) -> xgb.Booster:
        # Update trees based on local training data.
        for _ in range(self.num_local_round):
            bst_input.update(self.train_dmatrix, bst_input.num_boosted_rounds())

        # Bagging: extract the last N=num_local_round trees for sever aggregation
        bst = bst_input[
            bst_input.num_boosted_rounds() - self.num_local_round : bst_input.num_boosted_rounds()
        ]

        return bst

    def fit(self, ins: FitIns) -> FitRes:
        global_round = int(ins.config["global_round"])
        if global_round == 1:
            # First round local training
            bst = xgb.train(
                self.params,
                self.train_dmatrix,
                num_boost_round=self.num_local_round,
                evals=[(self.valid_dmatrix, "validate"), (self.train_dmatrix, "train")],
            )
        else:
            bst = xgb.Booster(params=self.params)
            global_model = bytearray(ins.parameters.tensors[0])

            # Load global model into booster
            bst.load_model(global_model)

            # Local training
            bst = self._local_boost(bst)

        # Save model
        local_model = bst.save_raw("json")
        local_model_bytes = bytes(local_model)

        return FitRes(
            status=Status(
                code=Code.OK,
                message="OK",
            ),
            parameters=Parameters(tensor_type="", tensors=[local_model_bytes]),
            num_examples=self.num_train,
            metrics={},
        )

    def evaluate(self, ins: EvaluateIns) -> EvaluateRes:
        # Load global model
        bst = xgb.Booster(params=self.params)
        para_b = bytearray(ins.parameters.tensors[0])
        bst.load_model(para_b)

        y_pred = bst.predict(self.valid_dmatrix, validate_features=False)
        y_true = self.valid_dmatrix.get_label()

        acc = accuracy_score_with_threshold(y_true, y_pred)
        # Run evaluation
        eval_results = bst.eval_set(
            evals=[(self.valid_dmatrix, "valid")],
            iteration=bst.num_boosted_rounds() - 1,
        )
        auc = round(float(eval_results.split("\t")[1].split(":")[1]), 4)

        return EvaluateRes(
            status=Status(
                code=Code.OK,
                message="OK",
            ),
            loss=0.0,
            num_examples=self.num_val,
            metrics={"acc": acc, "auc": auc},
        )


def xgb_client_fn(context: Context) -> XGBFlowerClient:
    # Load model and data
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    if not isinstance(partition_id, int) or not isinstance(num_partitions, int):
        raise TypeError("partition_id and num_partitions must be integers")
    train_dmatrix, valid_dmatrix, num_train, num_val = load_data_for_xgb(
        partition_id, num_partitions, smote=True
    )

    num_local_round = 3

    return XGBFlowerClient(
        train_dmatrix,
        valid_dmatrix,
        num_train,
        num_val,
        num_local_round,
        booster_params_from_hp,
    )
