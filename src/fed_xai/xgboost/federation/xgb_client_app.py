from typing import Any

import pandas as pd
import xgboost as xgb
from flwr.client import Client
from flwr.common import Code, EvaluateIns, EvaluateRes, FitIns, FitRes, Parameters, Scalar, Status
from flwr.common.context import Context
from rulecosi import RuleCOSIClassifier, RuleSet
from sklearn.base import check_array
from sklearn.metrics import accuracy_score, roc_auc_score  # noqa: F401

from fed_xai.data_loaders.loader import load_data_for_xgb
from fed_xai.helpers.accuracy_score_with_threshold import accuracy_score_with_threshold
from fed_xai.helpers.booster_to_classifier import booster_to_classifier, load_booster_from_bytes
from fed_xai.helpers.rulecosi_helpers import bytes_to_ruleset, ruleset_to_bytes
from fed_xai.xgboost.const import booster_params_from_hp, class_names


class XGBFlowerClient(Client):
    def __init__(
        self,
        client_id: int,
        train_dmatrix: xgb.DMatrix,
        valid_dmatrix: xgb.DMatrix,
        num_train: int,
        num_val: int,
        num_local_round: int,
        params: dict[str, Any],
    ) -> None:
        self.client_id = client_id
        self.train_dmatrix = train_dmatrix
        self.valid_dmatrix = valid_dmatrix
        self.num_train = num_train
        self.num_val = num_val
        self.num_local_round = num_local_round
        self.params = params

    def fit(self, ins: FitIns) -> FitRes:
        global_round = int(ins.config["global_round"])
        num_rounds = int(ins.config["num_rounds"])
        last_round_rulecosi = ins.config["last_round_rulecosi"] == "True"
        if last_round_rulecosi and global_round == num_rounds:
            return self._fit_rules(ins)

        return self._fit_bst(ins, global_round)

    def evaluate(self, ins: EvaluateIns) -> EvaluateRes:
        global_round = int(ins.config["global_round"])
        num_rounds = int(ins.config["num_rounds"])
        last_round_rulecosi = ins.config["last_round_rulecosi"] == "True"
        if last_round_rulecosi and global_round == num_rounds:
            return self._eval_rules(ins)
        return self._eval_bst(ins)

    def _local_boost(self, bst_input: xgb.Booster) -> xgb.Booster:
        # Update trees based on local training data.
        for _ in range(self.num_local_round):
            bst_input.update(self.train_dmatrix, bst_input.num_boosted_rounds())

        # Bagging: extract the last N=num_local_round trees for sever aggregation
        bst = bst_input[
            bst_input.num_boosted_rounds() - self.num_local_round : bst_input.num_boosted_rounds()
        ]

        return bst

    def _load_model(self, parameters: Parameters) -> xgb.Booster:
        return load_booster_from_bytes(self.params, parameters.tensors[0])

    def _fit_rules(self, ins: FitIns) -> FitRes:
        bst = self._load_model(ins.parameters)
        clf = booster_to_classifier(bst)
        X_train = pd.DataFrame(self.train_dmatrix.get_data().toarray())
        y_train = pd.Series(self.train_dmatrix.get_label())

        rc = RuleCOSIClassifier(
            base_ensemble=clf,
            metric="f1",
            random_state=0,
            column_names=class_names,
        )

        rc.fit(X_train, y_train)
        rules = ruleset_to_bytes(rc.simplified_ruleset_)
        return FitRes(
            status=Status(
                code=Code.OK,
                message="OK",
            ),
            parameters=Parameters(tensor_type="", tensors=[rules]),
            num_examples=self.num_train,
            metrics={},
        )

    def _fit_bst(self, ins: FitIns, global_round: int) -> FitRes:
        if global_round == 1:
            # First round local training
            bst = xgb.train(
                self.params,
                self.train_dmatrix,
                num_boost_round=self.num_local_round,
                evals=[(self.valid_dmatrix, "validate"), (self.train_dmatrix, "train")],
            )
        else:
            bst = self._load_model(ins.parameters)

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

    def _eval_rules(self, ins: EvaluateIns) -> EvaluateRes:
        X_test = pd.DataFrame(self.valid_dmatrix.get_data().toarray())
        y_test = self.valid_dmatrix.get_label()
        rules = bytes_to_ruleset(ins.parameters.tensors[0])
        X_test = check_array(X_test)
        y_pred = rules.predict(X_test)
        acc = accuracy_score_with_threshold(y_test, y_pred)
        auc = roc_auc_score(y_test, y_pred)

        global_eval_res = global_eval_rules(self.client_id, rules)

        return EvaluateRes(
            status=Status(
                code=Code.OK,
                message="OK",
            ),
            loss=0.0,
            num_examples=self.num_val,
            metrics={"acc": acc, "auc": auc} | global_eval_res,
        )

    def _eval_bst(self, ins: EvaluateIns) -> EvaluateRes:
        # Load global model
        bst = self._load_model(ins.parameters)

        y_pred = bst.predict(self.valid_dmatrix, validate_features=False)
        y_true = self.valid_dmatrix.get_label()

        acc = accuracy_score_with_threshold(y_true, y_pred)
        # Run evaluation
        eval_results = bst.eval_set(
            evals=[(self.valid_dmatrix, "valid")],
            iteration=bst.num_boosted_rounds() - 1,
        )
        auc = round(float(eval_results.split("\t")[1].split(":")[1]), 4)

        global_eval_res = global_eval_bst(self.client_id, bst)

        return EvaluateRes(
            status=Status(
                code=Code.OK,
                message="OK",
            ),
            loss=0.0,
            num_examples=self.num_val,
            metrics={"acc": acc, "auc": auc} | global_eval_res,
        )


def xgb_client_fn(context: Context, local_rounds: int) -> XGBFlowerClient:
    # Load model and data
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    if not isinstance(partition_id, int) or not isinstance(num_partitions, int):
        raise TypeError("partition_id and num_partitions must be integers")
    train_dmatrix, valid_dmatrix, num_train, num_val = load_data_for_xgb(
        partition_id, num_partitions, smote=True, withGlobal=False
    )

    return XGBFlowerClient(
        partition_id,
        train_dmatrix,
        valid_dmatrix,
        num_train,
        num_val,
        local_rounds,
        booster_params_from_hp,
    )


client_id_for_global_eval = 0


# Hacky temp solution for testing purposes. We want to calculate also the global accuracy and auc
# all data (this wouldn't be possible in real federated learning, but we want to have
# the comparison to model learned the classical way)
def global_eval_bst(client_id: int, bst: xgb.Booster) -> dict[str, Scalar]:
    # Only calculate it on one client
    if client_id != client_id_for_global_eval:
        return {}

    train_dmatrix, valid_dmatrix, num_train, num_val = load_data_for_xgb(0, 1, withGlobal=False)
    y_pred = bst.predict(valid_dmatrix, validate_features=False)
    y_true = valid_dmatrix.get_label()
    return {
        "acc_global": accuracy_score_with_threshold(y_true, y_pred),
        "auc_global": roc_auc_score(y_true, y_pred),
    }


def global_eval_rules(client_id: int, rules: RuleSet) -> dict[str, Scalar]:
    # Only calculate it on one client
    if client_id != client_id_for_global_eval:
        return {}

    train_dmatrix, valid_dmatrix, num_train, num_val = load_data_for_xgb(0, 1, withGlobal=False)
    X_test = pd.DataFrame(valid_dmatrix.get_data().toarray())
    y_test = valid_dmatrix.get_label()
    X_test = check_array(X_test)
    y_pred = rules.predict(X_test)

    return {
        "acc_global": accuracy_score_with_threshold(y_test, y_pred),
        "auc_global": roc_auc_score(y_test, y_pred),
    }
