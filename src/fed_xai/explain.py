import warnings

import numpy as np
import pandas as pd
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
from flwr_datasets import FederatedDataset
from matplotlib import pyplot as plt
from pandas import Series
from rulecosi import RuleCOSIClassifier, RuleSet
from rulecosi.rule_extraction import XGBClassifierExtractor

from fed_xai.data_loaders.loader import load_data, replace_keys
from fed_xai.explainers.bellatrex_explainer import bellatrex_explainer
from fed_xai.explainers.rulecosi_explainer import XGBClassifierExtractorForDebug
from fed_xai.explainers.shap_explainer import shap_explainer

# from fed_xai.xgb_classifier import XGBClassifierExtractor


def generate_viz(bst: xgb.Booster):
    fig, ax = plt.subplots(figsize=(30, 30))
    xgb.plot_tree(bst, ax=ax, tree_idx=1)
    print(xgb.build_info())
    print(xgb.config.get_config())
    # Save tree visualization
    plt.savefig("output/tree.pdf")

    # Create feature importance plot
    fig_importance, ax_importance = plt.subplots(figsize=(10, 10))
    xgb.plot_importance(bst, ax=ax_importance)
    plt.savefig("output/importance.pdf")
    bst.dump_model("output/dump.json")


# def generate_rules(bst: xgb.Booster):
#     print(f"Feature names: {bst.feature_names}")
#     dump_list = bst.get_dump(dump_format="json")
#     num_trees = len(dump_list)

#     xgb_classifier = xgb.XGBClassifier()
#     xgb_classifier._Booster = bst
#     xgb_classifier.n_estimators = num_trees

#     train_data, num_train, _, __ = load_data(0, num_clients=1)

#     y = train_data.get_label()
#     X = pd.DataFrame(train_data.get_data().toarray())

#     classifier = RuleCOSIClassifierDebug(
#         xgb_classifier, column_names=Series(bst.feature_names)
#     )
#     classifier.fit(X, y, np.array(bst.feature_names))
#     classifier.simplified_ruleset_.print_rules(heuristics_digits=4, condition_digits=1)


def main():
    with open("output/output4.bin", "rb") as file:
        data = file.read()
    bst = xgb.Booster(params={"objective": "binary:logistic"})
    para_b = bytearray(data)
    bst.load_model(para_b)

    # generate_viz(bst)
    # generate_rules(bst)
    # shap_explainer(bst)
    bellatrex_explainer(bst)


if __name__ == "__main__":
    main()
