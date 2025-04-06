import warnings

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
from matplotlib import pyplot as plt
from pandas import Series
from rulecosi import RuleCOSIClassifier, RuleSet
from rulecosi.rule_extraction import XGBClassifierExtractor

from fed_xai.task import load_data, replace_keys
from fed_xai.xgb_classifier import XGBClassifierExtractorForDebug

# from fed_xai.xgb_classifier import XGBClassifierExtractor

if __name__ == "__main__":
    with open("output/output1.bin", "rb") as file:
        data = file.read()
    bst = xgb.Booster(params={"objective": "binary:logistic"})
    para_b = bytearray(data)
    bst.load_model(para_b)

    # fig, ax = plt.subplots(figsize=(30, 30))
    # xgb.plot_tree(bst, ax=ax, tree_idx=1)
    # print(xgb.build_info())
    # print(xgb.config.get_config())
    # # Save tree visualization
    # plt.savefig("output/plt/tree.pdf")

    # Create feature importance plot
    # fig_importance, ax_importance = plt.subplots(figsize=(10, 10))
    # xgb.plot_importance(bst, ax=ax_importance)
    # plt.savefig("output/plt/importance.pdf")
    # bst.dump_model("output/plt/dump.json")

    # rc = RuleCOSIClassifier(base_ensemble=bst,
    #     metric='f1',n_estimators=100, tree_max_depth=3,
    #     conf_threshold=0.9, cov_threshold=0.0,
    #     random_state=1212, column_names=X_train.columns)
    # rc.fit
    print(f"Feature names: {bst.feature_names}")
    dump_list = bst.get_dump(dump_format="json")
    num_trees = len(dump_list)

    xgb_classifier = xgb.XGBClassifier()
    xgb_classifier._Booster = bst
    xgb_classifier.n_estimators = num_trees

    print(f"Number of trees: {num_trees}")
    print(f"Number of estimators: {xgb_classifier.n_estimators}")

    classifier = RuleCOSIClassifier(xgb_classifier)
    classifier
    # classifier.fit(None, None)
    # extractor = XGBClassifierExtractor(
    extractor = XGBClassifierExtractorForDebug(
        xgb_classifier,
        Series(bst.feature_names),
        np.array(bst.feature_names),
        None,
        None,
        -1e-6,
    )
    rules = extractor.extract_rules()
    print(rules)

    # rule_extractor = XGBClassifierExtractor(
    #     bst,
    #     bst.feature_names,
    #     bst.feature_names,
    #     None,
    #     None,
    #     1e-6,
    # )
    # rules = rule_extractor.extract_rules()
    # print(rules)
