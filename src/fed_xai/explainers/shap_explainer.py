import numpy as np
import shap
import xgboost

from fed_xai.data_loaders import load_data


def shap_explainer(model: xgboost.Booster):
    train_dmatrix, valid_dmatrix, num_train, num_val = load_data(0, 5)
    names = model.feature_names
    Xd = xgboost.DMatrix(
        train_dmatrix.get_data(),
        label=train_dmatrix.get_label(),
        feature_names=names,
    )

    print(names)
    pred = model.predict(Xd, output_margin=True)

    explainer = shap.TreeExplainer(model, feature_names=names)
    explanation = explainer(Xd)
    print(explanation)
    shap_values = explanation.values

    np.abs(shap_values.sum(axis=1) + explanation.base_values - pred).max()
    shap.plots.beeswarm(explanation)
