import numpy as np
import shap
import xgboost

from fed_xai.data_loaders.loader import load_data, load_data_for_xgb


def shap_explainer(model: xgboost.Booster):
    train, test, num_train, num_test = load_data_for_xgb(0, 1, smote=False)
    names = model.feature_names

    explainer = shap.TreeExplainer(model, feature_names=names)
    explanation = explainer(test)

    shap.plots.beeswarm(explanation)
