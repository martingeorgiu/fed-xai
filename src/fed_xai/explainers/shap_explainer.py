import shap
import xgboost

from fed_xai.data_loaders.loader import load_data


# This technique was not used eventually
def shap_explainer(model: xgboost.Booster) -> None:
    # This way we make sure that there are no labels in the DMatrix
    X_train, X_test, y_train, y_test = load_data(0, 1)
    test = xgboost.DMatrix(X_test)
    # train, test, num_train, num_test = load_data_for_xgb(0, 1, smote=False)

    names = model.feature_names
    explainer = shap.TreeExplainer(model, feature_names=names)
    explanation = explainer(test)

    # Global explanation
    shap.plots.beeswarm(explanation)

    selected_example = 0

    # print(test.get_data()[selected_example])
    # print(test.get_label()[selected_example])
    shap.plots.waterfall(explanation[selected_example])
