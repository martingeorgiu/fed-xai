from typing import Tuple

from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier, plot_importance

from fed_xai.data_loaders.loader import load_data_with_smote
from fed_xai.explainers.rulecosi_explainer import rulecosi_explainer
from fed_xai.explainers.shap_explainer import shap_explainer


def objective_train_xgboost(
    space, X_train, y_train, X_test, y_test
) -> Tuple[XGBClassifier, float]:

    clf = XGBClassifier(
        max_depth=int(space["max_depth"]),
        gamma=space["gamma"],
        learning_rate=space["learning_rate"],
        reg_alpha=space["reg_alpha"],
        reg_lambda=space["reg_lambda"],
        colsample_bytree=space["colsample_bytree"],
        min_child_weight=space["min_child_weight"],
        n_estimators=int(space["n_estimators"]),
        eval_metric="auc",
        early_stopping_rounds=10,
    )

    evaluation = [(X_train, y_train), (X_test, y_test)]

    clf.fit(
        X_train,
        y_train,
        eval_set=evaluation,
        verbose=False,
    )

    y_test_pred = clf.predict(X_test)
    # cannot use accuracy because of use of synthetic data -> THIS IS NOT CORRECT
    # https://www.kaggle.com/code/tanmay111999/diabetes-classification-xgb-lgbm-stack-smote?scriptVersionId=106483964&cellId=66
    accuracy = accuracy_score(y_test, y_test_pred)
    print("Accuracy:", accuracy)

    return (clf, accuracy)


selected_space = {
    "colsample_bytree": 0.6430856119765089,
    "gamma": 11.131971049496897,
    "learning_rate": 0.13217260031428005,
    "max_depth": 12.0,
    "min_child_weight": 1.1822174379587778,
    "n_estimators": 62.0,
    "reg_alpha": 8.701579711100049,
    "reg_lambda": 0.3148826988724287,
}


def main():
    X_train, X_test, y_train, y_test = load_data_with_smote(0, 1)
    clf, acc = objective_train_xgboost(selected_space, X_train, y_train, X_test, y_test)
    bst = clf.get_booster()

    shap_explainer(bst)

    # plot_importance(bst)
    # plt.show()

    # rulecosi_explainer(clf)


if __name__ == "__main__":
    main()
