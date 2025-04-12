from typing import Any

from pandas import DataFrame, Series
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier

from fed_xai.data_loaders.loader import load_data_with_smote


def objective_train_xgboost(
    space: dict[str, Any],
    X_train: DataFrame,
    X_test: DataFrame,
    y_train: Series,
    y_test: Series,
) -> tuple[XGBClassifier, float]:
    clf = XGBClassifier(
        max_depth=int(space["max_depth"]),
        gamma=space["gamma"],
        learning_rate=space["learning_rate"],
        reg_alpha=space["reg_alpha"],
        reg_lambda=space["reg_lambda"],
        colsample_bytree=space["colsample_bytree"],
        min_child_weight=space["min_child_weight"],
        n_estimators=int(space["n_estimators"]),
        tree_method="hist",
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


# The best hyperparameters found by hyperopt
selected_space = {
    "colsample_bytree": 0.6430856119765089,
    "gamma": 11.131971049496897,
    "learning_rate": 0.13217260031428005,
    "max_depth": 12,
    "min_child_weight": 1.1822174379587778,
    "n_estimators": 62,
    "reg_alpha": 8.701579711100049,
    "reg_lambda": 0.3148826988724287,
}


def main() -> None:
    X_train, X_test, y_train, y_test = load_data_with_smote(0, 1)
    clf, acc = objective_train_xgboost(selected_space, X_train, X_test, y_train, y_test)
    bst = clf.get_booster()  # noqa: F841

    # shap_explainer(bst)

    # plot_importance(bst)
    # plt.show()

    # rulecosi_explainer(clf)


if __name__ == "__main__":
    main()
