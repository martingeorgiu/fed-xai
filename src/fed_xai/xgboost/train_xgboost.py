from typing import Any

from pandas import DataFrame, Series
from sklearn.metrics import accuracy_score, roc_auc_score
from xgboost import DMatrix, XGBClassifier

from fed_xai.data_loaders.loader import load_data_with_smote
from fed_xai.explainers.combining_rulecosi_explainer import combining_rulecosi_explainer
from fed_xai.helpers.number_of_trees import get_number_of_trees
from fed_xai.xgboost.const import booster_params_from_hp


def objective_train_xgboost(
    space: dict[str, Any],
    X_train: DataFrame,
    X_test: DataFrame,
    y_train: Series,
    y_test: Series,
) -> tuple[XGBClassifier, float, float]:
    clf = XGBClassifier(
        max_depth=int(space["max_depth"]),
        gamma=space["gamma"],
        learning_rate=space["learning_rate"],
        reg_alpha=space["reg_alpha"],
        reg_lambda=space["reg_lambda"],
        colsample_bytree=space["colsample_bytree"],
        min_child_weight=space["min_child_weight"],
        n_estimators=int(space["n_estimators"]),
        tree_method=space["tree_method"],
        eval_metric=space["eval_metric"],
        early_stopping_rounds=int(space["early_stopping_rounds"]),
    )

    evaluation = [(X_train, y_train), (X_test, y_test)]

    clf.fit(
        X_train,
        y_train,
        eval_set=evaluation,
        verbose=False,
    )

    y_test_pred = clf.predict(X_test)

    accuracy = accuracy_score(y_test, y_test_pred)
    X_test_matrix = DMatrix(X_test, label=y_test)
    y_test_pred = clf.get_booster().predict(X_test_matrix)
    # the value from clf.predict() is thresholded at 0.5, but we don't want that for roc_auc_score
    auc = roc_auc_score(y_test, y_test_pred)
    print("Accuracy:", accuracy)
    print("AUC:", auc)

    return (clf, accuracy, auc)


def main() -> None:
    X_train, X_test, y_train, y_test = load_data_with_smote(0, 1)
    clf, acc, auc = objective_train_xgboost(
        booster_params_from_hp, X_train, X_test, y_train, y_test
    )
    no_trees = get_number_of_trees(clf.get_booster())
    print(f"No trees: {no_trees}")
    # bst = clf.get_booster()  # noqa: F841

    # shap_explainer(bst)

    # plot_importance(bst)
    # plt.show()

    # rulecosi_explainer(clf)
    combining_rulecosi_explainer(clf)


if __name__ == "__main__":
    main()
