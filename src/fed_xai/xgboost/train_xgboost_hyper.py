from typing import Any

from hyperopt import STATUS_OK, Trials, fmin, hp, tpe

from fed_xai.data_loaders.loader import load_data_with_smote
from fed_xai.xgboost.train_xgboost import objective_train_xgboost

# tuning here
# https://www.kaggle.com/code/prashant111/a-guide-on-xgboost-hyperparameters-tuning/notebook#4.-Bayesian-Optimization-with-HYPEROPT-
space = {
    "max_depth": hp.quniform("max_depth", 3, 18, 1),
    "gamma": hp.uniform("gamma", 0, 20),
    "learning_rate": hp.uniform("learning_rate", 0.01, 0.2),
    "reg_alpha": hp.uniform("reg_alpha", 0, 10),
    "reg_lambda": hp.uniform("reg_lambda", 0, 2),
    "colsample_bytree": hp.uniform("colsample_bytree", 0, 1),
    "min_child_weight": hp.uniform("min_child_weight", 0, 10),
    "n_estimators": hp.quniform("n_estimators", 50, 200, 1),
    "seed": 0,
}


def main() -> None:
    X_train, X_test, y_train, y_test = load_data_with_smote(0, 1)

    def objective(space: dict[str, Any]) -> dict[str, Any]:
        clf, accuracy = objective_train_xgboost(space, X_train, X_test, y_train, y_test)
        return {"loss": -accuracy, "status": STATUS_OK}

    trials = Trials()

    best_hyperparams = fmin(
        fn=objective, space=space, algo=tpe.suggest, max_evals=2000, trials=trials
    )
    print("The best hyperparameters are : ", "\n")
    print(best_hyperparams)


if __name__ == "__main__":
    main()
