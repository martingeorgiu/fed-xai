from typing import Any

from hyperopt import STATUS_OK, Trials, fmin, hp, tpe

from fed_xai.data_loaders.loader import fds_one, load_data_with_smote
from fed_xai.xgboost.const import base_params
from fed_xai.xgboost.standard.train_xgboost import objective_train_xgboost

space = {
    "max_depth": hp.quniform("max_depth", 3, 18, 1),
    "gamma": hp.uniform("gamma", 0, 20),
    "learning_rate": hp.uniform("learning_rate", 0.01, 0.2),
    "reg_alpha": hp.uniform("reg_alpha", 0, 10),
    "reg_lambda": hp.uniform("reg_lambda", 0, 2),
    "colsample_bytree": hp.uniform("colsample_bytree", 0, 1),
    "min_child_weight": hp.uniform("min_child_weight", 0, 10),
    "n_estimators": hp.quniform("n_estimators", 5, 80, 1),
    "early_stopping_rounds": hp.quniform("early_stopping_rounds", 10, 20, 1),
    "random_state": hp.randint("random_state", 0, 10000),
} | base_params


# Made for researching which hyperparameters are the best
def main() -> None:
    def objective(space: dict[str, Any]) -> dict[str, Any]:
        random_state = int(space["random_state"])

        global fds_one
        if fds_one is not None and fds_one._dataset is not None:
            # Shuffle the dataset to ensure randomness according to the random_state
            fds_one._dataset = fds_one._dataset.shuffle(seed=random_state)

        X_train, X_test, y_train, y_test = load_data_with_smote(0, 1, random_state=random_state)
        clf, accuracy, auc = objective_train_xgboost(space, X_train, X_test, y_train, y_test)
        return {"loss": -auc, "status": STATUS_OK}

    trials = Trials()

    best_hyperparams = fmin(
        fn=objective, space=space, algo=tpe.suggest, max_evals=2000, trials=trials
    )
    print("The best hyperparameters are : ", "\n")
    print(best_hyperparams)


if __name__ == "__main__":
    main()
