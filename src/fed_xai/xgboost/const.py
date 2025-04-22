# The best hyperparameters found by hyperopt
import pandas as pd

selected_space = {
    "colsample_bytree": 0.79913320560587,
    "early_stopping_rounds": 11,
    "gamma": 10.552380569366205,
    "learning_rate": 0.1642341446308446,
    "max_depth": 7,
    "min_child_weight": 2.2233543333103,
    "n_estimators": 15,
    "reg_alpha": 5.639307424170162,
    "reg_lambda": 0.6644100479124082,
}

selected_space_old = {
    "colsample_bytree": 0.6430856119765089,
    "gamma": 11.131971049496897,
    "learning_rate": 0.13217260031428005,
    "max_depth": 12,
    "min_child_weight": 1.1822174379587778,
    "n_estimators": 62,
    "reg_alpha": 8.701579711100049,
    "reg_lambda": 0.3148826988724287,
    "early_stopping_rounds": 10,
}


base_params = {
    "seed": 0,
    "tree_method": "hist",
    "eval_metric": "auc",
}

booster_params_from_hp = (
    selected_space
    | base_params
    | {
        "objective": "binary:logistic",
    }
)

rules_suffix = "rules"
class_names = pd.Series(
    [
        "Pregnancies",
        "Glucose",
        "BloodPressure",
        "SkinThickness",
        "Insulin",
        "BMI",
        "DiabetesPedigreeFunction",
        "Age",
    ]
)
