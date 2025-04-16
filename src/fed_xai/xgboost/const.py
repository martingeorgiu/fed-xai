# The best hyperparameters found by hyperopt
selected_space1 = {
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

# New space with calculated early_stopping_rounds
selected_space2 = {
    "colsample_bytree": 0.35481936367348016,
    "early_stopping_rounds": 25,
    "gamma": 0.62391318284083,
    "learning_rate": 0.18558469825365817,
    "max_depth": 6,
    "min_child_weight": 8.124622468374259,
    "n_estimators": 192,
    "reg_alpha": 7.331135046167648,
    "reg_lambda": 1.8590288099666696,
}

# small number of estimators
selected_space3 = {
    "colsample_bytree": 0.6050922156776783,
    "early_stopping_rounds": 26,
    "gamma": 15.3386409963142,
    "learning_rate": 0.1997795717301235,
    "max_depth": 5,
    "min_child_weight": 5.095461516517645,
    "n_estimators": 10,
    "reg_alpha": 8.654995022392628,
    "reg_lambda": 0.7188640910421842,
}

base_params = {
    "seed": 0,
    "tree_method": "hist",
    "eval_metric": "auc",
}

booster_params_from_hp = (
    selected_space1
    | base_params
    | {
        "objective": "binary:logistic",
    }
)
