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

booster_params_from_hp = selected_space | {
    "objective": "binary:logistic",
    "tree_method": "hist",
    "eval_metric": "auc",
}
