# Explainability of federatively trained machine learning models

## Getting started

1. Make sure you have [Poetry](https://python-poetry.org/) installed
2. Run `poetry install` in the root of the project
3. Activate virtual environment created by Poetry using: `eval $(poetry env activate)`

## Running the training

To run a classic training (without any federation): `python src/fed_xai/xgboost/standard/train_xgboost.py`
To run a federated training: `python src/fed_xai/xgboost/federation/run_xgb_simulation.py`
To run a benchmark run (many federated trainings with different parameters and seeds): `python src/fed_xai/xgboost/federation/run_benchmark.py`


## References

Dataset used in this repo: https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database/data
