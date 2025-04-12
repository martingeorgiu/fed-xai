import warnings
from logging import INFO

import pandas as pd
import xgboost as xgb
from datasets import Dataset
from flwr.common import log
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore", category=UserWarning, message="The currently tested dataset are")


def transform_lung_dataset_to_dmatrix(dataset: Dataset) -> xgb.DMatrix:
    """Transform dataset to DMatrix format for xgboost."""
    df = dataset.to_pandas()
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Expected a pandas DataFrame for 'df'.")

    df = df.drop_duplicates(ignore_index=True)
    log(INFO, f"Data shape: {df}")
    encoder = LabelEncoder()

    df["LUNG_CANCER"] = encoder.fit_transform(df["LUNG_CANCER"])
    df["GENDER"] = encoder.fit_transform(df["GENDER"])

    X = df.drop(["LUNG_CANCER"], axis=1)
    y = df["LUNG_CANCER"]
    for i in X.columns[2:]:
        temp = []
        for j in X[i]:
            temp.append(j - 1)
        X[i] = temp

    # Create DMatrix with features and label separated
    new_data = xgb.DMatrix(X, label=y)
    return new_data
