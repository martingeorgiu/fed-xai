import warnings
from logging import INFO

import numpy as np
import xgboost as xgb
from datasets import Dataset
from flwr.common import log
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import IidPartitioner
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings(
    "ignore", category=UserWarning, message="The currently tested dataset are"
)


def transform_diabetes_dataset_to_dmatrix(dataset: Dataset):
    dataset.set_format("numpy")
    df = dataset.to_pandas()

    X_data = df.drop("Outcome", axis=1).drop("ID", axis=1)
    log(INFO, f"X_data: {X_data.shape[0]}")
    y = df.Outcome
    new_data = xgb.DMatrix(X_data, label=y)
    return new_data
