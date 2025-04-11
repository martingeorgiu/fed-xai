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


def train_test_split(partition: Dataset, test_fraction, seed):
    """Split the data into train and validation set given split rate."""
    train_test = partition.train_test_split(test_size=test_fraction, seed=seed)
    partition_train = train_test["train"]
    partition_test = train_test["test"]

    num_train = len(partition_train)
    num_test = len(partition_test)

    return partition_train, partition_test, num_train, num_test


def transform_diabetes_dataset_to_dmatrix(dataset: Dataset):
    dataset.set_format("numpy")
    df = dataset.to_pandas()

    X_data = df.drop("Outcome", axis=1)
    log(INFO, f"X_data: {X_data.shape[0]}")
    y = df.Outcome
    new_data = xgb.DMatrix(X_data, label=y)
    return new_data


def transform_lung_dataset_to_dmatrix(dataset: Dataset):
    """Transform dataset to DMatrix format for xgboost."""
    df = dataset.to_pandas()
    # TODO: Duplicate can still occour across partitions and accross train and test!
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


def transform_higgs_dataset_to_dmatrix(data):
    """Transform dataset to DMatrix format for xgboost."""
    x = data["inputs"]
    y = data["label"]
    new_data = xgb.DMatrix(x, label=y)
    return new_data


def transform_dataset_to_dmatrix(data):
    return transform_diabetes_dataset_to_dmatrix(data)


fds = None  # Cache FederatedDataset


def load_data(partition_id, num_clients):
    print("Loading data...")
    # Only initialize `FederatedDataset` once
    global fds
    if fds is None:
        # partitioner = IidPartitioner(num_partitions=num_clients)
        # fds = FederatedDataset(
        #     # dataset="jxie/higgs",
        #     dataset="virtual10/survey_lung_cancer",
        #     partitioners={"train": partitioner},
        # )
        fds = FederatedDataset(
            dataset="Genius-Society/Pima",
            partitioners={
                "train": num_clients,
                "validation": num_clients,
                "test": num_clients,
            },
        )

    # Load the partition for this `partition_id`
    train_partition = fds.load_partition(partition_id, split="train")
    num_train = len(train_partition)
    val_partition = fds.load_partition(partition_id, split="validation")
    num_val = len(val_partition)
    log(
        INFO,
        f"Loading data for partition {partition_id}...num train: {num_train}...num val: {num_val}",
    )

    # Reformat data to DMatrix for xgboost
    log(INFO, "Reformatting data...")
    train_dmatrix = transform_dataset_to_dmatrix(train_partition)
    valid_dmatrix = transform_dataset_to_dmatrix(val_partition)

    return train_dmatrix, valid_dmatrix, num_train, num_val


# Test the function
if __name__ == "__main__":
    partition_id = 0
    num_clients = 1
    train_dmatrix, valid_dmatrix, num_train, num_val = load_data(
        partition_id, num_clients
    )
    print(f"Train DMatrix: {train_dmatrix}")
    print(f"Validation DMatrix: {valid_dmatrix}")
    print(f"Number of training samples: {num_train}")
    print(f"Number of validation samples: {num_val}")


def replace_keys(input_dict, match="-", target="_"):
    """Recursively replace match string with target string in dictionary keys."""
    new_dict = {}
    for key, value in input_dict.items():
        new_key = key.replace(match, target)
        if isinstance(value, dict):
            new_dict[new_key] = replace_keys(value, match, target)
        else:
            new_dict[new_key] = value
    return new_dict
