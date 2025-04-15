import warnings

import pandas as pd
import xgboost as xgb
from flwr_datasets import FederatedDataset
from flwr_datasets.preprocessor import Merger
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore", category=UserWarning, message="The currently tested dataset are")


fds = None  # Cache FederatedDataset


def load_data_for_xgb(
    partition_id: int,
    num_clients: int,
    smote: bool = False,
) -> tuple[xgb.DMatrix, xgb.DMatrix, int, int]:
    if smote:
        X_train, X_test, y_train, y_test = load_data_with_smote(partition_id, num_clients)
    else:
        X_train, X_test, y_train, y_test = load_data(partition_id, num_clients)

    feature_names = list(X_train)

    return (
        xgb.DMatrix(X_train, label=y_train, feature_names=feature_names),
        xgb.DMatrix(X_test, label=y_test, feature_names=feature_names),
        len(X_train.index),
        len(X_test.index),
    )


def load_data_with_smote(
    partition_id: int, num_clients: int, withGlobal: bool = True
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    X_train, X_test, y_train, y_test = load_data(partition_id, num_clients, withGlobal)
    over = SMOTE(random_state=0)
    # We first split the data into train and test sets
    # and then oversample them, so there are no overlaps
    X_train, y_train = over.fit_resample(X_train, y_train)
    return (X_train, X_test, y_train, y_test)


def load_data(
    partition_id: int, num_clients: int, withGlobal: bool = True
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    print("Loading data...")

    if withGlobal:
        # Only initialize `FederatedDataset` once
        global fds

    # If withGlobal is False, we don't want to use the global variable and reassign every time
    if fds is None or not withGlobal:
        fds = FederatedDataset(
            dataset="Genius-Society/Pima",
            preprocessor=Merger(merge_config={"main": ("train", "validation", "test")}),
            partitioners={
                "main": num_clients,
            },
        )

    # Load the partition for this `partition_id`
    dataset = fds.load_partition(partition_id, split="main")
    dataset.set_format("numpy")
    df = dataset.to_pandas()
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Expected a pandas DataFrame for 'df'.")

    X = df.drop("Outcome", axis=1).drop("ID", axis=1)
    y = df.Outcome

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=1 / 5, random_state=42, stratify=y
    )

    return (X_train, X_test, y_train, y_test)


def main() -> None:
    partition_id = 0
    num_clients = 1
    X_train, X_test, y_train, y_test = load_data(partition_id, num_clients)
    print(list(X_train))
    print("Validation DMatrix: ")
    print(X_test.head())
    print(y_test)
    print(f"Number of training samples: {len(X_train.index)}")
    print(f"Number of validation samples: {len(X_test.index)}")


# Test the function
if __name__ == "__main__":
    main()
