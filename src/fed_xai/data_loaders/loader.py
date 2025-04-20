import gc
import warnings

import pandas as pd
import xgboost as xgb
from datasets import Dataset
from flwr_datasets import FederatedDataset
from flwr_datasets.preprocessor import Merger
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore", category=UserWarning, message="The currently tested dataset are")


fds = None  # Cache FederatedDataset
fds_one = None  # Cache FederatedDataset


def load_data_for_xgb(
    partition_id: int,
    num_clients: int,
    smote: bool = False,
    withGlobal: bool = True,
    random_state: int | None = 42,
) -> tuple[xgb.DMatrix, xgb.DMatrix, int, int]:
    if smote:
        X_train, X_test, y_train, y_test = load_data_with_smote(
            partition_id, num_clients, withGlobal, random_state
        )
    else:
        X_train, X_test, y_train, y_test = load_data(
            partition_id, num_clients, withGlobal, random_state
        )

    feature_names = list(X_train)

    return (
        xgb.DMatrix(X_train, label=y_train, feature_names=feature_names),
        xgb.DMatrix(X_test, label=y_test, feature_names=feature_names),
        len(X_train.index),
        len(X_test.index),
    )


def load_data_with_smote(
    partition_id: int,
    num_clients: int,
    withGlobal: bool = True,
    random_state: int | None = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    X_train, X_test, y_train, y_test = load_data(
        partition_id, num_clients, withGlobal, random_state
    )
    over = SMOTE(random_state=random_state)
    # We first split the data into train and test sets
    # and then oversample them, so there are no overlaps
    X_train, y_train = over.fit_resample(X_train, y_train)
    return (X_train, X_test, y_train, y_test)


def load_data(
    partition_id: int,
    num_clients: int,
    withGlobal: bool = True,
    random_state: int | None = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    print("Loading data...")

    def get_dataset() -> Dataset:
        # This speeds up the loading of the dataset for global evaluation
        if partition_id == 0 and num_clients == 1:
            print("\n\nLoading dataset for global evaluation...")
            global fds_one
            if fds_one is None:
                print("\n\nLoading dataset for the first time only global evaluation...")
                fds_one = FederatedDataset(
                    dataset="Genius-Society/Pima",
                    preprocessor=Merger(merge_config={"main": ("train", "validation", "test")}),
                    seed=random_state,
                    partitioners={
                        "main": num_clients,
                    },
                )
            return fds_one.load_partition(partition_id, split="main")

        if withGlobal:
            # Only initialize `FederatedDataset` once
            global fds

        # If withGlobal is False, we don't want to use the global variable and reassign every time
        if fds is None or not withGlobal:
            print("\n\nLoading dataset for local evaluation...")
            fds = FederatedDataset(
                dataset="Genius-Society/Pima",
                preprocessor=Merger(merge_config={"main": ("train", "validation", "test")}),
                seed=random_state,
                partitioners={
                    "main": num_clients,
                },
            )

        return fds.load_partition(partition_id, split="main")

    dataset = get_dataset()
    dataset.set_format("numpy")
    df = dataset.to_pandas()
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Expected a pandas DataFrame for 'df'.")

    X = df.drop("Outcome", axis=1).drop("ID", axis=1)
    y = df.Outcome

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=1 / 5, random_state=random_state, stratify=y
    )

    return (X_train, X_test, y_train, y_test)


def clear_dataset_cache() -> None:
    """Drop the cached FederatedDataset so that the next load_data call
    will re‑instantiate it."""
    global fds
    global fds_one
    fds = None
    fds_one = None
    gc.collect()


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
