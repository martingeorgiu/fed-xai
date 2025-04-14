import numpy as np
from sklearn.metrics import accuracy_score


def accuracy_score_with_threshold(
    y_true: np.ndarray, y_pred: np.ndarray, threshold: float = 0.5
) -> float:
    """
    Calculate the accuracy score with a threshold for binary classification.

    Parameters
    ----------
    y_true : np.ndarray
        True labels.
    y_pred : np.ndarray
        Predicted probabilities.

    Returns
    -------
    float
        Accuracy score.
    """
    y_pred_binary = (y_pred >= threshold).astype(int)
    return accuracy_score(y_true, y_pred_binary)
