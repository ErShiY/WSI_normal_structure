import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def compute_classification_metrics(y_true, y_pred):

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    metrics = {
        "acc": accuracy_score(y_true, y_pred),
        "precision": precision_score(
            y_true,
            y_pred,
            average="macro",
            zero_division=0
        ),
        "recall": recall_score(
            y_true,
            y_pred,
            average="macro",
            zero_division=0
        ),
        "macro_f1": f1_score(
            y_true,
            y_pred,
            average="macro",
            zero_division=0
        )
    }

    return metrics