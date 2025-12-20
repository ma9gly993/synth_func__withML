import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

def get_f1_average(y):
    unique_labels = len(np.unique(y))
    return "binary" if unique_labels == 2 else "macro"

def get_pos_label(y):
    unique_labels = np.unique(y)
    if len(unique_labels) == 2:
        if 1 in unique_labels:
            return 1
        else:
            return unique_labels[1]
    return None

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

def calculate_metrics(y_test, y_pred, y_proba=None):
    metrics = {}

    metrics["accuracy"] = accuracy_score(y_test, y_pred)

    if len(np.unique(y_test)) == 2:
        metrics["f1"] = f1_score(y_test, y_pred, average="binary")
        metrics["precision"] = precision_score(y_test, y_pred, average="binary", zero_division=0)
        metrics["recall"] = recall_score(y_test, y_pred, average="binary", zero_division=0)

        if y_proba is not None:
            metrics["roc_auc"] = roc_auc_score(y_test, y_proba)
        else:
            metrics["roc_auc"] = np.nan
    else:
        metrics["f1"] = f1_score(y_test, y_pred, average="macro")
        metrics["precision"] = precision_score(y_test, y_pred, average="macro", zero_division=0)
        metrics["recall"] = recall_score(y_test, y_pred, average="macro", zero_division=0)
        metrics["roc_auc"] = np.nan

    return metrics

