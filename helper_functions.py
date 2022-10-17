import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> tuple:
    """
    :param y_true: true labels provided by the dataset.
    :param y_pred: predicted labels provided by the model.
    :return: accuracy, f1 score, precision, recall and roc auc score.
    """
    accuracy = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    roc_auc = roc_auc_score(y_true, y_pred)
    return accuracy, f1, precision, recall, roc_auc

def calculate_metrics_multiclass(y_true: np.ndarray, y_pred: np.ndarray, y_pred_proba) -> tuple:
    """
    :param y_true: true labels provided by the dataset.
    :param y_pred: predicted labels provided by the model.
    :param y_pred_proba: predicted probabilities for labels.
    :return: accuracy, f1 score, precision, recall and roc auc score.
    """
    accuracy = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='micro')
    precision = precision_score(y_true, y_pred, average='micro')
    recall = recall_score(y_true, y_pred, average='micro')
    roc_auc = roc_auc_score(y_true, y_pred_proba, multi_class='ovr')
    return accuracy, f1, precision, recall, roc_auc
