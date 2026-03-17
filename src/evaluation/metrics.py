"""
Module: evaluation/metrics.py
Hàm tính metric dùng chung cho toàn bộ dự án.
"""

import numpy as np
from sklearn.metrics import (
    f1_score,
    accuracy_score,
    precision_score,
    recall_score,
    confusion_matrix,
    mean_absolute_error,
    mean_squared_error,
    silhouette_score,
)


def classification_metrics(y_true, y_pred, average="macro") -> dict:
    """Tính tất cả metric phân loại."""
    return {
        "f1_macro": round(f1_score(y_true, y_pred, average=average), 4),
        "accuracy": round(accuracy_score(y_true, y_pred), 4),
        "precision_macro": round(precision_score(y_true, y_pred, average=average, zero_division=0), 4),
        "recall_macro": round(recall_score(y_true, y_pred, average=average, zero_division=0), 4),
    }


def regression_metrics(y_true, y_pred) -> dict:
    """Tính metric hồi quy / chuỗi thời gian."""
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100
    return {
        "mae": round(mae, 4),
        "rmse": round(rmse, 4),
        "mape": round(mape, 2),
    }


def clustering_metrics(X, labels) -> dict:
    """Tính metric phân cụm."""
    if len(set(labels)) < 2:
        return {"silhouette": 0.0}
    sil = silhouette_score(X, labels)
    return {
        "silhouette": round(sil, 4),
    }


def get_confusion_matrix(y_true, y_pred, labels=None):
    """Trả về confusion matrix."""
    return confusion_matrix(y_true, y_pred, labels=labels)
