from __future__ import annotations

from typing import Any

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    davies_bouldin_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    silhouette_score,
)


def binary_classification_metrics(y_true, y_pred, y_score=None) -> dict[str, float | None]:
    metrics: dict[str, float | None] = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": None,
    }
    if y_score is not None:
        metrics["roc_auc"] = roc_auc_score(y_true, y_score)
    return metrics


def clustering_metrics(X_transformed: Any, labels) -> dict[str, float | None]:
    unique_labels = set(labels)
    if len(unique_labels) < 2:
        return {"silhouette": None, "davies_bouldin": None}
    return {
        "silhouette": silhouette_score(X_transformed, labels),
        "davies_bouldin": davies_bouldin_score(X_transformed, labels),
    }


def metrics_to_frame(metrics: dict[str, float | None]) -> pd.DataFrame:
    return pd.DataFrame([metrics])
