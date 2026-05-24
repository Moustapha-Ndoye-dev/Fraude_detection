from __future__ import annotations

import pandas as pd
import pytest

from ml_project.models.selection import (
    rank_clustering_models,
    rank_fraud_models,
    select_best_clustering_model,
    select_best_fraud_model,
)


def test_select_best_fraud_model_prefers_highest_f1():
    comparison = pd.DataFrame(
        [
            {"model": "random_forest", "status": "ok", "f1": 0.98, "recall": 0.96, "precision": 1.0, "roc_auc": 0.99},
            {"model": "lightgbm", "status": "ok", "f1": 0.99, "recall": 0.98, "precision": 1.0, "roc_auc": 1.0},
            {"model": "neural_network", "status": "ok", "f1": 0.30, "recall": 0.17, "precision": 1.0, "roc_auc": 0.99},
        ]
    )
    assert select_best_fraud_model(comparison) == "lightgbm"


def test_select_best_fraud_model_breaks_ties_with_preferred_order():
    comparison = pd.DataFrame(
        [
            {"model": "random_forest", "status": "ok", "f1": 0.98, "recall": 0.96, "precision": 1.0, "roc_auc": 0.99},
            {"model": "lightgbm", "status": "ok", "f1": 0.98, "recall": 0.96, "precision": 1.0, "roc_auc": 0.99},
            {"model": "xgboost", "status": "ok", "f1": 0.98, "recall": 0.96, "precision": 1.0, "roc_auc": 0.99},
        ]
    )
    assert select_best_fraud_model(comparison) == "lightgbm"


def test_select_best_clustering_model_excludes_dbscan_noise():
    comparison = pd.DataFrame(
        [
            {"model": "dbscan", "silhouette": -0.20, "davies_bouldin": 1.90, "noise_points": 500},
            {"model": "kmeans", "silhouette": 0.17, "davies_bouldin": 2.09, "noise_points": 0},
            {"model": "gaussian_mixture", "silhouette": 0.23, "davies_bouldin": 2.15, "noise_points": 0},
        ]
    )
    assert select_best_clustering_model(comparison) == "gaussian_mixture"


def test_rank_fraud_models_ignores_failed_runs():
    comparison = pd.DataFrame(
        [
            {"model": "xgboost", "status": "optional_dependency_missing", "f1": None, "recall": None, "precision": None, "roc_auc": None},
            {"model": "random_forest", "status": "ok", "f1": 0.98, "recall": 0.96, "precision": 1.0, "roc_auc": 0.99},
        ]
    )
    ranked = rank_fraud_models(comparison)
    assert len(ranked) == 1
    assert ranked.iloc[0]["model"] == "random_forest"
