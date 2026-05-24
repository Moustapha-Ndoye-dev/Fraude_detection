from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

FRAUD_MODEL_ORDER = [
    "lightgbm",
    "xgboost",
    "random_forest",
    "logistic_regression",
    "neural_network",
]

CLUSTERING_MODEL_ORDER = [
    "gaussian_mixture",
    "kmeans",
    "agglomerative",
    "dbscan",
]

FRAUD_RANK_COLUMNS = ["f1", "recall", "precision", "roc_auc"]
CLUSTERING_RANK_COLUMNS = ["silhouette", "davies_bouldin"]


def _model_rank(model_name: str, order: list[str]) -> int:
    try:
        return order.index(model_name)
    except ValueError:
        return len(order)


def rank_fraud_models(comparison: pd.DataFrame) -> pd.DataFrame:
    candidates = comparison[comparison["status"] == "ok"].copy()
    if candidates.empty:
        raise ValueError("Aucun modele fraude valide dans la comparaison.")

    for column in FRAUD_RANK_COLUMNS:
        candidates[column] = pd.to_numeric(candidates[column], errors="coerce")

    candidates = candidates.dropna(subset=["f1", "recall"])
    if candidates.empty:
        raise ValueError("Aucun modele fraude avec des metriques exploitables.")

    candidates["_rank"] = candidates["model"].map(lambda name: _model_rank(str(name), FRAUD_MODEL_ORDER))
    candidates = candidates.sort_values(
        by=[*FRAUD_RANK_COLUMNS, "_rank"],
        ascending=[False, False, False, False, True],
    )
    return candidates.drop(columns="_rank")


def select_best_fraud_model(comparison: pd.DataFrame) -> str:
    ranked = rank_fraud_models(comparison)
    return str(ranked.iloc[0]["model"])


def rank_clustering_models(comparison: pd.DataFrame) -> pd.DataFrame:
    candidates = comparison.copy()
    for column in CLUSTERING_RANK_COLUMNS:
        candidates[column] = pd.to_numeric(candidates[column], errors="coerce")

    candidates["noise_points"] = pd.to_numeric(candidates.get("noise_points"), errors="coerce").fillna(0)
    candidates = candidates[candidates["silhouette"].notna()]
    candidates = candidates[candidates["silhouette"] > 0]
    candidates = candidates[candidates["noise_points"] == 0]
    if candidates.empty:
        raise ValueError("Aucun algorithme de clustering valide dans la comparaison.")

    candidates["_rank"] = candidates["model"].map(lambda name: _model_rank(str(name), CLUSTERING_MODEL_ORDER))
    candidates = candidates.sort_values(
        by=["silhouette", "davies_bouldin", "_rank"],
        ascending=[False, True, True],
    )
    return candidates.drop(columns="_rank")


def select_best_clustering_model(comparison: pd.DataFrame) -> str:
    ranked = rank_clustering_models(comparison)
    return str(ranked.iloc[0]["model"])


def resolve_fraud_model(model_name: str, comparison_path: Path) -> str:
    if model_name != "best":
        return model_name
    if not comparison_path.exists():
        raise FileNotFoundError(
            f"Comparaison introuvable: {comparison_path}. "
            "Lancer: python scripts/compare_fraud_models.py"
        )
    comparison = pd.read_csv(comparison_path)
    return select_best_fraud_model(comparison)


def resolve_clustering_model(model_name: str, comparison_path: Path) -> str:
    if model_name != "best":
        return model_name
    if not comparison_path.exists():
        raise FileNotFoundError(
            f"Comparaison introuvable: {comparison_path}. "
            "Lancer: python scripts/compare_clustering_models.py"
        )
    comparison = pd.read_csv(comparison_path)
    return select_best_clustering_model(comparison)


def build_selection_record(
    task: str,
    selected_model: str,
    comparison: pd.DataFrame,
    *,
    criteria: str,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if task == "fraud":
        ranked = rank_fraud_models(comparison)
    elif task == "clustering":
        ranked = rank_clustering_models(comparison)
    else:
        raise ValueError(f"Tache inconnue: {task}")

    top_metrics = ranked.iloc[0].to_dict()
    return {
        "task": task,
        "selected_model": selected_model,
        "selection_criteria": criteria,
        "selected_at": datetime.now(timezone.utc).isoformat(),
        "ranking": [
            {key: (None if pd.isna(value) else value) for key, value in row.items()}
            for row in ranked.to_dict(orient="records")
        ],
        "selected_metrics": {key: (None if pd.isna(value) else value) for key, value in top_metrics.items()},
        **(extra or {}),
    }


def save_selection_record(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")


def load_selection_record(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))
