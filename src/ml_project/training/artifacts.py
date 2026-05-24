from __future__ import annotations

import json

import joblib
import pandas as pd

from ml_project.config import PATHS
from ml_project.features.customers import add_customer_features
from ml_project.models.clustering import ClusteringResult
from ml_project.models.fraud import FraudTrainingResult
from ml_project.models.selection import build_selection_record, save_selection_record


def persist_fraud_model(
    result: FraudTrainingResult,
    comparison: pd.DataFrame | None = None,
    *,
    deployment_meta: dict | None = None,
) -> None:
    PATHS.ensure_outputs()
    model_path = PATHS.model_dir / "fraud_pipeline.joblib"
    metrics_path = PATHS.report_dir / "fraud_metrics.json"
    selection_path = PATHS.report_dir / "fraud_model_selection.json"

    joblib.dump(result.pipeline, model_path)
    metrics_path.write_text(json.dumps(result.metrics, indent=2), encoding="utf-8")

    if comparison is not None:
        extra = {"artifact_path": str(model_path.relative_to(PATHS.root))}
        if deployment_meta:
            extra.update(deployment_meta)
        record = build_selection_record(
            "fraud",
            result.model_name,
            comparison,
            criteria="f1 desc, recall desc, precision desc, roc_auc desc",
            extra=extra,
        )
        save_selection_record(selection_path, record)


def persist_clustering_model(
    result: ClusteringResult,
    source_df: pd.DataFrame,
    comparison: pd.DataFrame | None = None,
) -> None:
    PATHS.ensure_outputs()
    enriched = add_customer_features(source_df)
    enriched["segment"] = result.labels

    model_path = PATHS.model_dir / "customer_clustering.joblib"
    segments_path = PATHS.processed_dir / "customer_segments.csv"
    metrics_path = PATHS.report_dir / "customer_clustering_metrics.json"
    selection_path = PATHS.report_dir / "clustering_model_selection.json"

    joblib.dump(result.pipeline, model_path)
    enriched.to_csv(segments_path, index=False)
    metrics_path.write_text(json.dumps(result.metrics, indent=2), encoding="utf-8")

    if comparison is not None:
        record = build_selection_record(
            "clustering",
            result.model_name,
            comparison,
            criteria="silhouette desc, davies_bouldin asc, sans bruit DBSCAN",
            extra={"artifact_path": str(model_path.relative_to(PATHS.root))},
        )
        save_selection_record(selection_path, record)
