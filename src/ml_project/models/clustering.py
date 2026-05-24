from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from sklearn.cluster import AgglomerativeClustering, DBSCAN, KMeans
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.mixture import GaussianMixture
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from ml_project.config import RANDOM_STATE
from ml_project.evaluation.metrics import clustering_metrics
from ml_project.features.customers import customer_model_frame


@dataclass
class ClusteringResult:
    pipeline: Pipeline
    labels: list[int]
    metrics: dict[str, float | None]
    model_name: str


def build_customer_preprocessor(df: pd.DataFrame) -> ColumnTransformer:
    categorical_columns = ["Education", "Marital_Status"]
    numeric_columns = [column for column in df.columns if column not in categorical_columns]

    numeric_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipe, numeric_columns),
            ("categorical", categorical_pipe, categorical_columns),
        ],
        verbose_feature_names_out=False,
    )


def _build_clusterer(model_name: str, n_clusters: int):
    if model_name == "kmeans":
        return KMeans(n_clusters=n_clusters, random_state=RANDOM_STATE, n_init="auto")
    if model_name == "agglomerative":
        return AgglomerativeClustering(n_clusters=n_clusters)
    if model_name == "dbscan":
        return DBSCAN(eps=1.5, min_samples=8)
    if model_name == "gaussian_mixture":
        return GaussianMixture(n_components=n_clusters, random_state=RANDOM_STATE)
    raise ValueError(f"Modele de clustering non supporte: {model_name}")


def fit_customer_clustering(
    df: pd.DataFrame,
    model_name: str = "kmeans",
    n_clusters: int = 4,
) -> ClusteringResult:
    X = customer_model_frame(df)
    preprocessor = build_customer_preprocessor(X)
    X_transformed = preprocessor.fit_transform(X)
    clusterer = _build_clusterer(model_name, n_clusters)

    if model_name == "gaussian_mixture":
        labels = clusterer.fit_predict(X_transformed)
        pipeline = Pipeline(steps=[("preprocess", preprocessor), ("clusterer", clusterer)])
    else:
        labels = clusterer.fit_predict(X_transformed)
        pipeline = Pipeline(steps=[("preprocess", preprocessor), ("clusterer", clusterer)])

    metrics = clustering_metrics(X_transformed, labels)
    return ClusteringResult(
        pipeline=pipeline,
        labels=[int(label) for label in labels],
        metrics=metrics,
        model_name=model_name,
    )


def score_candidate_k(df: pd.DataFrame, candidate_k: list[int] | None = None) -> pd.DataFrame:
    candidate_k = candidate_k or [2, 3, 4, 5, 6, 7, 8]
    rows = []
    for k in candidate_k:
        result = fit_customer_clustering(df, model_name="kmeans", n_clusters=k)
        rows.append({"k": k, **result.metrics})
    return pd.DataFrame(rows)
