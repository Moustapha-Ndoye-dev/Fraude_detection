from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from ml_project.config import RANDOM_STATE
from ml_project.evaluation.metrics import binary_classification_metrics
from ml_project.features.fraud import split_fraud_features


@dataclass
class FraudTrainingResult:
    pipeline: Pipeline
    metrics: dict[str, float | None]
    test_size: float
    model_name: str


def build_fraud_pipeline(X: pd.DataFrame, model_name: str = "random_forest") -> Pipeline:
    categorical_columns = ["type"]
    numeric_columns = [column for column in X.columns if column not in categorical_columns]

    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", StandardScaler(), numeric_columns),
            ("categorical", OneHotEncoder(handle_unknown="ignore"), categorical_columns),
        ]
    )

    if model_name == "logistic_regression":
        estimator = LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )
    elif model_name == "random_forest":
        estimator = RandomForestClassifier(
            n_estimators=250,
            min_samples_leaf=2,
            class_weight="balanced_subsample",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )
    elif model_name == "neural_network":
        estimator = MLPClassifier(
            hidden_layer_sizes=(64, 32),
            activation="relu",
            learning_rate_init=0.001,
            max_iter=80,
            random_state=RANDOM_STATE,
            early_stopping=True,
        )
    elif model_name == "xgboost":
        try:
            from xgboost import XGBClassifier
        except ImportError as exc:
            raise ImportError("Installer xgboost pour entrainer ce modele: pip install xgboost") from exc
        estimator = XGBClassifier(
            n_estimators=180,
            max_depth=5,
            learning_rate=0.08,
            subsample=0.9,
            colsample_bytree=0.9,
            objective="binary:logistic",
            eval_metric="logloss",
            tree_method="hist",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )
    elif model_name == "lightgbm":
        try:
            from lightgbm import LGBMClassifier
        except ImportError as exc:
            raise ImportError("Installer lightgbm pour entrainer ce modele: pip install lightgbm") from exc
        estimator = LGBMClassifier(
            n_estimators=180,
            learning_rate=0.08,
            num_leaves=31,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1,
            verbosity=-1,
        )
    else:
        raise ValueError(f"Modele non supporte: {model_name}")

    return Pipeline(steps=[("preprocess", preprocessor), ("model", estimator)])


def train_fraud_model(
    df: pd.DataFrame,
    model_name: str = "random_forest",
    test_size: float = 0.2,
) -> FraudTrainingResult:
    X, y = split_fraud_features(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    pipeline = build_fraud_pipeline(X_train, model_name=model_name)
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    y_score = pipeline.predict_proba(X_test)[:, 1] if hasattr(pipeline[-1], "predict_proba") else None
    metrics = binary_classification_metrics(y_test, y_pred, y_score)
    return FraudTrainingResult(
        pipeline=pipeline,
        metrics=metrics,
        test_size=test_size,
        model_name=model_name,
    )
