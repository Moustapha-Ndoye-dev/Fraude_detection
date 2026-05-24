from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd

from ml_project.config import PATHS


FRAUD_REQUIRED_COLUMNS = {
    "step",
    "type",
    "amount",
    "nameOrig",
    "oldbalanceOrg",
    "newbalanceOrig",
    "nameDest",
    "oldbalanceDest",
    "newbalanceDest",
    "isFraud",
    "isFlaggedFraud",
}

CUSTOMER_REQUIRED_COLUMNS = {
    "ID",
    "Year_Birth",
    "Education",
    "Marital_Status",
    "Income",
    "Kidhome",
    "Teenhome",
    "Dt_Customer",
    "Recency",
    "Response",
}


def _read_semicolon_csv(path: Path, nrows: int | None = None) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Fichier introuvable: {path}")
    return pd.read_csv(path, sep=";", nrows=nrows)


def _validate_columns(df: pd.DataFrame, required: Iterable[str], dataset_name: str) -> None:
    missing = sorted(set(required) - set(df.columns))
    if missing:
        raise ValueError(f"Colonnes manquantes dans {dataset_name}: {missing}")


def read_fraud_transactions(nrows: int | None = None) -> pd.DataFrame:
    df = _read_semicolon_csv(PATHS.fraud_data, nrows=nrows)
    _validate_columns(df, FRAUD_REQUIRED_COLUMNS, "detection_fraude")
    return df


def read_customer_data(nrows: int | None = None) -> pd.DataFrame:
    df = _read_semicolon_csv(PATHS.customer_data, nrows=nrows)
    _validate_columns(df, CUSTOMER_REQUIRED_COLUMNS, "data_cluster")
    return df
