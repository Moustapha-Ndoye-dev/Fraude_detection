from __future__ import annotations

import numpy as np
import pandas as pd


TARGET = "isFraud"
DROP_COLUMNS = ["nameOrig", "nameDest", "isFlaggedFraud"]


def add_fraud_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["orig_balance_delta"] = out["oldbalanceOrg"] - out["newbalanceOrig"]
    out["dest_balance_delta"] = out["newbalanceDest"] - out["oldbalanceDest"]
    out["orig_delta_error"] = out["orig_balance_delta"] - out["amount"]
    out["dest_delta_error"] = out["dest_balance_delta"] - out["amount"]
    out["amount_to_oldbalance_ratio"] = out["amount"] / (out["oldbalanceOrg"] + 1)
    out["emptied_origin"] = ((out["oldbalanceOrg"] > 0) & (out["newbalanceOrig"] == 0)).astype(int)
    out["dest_is_merchant"] = out["nameDest"].astype(str).str.startswith("M").astype(int)
    out["dest_is_customer"] = out["nameDest"].astype(str).str.startswith("C").astype(int)
    out["log_amount"] = np.log1p(out["amount"])
    return out


def split_fraud_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    if TARGET not in df:
        raise ValueError(f"Colonne cible manquante: {TARGET}")
    featured = add_fraud_features(df)
    y = featured[TARGET].astype(int)
    X = featured.drop(columns=[TARGET, *DROP_COLUMNS], errors="ignore")
    return X, y
