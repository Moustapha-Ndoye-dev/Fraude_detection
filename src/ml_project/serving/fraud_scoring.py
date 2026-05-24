from __future__ import annotations

from dataclasses import dataclass
from io import StringIO

import pandas as pd

from ml_project.features.fraud import DROP_COLUMNS, add_fraud_features


FRAUD_ALLOWED_TYPES = {"PAYMENT", "TRANSFER", "CASH_OUT", "DEBIT", "CASH_IN"}

FRAUD_REQUIRED_INPUT_COLUMNS = [
    "step",
    "type",
    "amount",
    "nameOrig",
    "oldbalanceOrg",
    "newbalanceOrig",
    "nameDest",
    "oldbalanceDest",
    "newbalanceDest",
]

FRAUD_OPTIONAL_INPUT_COLUMNS = ["isFlaggedFraud", "isFraud"]

FRAUD_NUMERIC_COLUMNS = [
    "step",
    "amount",
    "oldbalanceOrg",
    "newbalanceOrig",
    "oldbalanceDest",
    "newbalanceDest",
]


@dataclass(frozen=True)
class ValidationResult:
    is_valid: bool
    errors: list[str]
    warnings: list[str]
    dataframe: pd.DataFrame


def expected_fraud_schema() -> pd.DataFrame:
    return pd.DataFrame(
        [
            ["step", "integer", "obligatoire", "Unite temporelle de transaction"],
            ["type", "category", "obligatoire", "PAYMENT, TRANSFER, CASH_OUT, DEBIT ou CASH_IN"],
            ["amount", "float", "obligatoire", "Montant transfere"],
            ["nameOrig", "string", "obligatoire", "Identifiant client emetteur"],
            ["oldbalanceOrg", "float", "obligatoire", "Solde emetteur avant transaction"],
            ["newbalanceOrig", "float", "obligatoire", "Solde emetteur apres transaction"],
            ["nameDest", "string", "obligatoire", "Identifiant destinataire"],
            ["oldbalanceDest", "float", "obligatoire", "Solde destinataire avant transaction"],
            ["newbalanceDest", "float", "obligatoire", "Solde destinataire apres transaction"],
            ["isFlaggedFraud", "integer", "optionnel", "Flag amont, mis a 0 si absent"],
        ],
        columns=["colonne", "type_attendu", "statut", "description"],
    )


def template_fraud_csv(delimiter: str = ";") -> str:
    sample = pd.DataFrame(
        [
            {
                "step": 1,
                "type": "TRANSFER",
                "amount": 181.0,
                "nameOrig": "C1305486145",
                "oldbalanceOrg": 181.0,
                "newbalanceOrig": 0.0,
                "nameDest": "C553264065",
                "oldbalanceDest": 0.0,
                "newbalanceDest": 0.0,
                "isFlaggedFraud": 0,
            },
            {
                "step": 1,
                "type": "PAYMENT",
                "amount": 9839.64,
                "nameOrig": "C1231006815",
                "oldbalanceOrg": 170136.0,
                "newbalanceOrig": 160296.36,
                "nameDest": "M1979787155",
                "oldbalanceDest": 0.0,
                "newbalanceDest": 0.0,
                "isFlaggedFraud": 0,
            },
        ]
    )
    buffer = StringIO()
    sample.to_csv(buffer, sep=delimiter, index=False)
    return buffer.getvalue()


def validate_fraud_input(
    df: pd.DataFrame,
    strict: bool = True,
    max_errors: int = 20,
) -> ValidationResult:
    errors: list[str] = []
    warnings: list[str] = []

    if df.empty:
        return ValidationResult(False, ["Le fichier est vide."], warnings, df)

    df = df.copy()
    df.columns = [str(column).strip() for column in df.columns]
    expected = set(FRAUD_REQUIRED_INPUT_COLUMNS + FRAUD_OPTIONAL_INPUT_COLUMNS)
    missing = [column for column in FRAUD_REQUIRED_INPUT_COLUMNS if column not in df.columns]
    extra = sorted(set(df.columns) - expected)

    if missing:
        errors.append(f"Colonnes obligatoires manquantes: {', '.join(missing)}")
    if strict and extra:
        errors.append(f"Colonnes non attendues en mode strict: {', '.join(extra)}")
    elif extra:
        warnings.append(f"Colonnes ignorees: {', '.join(extra)}")

    if errors:
        return ValidationResult(False, errors[:max_errors], warnings, df)

    if "isFlaggedFraud" not in df.columns:
        df["isFlaggedFraud"] = 0
        warnings.append("Colonne isFlaggedFraud absente: valeur 0 appliquee par defaut.")

    for column in FRAUD_NUMERIC_COLUMNS + ["isFlaggedFraud"]:
        before_missing = df[column].isna().sum()
        df[column] = pd.to_numeric(df[column], errors="coerce")
        invalid_count = int(df[column].isna().sum() - before_missing)
        if invalid_count > 0:
            errors.append(f"{column}: {invalid_count} valeur(s) non numerique(s).")

    invalid_types = sorted(set(df["type"].dropna().astype(str).str.upper()) - FRAUD_ALLOWED_TYPES)
    if invalid_types:
        errors.append(f"type: valeurs non autorisees: {', '.join(invalid_types)}")
    df["type"] = df["type"].astype(str).str.upper()

    for column in ["nameOrig", "nameDest"]:
        empty_count = int(df[column].isna().sum() + (df[column].astype(str).str.strip() == "").sum())
        if empty_count > 0:
            errors.append(f"{column}: {empty_count} valeur(s) vide(s).")

    for column in ["amount", "oldbalanceOrg", "newbalanceOrig", "oldbalanceDest", "newbalanceDest"]:
        negative_count = int((df[column] < 0).sum())
        if negative_count > 0:
            errors.append(f"{column}: {negative_count} valeur(s) negative(s).")

    invalid_steps = int((df["step"] < 0).sum())
    if invalid_steps > 0:
        errors.append(f"step: {invalid_steps} valeur(s) negative(s).")

    if errors:
        return ValidationResult(False, errors[:max_errors], warnings, df)

    output_columns = FRAUD_REQUIRED_INPUT_COLUMNS + ["isFlaggedFraud"]
    if "isFraud" in df.columns:
        output_columns.append("isFraud")
    return ValidationResult(True, [], warnings, df[output_columns])


def score_fraud_dataframe(model, df: pd.DataFrame, threshold: float = 0.5) -> pd.DataFrame:
    X = add_fraud_features(df).drop(columns=[*DROP_COLUMNS, "isFraud"], errors="ignore")
    probabilities = model.predict_proba(X)[:, 1]
    predictions = (probabilities >= threshold).astype(int)

    scored = df.copy()
    scored["fraud_probability"] = probabilities
    scored["predicted_isFraud"] = predictions
    scored["risk_band"] = pd.cut(
        scored["fraud_probability"],
        bins=[-0.01, 0.2, 0.5, 0.8, 1.0],
        labels=["LOW", "MEDIUM", "HIGH", "CRITICAL"],
    ).astype(str)
    scored["recommended_action"] = scored["risk_band"].map(
        {
            "LOW": "AUTO_APPROVE",
            "MEDIUM": "MONITOR",
            "HIGH": "MANUAL_REVIEW",
            "CRITICAL": "TEMPORARY_BLOCK",
        }
    )
    return scored


def summarize_scoring(scored: pd.DataFrame, threshold: float = 0.5) -> dict[str, float | int]:
    total_rows = int(len(scored))
    predicted_frauds = int(scored["predicted_isFraud"].sum())
    return {
        "rows": total_rows,
        "threshold": threshold,
        "predicted_frauds": predicted_frauds,
        "predicted_fraud_rate": predicted_frauds / total_rows if total_rows else 0.0,
        "average_probability": float(scored["fraud_probability"].mean()) if total_rows else 0.0,
        "critical_transactions": int((scored["risk_band"] == "CRITICAL").sum()),
        "manual_review_transactions": int((scored["recommended_action"] == "MANUAL_REVIEW").sum()),
    }
