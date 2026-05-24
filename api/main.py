from __future__ import annotations

import sys
from io import BytesIO
from pathlib import Path
from typing import Literal

import joblib
import pandas as pd
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from ml_project.config import PATHS
from ml_project.serving.fraud_scoring import (
    expected_fraud_schema,
    score_fraud_dataframe,
    summarize_scoring,
    template_fraud_csv,
    validate_fraud_input,
)


class FraudTransaction(BaseModel):
    step: int
    type: Literal["PAYMENT", "TRANSFER", "CASH_OUT", "DEBIT", "CASH_IN"]
    amount: float
    nameOrig: str = "C_UNKNOWN"
    oldbalanceOrg: float
    newbalanceOrig: float
    nameDest: str = "C_UNKNOWN"
    oldbalanceDest: float
    newbalanceDest: float
    isFlaggedFraud: int = 0


app = FastAPI(title="Fraud and Customer ML API", version="0.1.0")


def _load_fraud_model():
    model_path = PATHS.model_dir / "fraud_pipeline.joblib"
    if not model_path.exists():
        raise HTTPException(
            status_code=503,
            detail="Modele fraude introuvable. Lancer scripts/train_fraud.py avant l'API.",
        )
    try:
        return joblib.load(model_path)
    except ModuleNotFoundError as exc:
        missing = getattr(exc, "name", None) or str(exc)
        raise HTTPException(
            status_code=503,
            detail=(
                f"Dependance manquante pour charger le modele ({missing}). "
                "Regenerer models/fraud_pipeline.joblib avec random_forest "
                "(python scripts/train_fraud.py --model random_forest)."
            ),
        ) from exc


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "service": "fraud-scoring-api"}


@app.get("/schema/fraud")
def fraud_schema() -> dict[str, object]:
    return {
        "delimiter": ";",
        "strict_schema": True,
        "columns": expected_fraud_schema().to_dict(orient="records"),
        "csv_template": template_fraud_csv(),
    }


@app.post("/predict/fraud")
def predict_fraud(transaction: FraudTransaction) -> dict[str, float | int]:
    model = _load_fraud_model()
    payload = transaction.model_dump() if hasattr(transaction, "model_dump") else transaction.dict()
    validation = validate_fraud_input(pd.DataFrame([payload]), strict=True)
    if not validation.is_valid:
        raise HTTPException(status_code=422, detail=validation.errors)
    scored = score_fraud_dataframe(model, validation.dataframe, threshold=0.5)
    row = scored.iloc[0]
    return {
        "is_fraud": int(row["predicted_isFraud"]),
        "fraud_probability": float(row["fraud_probability"]),
        "risk_band": str(row["risk_band"]),
        "recommended_action": str(row["recommended_action"]),
    }


def _read_uploaded_csv(upload: UploadFile, delimiter: str) -> pd.DataFrame:
    if not upload.filename or not upload.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=415, detail="Le fichier doit etre un CSV.")
    try:
        upload.file.seek(0)
        return pd.read_csv(upload.file, sep=delimiter)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Lecture CSV impossible: {exc}") from exc


@app.post("/score/fraud/csv/summary")
def score_fraud_csv_summary(
    file: UploadFile = File(...),
    delimiter: str = Query(";", min_length=1, max_length=1),
    threshold: float = Query(0.5, ge=0.0, le=1.0),
    strict: bool = Query(True),
) -> dict[str, object]:
    model = _load_fraud_model()
    df = _read_uploaded_csv(file, delimiter)
    validation = validate_fraud_input(df, strict=strict)
    if not validation.is_valid:
        raise HTTPException(status_code=422, detail={"errors": validation.errors, "warnings": validation.warnings})

    scored = score_fraud_dataframe(model, validation.dataframe, threshold=threshold)
    top_risk = (
        scored.sort_values("fraud_probability", ascending=False)
        .head(10)
        .to_dict(orient="records")
    )
    return {
        "status": "accepted",
        "warnings": validation.warnings,
        "summary": summarize_scoring(scored, threshold=threshold),
        "top_risk": top_risk,
    }


@app.post("/score/fraud/csv")
def score_fraud_csv(
    file: UploadFile = File(...),
    delimiter: str = Query(";", min_length=1, max_length=1),
    threshold: float = Query(0.5, ge=0.0, le=1.0),
    strict: bool = Query(True),
) -> StreamingResponse:
    model = _load_fraud_model()
    df = _read_uploaded_csv(file, delimiter)
    validation = validate_fraud_input(df, strict=strict)
    if not validation.is_valid:
        raise HTTPException(status_code=422, detail={"errors": validation.errors, "warnings": validation.warnings})

    scored = score_fraud_dataframe(model, validation.dataframe, threshold=threshold)
    buffer = BytesIO()
    scored.to_csv(buffer, index=False, sep=delimiter)
    buffer.seek(0)
    return StreamingResponse(
        buffer,
        media_type="text/csv",
        headers={"Content-Disposition": 'attachment; filename="fraud_scoring_results.csv"'},
    )
