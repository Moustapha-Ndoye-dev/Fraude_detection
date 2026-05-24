from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from ml_project.config import PATHS, RANDOM_STATE
from ml_project.data.loaders import read_fraud_transactions
from ml_project.features.fraud import split_fraud_features


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Interpreter le modele fraude et analyser FP/FN.")
    parser.add_argument("--nrows", type=int, default=None)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--shap-sample", type=int, default=200, help="Taille de l'echantillon SHAP.")
    return parser.parse_args()


def feature_importance_frame(pipeline) -> pd.DataFrame:
    preprocessor = pipeline.named_steps["preprocess"]
    model = pipeline.named_steps["model"]
    names = preprocessor.get_feature_names_out()

    if hasattr(model, "feature_importances_"):
        values = model.feature_importances_
    elif hasattr(model, "coef_"):
        values = abs(model.coef_[0])
    else:
        return pd.DataFrame(columns=["feature", "importance"])

    return (
        pd.DataFrame({"feature": names, "importance": values})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )


def main() -> None:
    args = parse_args()
    PATHS.ensure_outputs()

    model_path = PATHS.model_dir / "fraud_pipeline.joblib"
    if not model_path.exists():
        raise FileNotFoundError("Modele fraude introuvable. Lancez scripts/train_fraud.py avant l'interpretation.")

    df = read_fraud_transactions(nrows=args.nrows)
    X, y = split_fraud_features(df)
    _, X_test, _, y_test = train_test_split(
        X,
        y,
        test_size=args.test_size,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    pipeline = joblib.load(model_path)
    probability = pipeline.predict_proba(X_test)[:, 1]
    prediction = (probability >= args.threshold).astype(int)

    errors = X_test.copy()
    errors["isFraud"] = y_test.values
    errors["fraud_probability"] = probability
    errors["predicted_isFraud"] = prediction
    false_positives = errors[(errors["isFraud"] == 0) & (errors["predicted_isFraud"] == 1)]
    false_negatives = errors[(errors["isFraud"] == 1) & (errors["predicted_isFraud"] == 0)]

    importance = feature_importance_frame(pipeline)
    importance_path = PATHS.report_dir / "fraud_feature_importance.csv"
    fp_path = PATHS.report_dir / "fraud_false_positives_sample.csv"
    fn_path = PATHS.report_dir / "fraud_false_negatives_sample.csv"
    shap_note_path = PATHS.report_dir / "fraud_shap_note.json"
    shap_path = PATHS.report_dir / "fraud_shap_importance.csv"

    importance.head(30).to_csv(importance_path, index=False)
    false_positives.sort_values("fraud_probability", ascending=False).head(50).to_csv(fp_path, index=False)
    false_negatives.sort_values("fraud_probability", ascending=False).head(50).to_csv(fn_path, index=False)

    shap_note = {"status": "not_run", "false_positives": int(len(false_positives)), "false_negatives": int(len(false_negatives))}
    try:
        import shap

        preprocessor = pipeline.named_steps["preprocess"]
        model = pipeline.named_steps["model"]
        if hasattr(model, "estimators_"):
            sample = X_test.sample(min(args.shap_sample, len(X_test)), random_state=RANDOM_STATE)
            transformed = preprocessor.transform(sample)
            if hasattr(transformed, "toarray"):
                transformed = transformed.toarray()
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(transformed)
            values = shap_values[1] if isinstance(shap_values, list) else shap_values
            values = np.asarray(values)
            if values.ndim == 3:
                values = values[:, :, -1]
            shap_importance = (
                pd.DataFrame(
                    {
                        "feature": preprocessor.get_feature_names_out(),
                        "mean_abs_shap": np.abs(values).mean(axis=0),
                    }
                )
                .sort_values("mean_abs_shap", ascending=False)
                .reset_index(drop=True)
            )
            shap_importance.head(30).to_csv(shap_path, index=False)
            shap_note.update(
                {
                    "status": "ok",
                    "sample_size": int(len(sample)),
                    "output": str(shap_path),
                    "message": "SHAP TreeExplainer execute sur un echantillon du modele RandomForest.",
                }
            )
        else:
            shap_note.update({"status": "skipped", "message": "Le modele courant n'est pas un modele arbre compatible TreeExplainer."})
    except Exception as exc:
        shap_note.update({"status": "error", "message": str(exc)})
    shap_note_path.write_text(json.dumps(shap_note, indent=2), encoding="utf-8")

    print(f"Importance variables: {importance_path}")
    print(f"Faux positifs: {fp_path} ({len(false_positives)})")
    print(f"Faux negatifs: {fn_path} ({len(false_negatives)})")
    print(f"SHAP importance: {shap_path if shap_path.exists() else 'non genere'}")
    print(f"Note SHAP: {shap_note_path}")


if __name__ == "__main__":
    main()
