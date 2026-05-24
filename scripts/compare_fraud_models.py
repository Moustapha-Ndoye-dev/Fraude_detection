from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from ml_project.config import PATHS
from ml_project.data.loaders import read_fraud_transactions
from ml_project.models.fraud import train_fraud_model


MODEL_ORDER = ["logistic_regression", "random_forest", "xgboost", "lightgbm", "neural_network"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Comparer plusieurs modeles de detection de fraude.")
    parser.add_argument("--nrows", type=int, default=200000, help="Nombre de lignes utilisees pour la comparaison.")
    parser.add_argument("--test-size", type=float, default=0.2)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    PATHS.ensure_outputs()
    df = read_fraud_transactions(nrows=args.nrows)
    rows = []

    for model_name in MODEL_ORDER:
        started = time.perf_counter()
        try:
            result = train_fraud_model(df, model_name=model_name, test_size=args.test_size)
            elapsed = time.perf_counter() - started
            rows.append(
                {
                    "model": model_name,
                    "status": "ok",
                    "nrows": len(df),
                    "elapsed_seconds": round(elapsed, 2),
                    **result.metrics,
                    "note": "",
                }
            )
        except ImportError as exc:
            rows.append(
                {
                    "model": model_name,
                    "status": "optional_dependency_missing",
                    "nrows": len(df),
                    "elapsed_seconds": None,
                    "accuracy": None,
                    "precision": None,
                    "recall": None,
                    "f1": None,
                    "roc_auc": None,
                    "note": str(exc),
                }
            )

    output = pd.DataFrame(rows)
    output_path = PATHS.report_dir / "fraud_model_comparison.csv"
    output.to_csv(output_path, index=False)
    print(output.to_string(index=False))
    print(f"Comparaison sauvegardee: {output_path}")


if __name__ == "__main__":
    main()
