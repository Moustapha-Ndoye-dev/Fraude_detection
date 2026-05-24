from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from ml_project.config import PATHS
from ml_project.data.loaders import read_fraud_transactions
from ml_project.models.fraud import train_fraud_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Entrainer un modele de detection de fraude.")
    parser.add_argument("--model", default="random_forest", choices=["random_forest", "logistic_regression"])
    parser.add_argument("--nrows", type=int, default=None, help="Limiter le nombre de lignes pour un test rapide.")
    parser.add_argument("--test-size", type=float, default=0.2)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    PATHS.ensure_outputs()

    df = read_fraud_transactions(nrows=args.nrows)
    result = train_fraud_model(df, model_name=args.model, test_size=args.test_size)

    model_path = PATHS.model_dir / "fraud_pipeline.joblib"
    metrics_path = PATHS.report_dir / "fraud_metrics.json"
    joblib.dump(result.pipeline, model_path)
    metrics_path.write_text(json.dumps(result.metrics, indent=2), encoding="utf-8")

    print(f"Modele sauvegarde: {model_path}")
    print(f"Metriques sauvegardees: {metrics_path}")
    print(json.dumps(result.metrics, indent=2))


if __name__ == "__main__":
    main()
