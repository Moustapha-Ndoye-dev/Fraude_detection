from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from ml_project.config import PATHS
from ml_project.data.loaders import read_fraud_transactions
from ml_project.models.fraud import train_fraud_model
from ml_project.models.selection import resolve_fraud_model
from ml_project.training.artifacts import persist_fraud_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Entrainer un modele de detection de fraude.")
    parser.add_argument(
        "--model",
        default="best",
        choices=["best", "logistic_regression", "random_forest", "xgboost", "lightgbm", "neural_network"],
    )
    parser.add_argument("--nrows", type=int, default=None, help="Limiter le nombre de lignes pour un test rapide.")
    parser.add_argument("--test-size", type=float, default=0.2)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    PATHS.ensure_outputs()

    comparison_path = PATHS.report_dir / "fraud_model_comparison.csv"
    model_name = resolve_fraud_model(args.model, comparison_path, deployable=args.model == "best")

    df = read_fraud_transactions(nrows=args.nrows)
    result = train_fraud_model(df, model_name=model_name, test_size=args.test_size)

    comparison = None
    if comparison_path.exists():
        import pandas as pd

        comparison = pd.read_csv(comparison_path)

    persist_fraud_model(result, comparison=comparison if args.model == "best" else None)

    print(f"Modele entraine: {model_name}")
    print(f"Modele sauvegarde: {PATHS.model_dir / 'fraud_pipeline.joblib'}")
    print(f"Metriques sauvegardees: {PATHS.report_dir / 'fraud_metrics.json'}")
    print(json.dumps(result.metrics, indent=2))


if __name__ == "__main__":
    main()
