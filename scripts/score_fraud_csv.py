from __future__ import annotations

import argparse
import sys
from pathlib import Path

import joblib
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from ml_project.config import PATHS
from ml_project.serving.fraud_scoring import score_fraud_dataframe, summarize_scoring, validate_fraud_input


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Scorer un CSV de transactions avec validation stricte.")
    parser.add_argument("input_csv", help="Chemin du CSV a scorer.")
    parser.add_argument("--output", default="data/processed/fraud_scoring_results.csv")
    parser.add_argument("--delimiter", default=";")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--allow-extra-columns", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input_csv)
    output_path = PROJECT_ROOT / args.output if not Path(args.output).is_absolute() else Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    model_path = PATHS.model_dir / "fraud_pipeline.joblib"
    if not model_path.exists():
        raise FileNotFoundError(f"Modele introuvable: {model_path}")

    df = pd.read_csv(input_path, sep=args.delimiter)
    validation = validate_fraud_input(df, strict=not args.allow_extra_columns)
    if not validation.is_valid:
        print("Fichier rejete.")
        for error in validation.errors:
            print(f"- {error}")
        raise SystemExit(2)

    model = joblib.load(model_path)
    scored = score_fraud_dataframe(model, validation.dataframe, threshold=args.threshold)
    scored.to_csv(output_path, index=False, sep=args.delimiter)
    summary = summarize_scoring(scored, threshold=args.threshold)

    print("Scoring termine.")
    for key, value in summary.items():
        print(f"{key}: {value}")
    print(f"Fichier score: {output_path}")


if __name__ == "__main__":
    main()
