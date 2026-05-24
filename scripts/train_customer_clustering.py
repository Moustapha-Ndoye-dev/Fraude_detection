from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from ml_project.config import PATHS
from ml_project.data.loaders import read_customer_data
from ml_project.models.clustering import fit_customer_clustering, score_candidate_k
from ml_project.models.selection import resolve_clustering_model
from ml_project.training.artifacts import persist_clustering_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Entrainer un modele de segmentation client.")
    parser.add_argument(
        "--model",
        default="best",
        choices=["best", "kmeans", "agglomerative", "dbscan", "gaussian_mixture"],
    )
    parser.add_argument("--clusters", type=int, default=4)
    parser.add_argument("--search-k", action="store_true", help="Calculer les scores pour plusieurs k.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    PATHS.ensure_outputs()

    df = read_customer_data()
    if args.search_k:
        scores = score_candidate_k(df)
        scores_path = PATHS.report_dir / "customer_k_scores.csv"
        scores.to_csv(scores_path, index=False)
        print(scores)
        print(f"Scores k sauvegardes: {scores_path}")

    comparison_path = PATHS.report_dir / "clustering_model_comparison.csv"
    model_name = resolve_clustering_model(args.model, comparison_path)
    result = fit_customer_clustering(df, model_name=model_name, n_clusters=args.clusters)

    comparison = None
    if comparison_path.exists():
        import pandas as pd

        comparison = pd.read_csv(comparison_path)

    persist_clustering_model(result, df, comparison=comparison if args.model == "best" else None)

    print(f"Modele entraine: {model_name}")
    print(f"Modele sauvegarde: {PATHS.model_dir / 'customer_clustering.joblib'}")
    print(f"Segments sauvegardes: {PATHS.processed_dir / 'customer_segments.csv'}")
    print(json.dumps(result.metrics, indent=2))


if __name__ == "__main__":
    main()
