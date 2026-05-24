from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from ml_project.config import PATHS
from ml_project.data.loaders import read_customer_data
from ml_project.features.customers import add_customer_features
from ml_project.models.clustering import fit_customer_clustering, score_candidate_k


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Entrainer un modele de segmentation client.")
    parser.add_argument(
        "--model",
        default="kmeans",
        choices=["kmeans", "agglomerative", "dbscan", "gaussian_mixture"],
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

    result = fit_customer_clustering(df, model_name=args.model, n_clusters=args.clusters)
    enriched = add_customer_features(df)
    enriched["segment"] = result.labels

    model_path = PATHS.model_dir / "customer_clustering.joblib"
    segments_path = PATHS.processed_dir / "customer_segments.csv"
    metrics_path = PATHS.report_dir / "customer_clustering_metrics.json"

    joblib.dump(result.pipeline, model_path)
    enriched.to_csv(segments_path, index=False)
    metrics_path.write_text(json.dumps(result.metrics, indent=2), encoding="utf-8")

    print(f"Modele sauvegarde: {model_path}")
    print(f"Segments sauvegardes: {segments_path}")
    print(json.dumps(result.metrics, indent=2))


if __name__ == "__main__":
    main()
