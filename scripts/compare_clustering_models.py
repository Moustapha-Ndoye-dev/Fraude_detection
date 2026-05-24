from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from ml_project.config import PATHS
from ml_project.data.loaders import read_customer_data
from ml_project.models.clustering import fit_customer_clustering
from ml_project.models.selection import select_best_clustering_model
from ml_project.training.artifacts import persist_clustering_model

MODEL_ORDER = ["kmeans", "dbscan", "agglomerative", "gaussian_mixture"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Comparer les algorithmes de clustering client.")
    parser.add_argument("--clusters", type=int, default=4)
    parser.add_argument(
        "--skip-deploy",
        action="store_true",
        help="Ne pas sauvegarder automatiquement le meilleur modele.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    PATHS.ensure_outputs()
    df = read_customer_data()
    rows = []

    for model_name in MODEL_ORDER:
        started = time.perf_counter()
        result = fit_customer_clustering(df, model_name=model_name, n_clusters=args.clusters)
        elapsed = time.perf_counter() - started
        labels = pd.Series(result.labels)
        rows.append(
            {
                "model": model_name,
                "clusters_requested": args.clusters if model_name != "dbscan" else None,
                "clusters_found": labels.nunique(),
                "noise_points": int((labels == -1).sum()),
                "elapsed_seconds": round(elapsed, 2),
                **result.metrics,
            }
        )

    output = pd.DataFrame(rows)
    output_path = PATHS.report_dir / "clustering_model_comparison.csv"
    output.to_csv(output_path, index=False)
    print(output.to_string(index=False))
    print(f"Comparaison sauvegardee: {output_path}")

    if args.skip_deploy:
        return

    best_model = select_best_clustering_model(output)
    result = fit_customer_clustering(df, model_name=best_model, n_clusters=args.clusters)
    persist_clustering_model(result, df, comparison=output)

    print(f"Meilleur algorithme selectionne: {best_model}")
    print(f"Modele de production sauvegarde: {PATHS.model_dir / 'customer_clustering.joblib'}")


if __name__ == "__main__":
    main()
