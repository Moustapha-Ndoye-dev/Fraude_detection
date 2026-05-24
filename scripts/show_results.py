from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from ml_project.config import PATHS


LINE = "=" * 92


def money(value: float) -> str:
    return f"{value:,.0f}".replace(",", " ")


def pct(value: float) -> str:
    return f"{value * 100:.3f}%"


def section(title: str) -> None:
    print()
    print(LINE)
    print(title.upper())
    print(LINE)


def load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def print_artifacts() -> None:
    section("Artefacts disponibles")
    artifacts = [
        PATHS.model_dir / "fraud_pipeline.joblib",
        PATHS.model_dir / "customer_clustering.joblib",
        PATHS.report_dir / "fraud_metrics.json",
        PATHS.report_dir / "customer_clustering_metrics.json",
        PATHS.report_dir / "customer_k_scores.csv",
        PATHS.processed_dir / "customer_segments.csv",
    ]
    for path in artifacts:
        status = "OK" if path.exists() else "MANQUANT"
        size = f"{path.stat().st_size / 1024:.1f} Ko" if path.exists() else "-"
        print(f"{status:<8} {size:>12}  {path}")


def print_fraud_results() -> None:
    section("Detection de fraude - resultats modele")
    metrics = load_json(PATHS.report_dir / "fraud_metrics.json")
    if not metrics:
        print("Aucune metrique disponible. Lancer: python scripts/train_fraud.py --model random_forest")
        return

    for key in ["accuracy", "precision", "recall", "f1", "roc_auc"]:
        print(f"{key:<12}: {metrics[key]:.6f}")

    df = pd.read_csv(
        PATHS.fraud_data,
        sep=";",
        usecols=["type", "amount", "isFraud", "step"],
    )
    total = len(df)
    frauds = int(df["isFraud"].sum())
    print()
    print(f"Transactions totales : {total:,}".replace(",", " "))
    print(f"Fraudes detectees    : {frauds:,}".replace(",", " "))
    print(f"Taux de fraude       : {pct(frauds / total)}")
    print(f"Montant median       : {money(df['amount'].median())}")
    print(f"Montant median fraude: {money(df.loc[df['isFraud'] == 1, 'amount'].median())}")

    by_type = (
        df.groupby("type")
        .agg(transactions=("isFraud", "size"), fraudes=("isFraud", "sum"), montant_median=("amount", "median"))
        .assign(taux_fraude=lambda x: x["fraudes"] / x["transactions"])
        .sort_values("taux_fraude", ascending=False)
    )
    by_type["taux_fraude"] = by_type["taux_fraude"].map(lambda x: f"{x * 100:.4f}%")
    by_type["montant_median"] = by_type["montant_median"].map(money)
    print()
    print("Fraude par type de transaction")
    print(by_type.to_string())

    comparison_path = PATHS.report_dir / "fraud_model_comparison.csv"
    if comparison_path.exists():
        print()
        print("Comparaison des modeles fraude")
        print(pd.read_csv(comparison_path).to_string(index=False))

    importance_path = PATHS.report_dir / "fraud_feature_importance.csv"
    if importance_path.exists():
        print()
        print("Top variables importantes")
        print(pd.read_csv(importance_path).head(10).to_string(index=False))

    shap_path = PATHS.report_dir / "fraud_shap_importance.csv"
    if shap_path.exists():
        print()
        print("Top variables SHAP")
        print(pd.read_csv(shap_path).head(10).to_string(index=False))


def label_segments(profiles: pd.DataFrame) -> pd.DataFrame:
    profiles = profiles.copy()
    profiles["label_metier"] = "Clients economes"
    profiles.loc[profiles["depense_moyenne"].idxmax(), "label_metier"] = "Clients premium"
    profiles.loc[profiles["recence_moyenne"].idxmax(), "label_metier"] = "Clients dormants"
    web_promo_score = profiles["achats_web_moyens"].rank() + profiles["achats_promo_moyens"].rank()
    profiles.loc[web_promo_score.idxmax(), "label_metier"] = "Digitaux et promotions"
    return profiles


def print_customer_results() -> None:
    section("Segmentation client - resultats clustering")
    metrics = load_json(PATHS.report_dir / "customer_clustering_metrics.json")
    if metrics:
        for key, value in metrics.items():
            print(f"{key:<16}: {value:.6f}")
    else:
        print("Aucune metrique disponible. Lancer: python scripts/train_customer_clustering.py --search-k")

    k_path = PATHS.report_dir / "customer_k_scores.csv"
    if k_path.exists():
        print()
        print("Comparaison des nombres de clusters")
        print(pd.read_csv(k_path).to_string(index=False))

    clustering_comparison_path = PATHS.report_dir / "clustering_model_comparison.csv"
    if clustering_comparison_path.exists():
        print()
        print("Comparaison des algorithmes de clustering")
        print(pd.read_csv(clustering_comparison_path).to_string(index=False))

    segments_path = PATHS.processed_dir / "customer_segments.csv"
    if not segments_path.exists():
        return

    segments = pd.read_csv(segments_path)
    profiles = (
        segments.groupby("segment")
        .agg(
            clients=("ID", "size"),
            revenu_median=("Income", "median"),
            depense_moyenne=("Total_Spend", "mean"),
            achats_moyens=("Total_Purchases", "mean"),
            achats_web_moyens=("NumWebPurchases", "mean"),
            achats_promo_moyens=("NumDealsPurchases", "mean"),
            recence_moyenne=("Recency", "mean"),
            reponse_campagne=("Response", "mean"),
        )
        .sort_index()
    )
    profiles = label_segments(profiles)
    display = profiles.copy()
    for column in ["revenu_median", "depense_moyenne"]:
        display[column] = display[column].map(money)
    for column in ["achats_moyens", "achats_web_moyens", "achats_promo_moyens", "recence_moyenne"]:
        display[column] = display[column].map(lambda x: f"{x:.2f}")
    display["reponse_campagne"] = display["reponse_campagne"].map(pct)

    print()
    print("Profils de segments")
    print(display.to_string())


def print_demo_commands() -> None:
    section("Commandes de demo")
    print("Dashboard : http://127.0.0.1:8501")
    print("API       : http://127.0.0.1:8000/docs")
    print()
    print("Relancer API       : python -m uvicorn api.main:app --reload")
    print("Relancer dashboard : python -m streamlit run dashboard/app.py")
    print("Revoir resultats   : python scripts/show_results.py")


def main() -> None:
    print(LINE)
    print("PROJET ML - RESTITUTION TERMINAL ENTREPRISE")
    print(LINE)
    print_artifacts()
    print_fraud_results()
    print_customer_results()
    print_demo_commands()


if __name__ == "__main__":
    main()
