from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ml_project.features.customers import (
    customer_segment_profile,
    label_customer_segments,
    segment_by_label,
)
REPORT_PATH = ROOT / "reports" / "rapport_final.html"


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def fmt_int(value: float | int) -> str:
    return f"{value:,.0f}".replace(",", " ")


def fmt_pct(value: float, digits: int = 2) -> str:
    return f"{value * 100:.{digits}f}%"


def table_rows(rows: list[list[str]]) -> str:
    return "\n".join(
        "<tr>" + "".join(f"<td>{cell}</td>" for cell in row) + "</tr>"
        for row in rows
    )


def build_report() -> str:
    fraud_metrics = load_json(ROOT / "reports" / "fraud_metrics.json")
    cluster_metrics = load_json(ROOT / "reports" / "customer_clustering_metrics.json")
    fraud_comparison = pd.read_csv(ROOT / "reports" / "fraud_model_comparison.csv")
    clustering_comparison = pd.read_csv(ROOT / "reports" / "clustering_model_comparison.csv")

    fraud_df = pd.read_csv(ROOT / "detection_fraude.csv", sep=";", usecols=["type", "amount", "isFraud"])
    fraud_count = int(fraud_df["isFraud"].sum())
    fraud_rate = fraud_count / len(fraud_df)
    type_summary = (
        fraud_df.groupby("type")
        .agg(transactions=("isFraud", "size"), fraudes=("isFraud", "sum"), montant_median=("amount", "median"))
        .assign(taux_fraude=lambda frame: frame["fraudes"] / frame["transactions"])
        .reset_index()
        .sort_values("taux_fraude", ascending=False)
    )

    segments = pd.read_csv(ROOT / "data" / "processed" / "customer_segments.csv")
    customer_summary = label_customer_segments(customer_segment_profile(segments))

    top_rate = type_summary.iloc[0]
    second_rate = type_summary.iloc[1]
    max_frauds = type_summary.sort_values("fraudes", ascending=False).iloc[0]
    zero_types = ", ".join(type_summary.loc[type_summary["fraudes"] == 0, "type"].astype(str).tolist())
    fraud_only = fraud_df[fraud_df["isFraud"] == 1]
    p95_amount = fraud_only["amount"].quantile(0.95)
    top25_share = fraud_only.sort_values("amount", ascending=False).head(25)["amount"].sum() / fraud_only["amount"].sum()
    premium = segment_by_label(customer_summary, "Clients premium")
    dormant = segment_by_label(customer_summary, "Clients dormants")
    digital = segment_by_label(customer_summary, "Digitaux et promotions")
    low_value = segment_by_label(customer_summary, "Clients economes")
    global_spend = segments["Total_Spend"].mean()
    global_response = segments["Response"].mean()

    type_rows = [
        [
            str(row.type),
            fmt_int(row.transactions),
            fmt_int(row.fraudes),
            fmt_pct(row.taux_fraude, 4),
            fmt_int(row.montant_median),
        ]
        for row in type_summary.itertuples(index=False)
    ]
    segment_rows = [
        [
            str(row.segment),
            str(row.label_metier),
            fmt_int(row.clients),
            fmt_int(row.revenu_median),
            fmt_int(row.depense_moyenne),
            fmt_pct(row.reponse_campagne, 1),
        ]
        for row in customer_summary.itertuples(index=False)
    ]
    fraud_model_rows = [
        [
            str(row.model),
            str(row.status),
            "" if pd.isna(row.accuracy) else fmt_pct(row.accuracy, 4),
            "" if pd.isna(row.precision) else fmt_pct(row.precision, 2),
            "" if pd.isna(row.recall) else fmt_pct(row.recall, 2),
            "" if pd.isna(row.f1) else fmt_pct(row.f1, 2),
            "" if pd.isna(row.roc_auc) else f"{row.roc_auc:.4f}",
        ]
        for row in fraud_comparison.itertuples(index=False)
    ]
    clustering_rows = [
        [
            str(row.model),
            str(row.clusters_found),
            str(row.noise_points),
            "" if pd.isna(row.silhouette) else f"{row.silhouette:.4f}",
            "" if pd.isna(row.davies_bouldin) else f"{row.davies_bouldin:.4f}",
        ]
        for row in clustering_comparison.itertuples(index=False)
    ]

    generated_at = datetime.now().strftime("%d/%m/%Y %H:%M")
    return f"""<!doctype html>
<html lang="fr">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Rapport final - Fraud Intelligence</title>
  <style>
    :root {{
      --bg: #f4f7fb;
      --surface: #ffffff;
      --ink: #101828;
      --muted: #475467;
      --line: #d0d5dd;
      --primary: #0f766e;
      --risk: #b42318;
      --good: #067647;
      --accent: #1d4ed8;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      background: var(--bg);
      color: var(--ink);
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      line-height: 1.55;
    }}
    main {{ max-width: 1120px; margin: 0 auto; padding: 34px 22px 56px; }}
    .hero {{
      background: var(--surface);
      border: 1px solid var(--line);
      border-left: 8px solid var(--primary);
      border-radius: 12px;
      padding: 34px;
      box-shadow: 0 16px 42px rgba(16, 24, 40, 0.08);
    }}
    .eyebrow {{
      color: var(--primary);
      font-size: 0.8rem;
      font-weight: 800;
      letter-spacing: 0.08em;
      text-transform: uppercase;
    }}
    h1 {{ font-size: clamp(2rem, 4vw, 3.2rem); line-height: 1.05; margin: 10px 0; }}
    h2 {{ font-size: 1.45rem; margin: 0 0 12px; }}
    h3 {{ font-size: 1.05rem; margin: 0 0 8px; }}
    p {{ color: var(--muted); margin: 0; }}
    .chips {{ display: flex; flex-wrap: wrap; gap: 10px; margin-top: 22px; }}
    .chip {{
      border: 1px solid color-mix(in srgb, var(--primary) 38%, var(--line));
      border-radius: 999px;
      color: var(--ink);
      font-weight: 760;
      padding: 7px 12px;
      background: color-mix(in srgb, var(--primary) 10%, white);
    }}
    .grid {{ display: grid; gap: 18px; margin-top: 22px; }}
    .kpis {{ grid-template-columns: repeat(auto-fit, minmax(210px, 1fr)); }}
    .two {{ grid-template-columns: repeat(auto-fit, minmax(320px, 1fr)); }}
    .card {{
      background: var(--surface);
      border: 1px solid var(--line);
      border-radius: 10px;
      padding: 22px;
      box-shadow: 0 2px 8px rgba(16, 24, 40, 0.04);
    }}
    .kpi {{ border-left: 5px solid var(--primary); min-height: 130px; }}
    .kpi.risk {{ border-left-color: var(--risk); }}
    .kpi.good {{ border-left-color: var(--good); }}
    .label {{ color: var(--muted); font-size: 0.76rem; font-weight: 800; text-transform: uppercase; }}
    .value {{ font-size: 2rem; font-weight: 850; margin-top: 8px; }}
    .detail {{ color: var(--muted); margin-top: 8px; }}
    section {{ margin-top: 28px; }}
    table {{ border-collapse: collapse; width: 100%; overflow: hidden; border-radius: 8px; }}
    th, td {{ border-bottom: 1px solid var(--line); padding: 11px 12px; text-align: left; }}
    th {{ background: #eef4ff; color: #1d2939; font-size: 0.78rem; text-transform: uppercase; }}
    tr:last-child td {{ border-bottom: 0; }}
    .insight {{
      background: #ecfdf3;
      border: 1px solid #abefc6;
      border-left: 6px solid var(--good);
      border-radius: 10px;
      padding: 20px;
    }}
    .decision {{
      background: #fffbeb;
      border: 1px solid #fedf89;
      border-left: 6px solid #b54708;
      border-radius: 10px;
      color: #344054;
      margin-top: 18px;
      padding: 16px 18px;
    }}
    ul {{ margin: 10px 0 0 20px; padding: 0; color: var(--muted); }}
    li {{ margin: 6px 0; }}
    footer {{ color: var(--muted); margin-top: 30px; text-align: center; font-size: 0.9rem; }}
    @media print {{
      body {{ background: white; }}
      main {{ max-width: none; padding: 18px; }}
      .card, .hero {{ box-shadow: none; }}
    }}
  </style>
</head>
<body>
  <main>
    <header class="hero">
      <div class="eyebrow">Rapport final entreprise</div>
      <h1>Fraud & Customer Intelligence</h1>
      <p>Solution complete de detection de fraude, segmentation client, scoring operationnel et dashboard decisionnel.</p>
      <div class="chips">
        <span class="chip">Dashboard Streamlit</span>
        <span class="chip">Modele fraude entraine</span>
        <span class="chip">Segmentation client</span>
        <span class="chip">Scoring CSV</span>
        <span class="chip">Depot GitHub deployable</span>
      </div>
    </header>

    <section class="grid kpis">
      <article class="card kpi"><div class="label">Transactions analysees</div><div class="value">{fmt_int(len(fraud_df))}</div><div class="detail">Historique complet</div></article>
      <article class="card kpi risk"><div class="label">Taux de fraude</div><div class="value">{fmt_pct(fraud_rate, 3)}</div><div class="detail">{fmt_int(fraud_count)} cas observes</div></article>
      <article class="card kpi good"><div class="label">Recall modele</div><div class="value">{fmt_pct(fraud_metrics.get("recall", 0), 2)}</div><div class="detail">Fraudes retrouvees</div></article>
      <article class="card kpi"><div class="label">Clients segmentes</div><div class="value">{fmt_int(len(segments))}</div><div class="detail">{segments["segment"].nunique()} profils metiers</div></article>
    </section>

    <section class="card insight">
      <h2>Lecture executive</h2>
      <p>Sur {fmt_int(len(fraud_df))} transactions, {fmt_int(fraud_count)} fraudes sont observees ({fmt_pct(fraud_rate, 3)}). Les fraudes historiques sont concentrees sur TRANSFER et CASH_OUT. Le modele retrouve environ {fmt_pct(fraud_metrics.get("recall", 0), 2)} des fraudes, ce qui permet de prioriser une file analyste courte par rapport au volume total.</p>
    </section>

    <section class="grid two">
      <article class="card">
        <h2>Performance fraude</h2>
        <table>
          <thead><tr><th>Metrique</th><th>Valeur</th></tr></thead>
          <tbody>
            <tr><td>Accuracy</td><td>{fmt_pct(fraud_metrics.get("accuracy", 0), 4)}</td></tr>
            <tr><td>Precision</td><td>{fmt_pct(fraud_metrics.get("precision", 0), 2)}</td></tr>
            <tr><td>Recall</td><td>{fmt_pct(fraud_metrics.get("recall", 0), 2)}</td></tr>
            <tr><td>F1-score</td><td>{fmt_pct(fraud_metrics.get("f1", 0), 2)}</td></tr>
            <tr><td>ROC-AUC</td><td>{fraud_metrics.get("roc_auc", 0):.4f}</td></tr>
          </tbody>
        </table>
      </article>
      <article class="card">
        <h2>Segmentation client</h2>
        <table>
          <thead><tr><th>Indicateur</th><th>Valeur</th></tr></thead>
          <tbody>
            <tr><td>Clients</td><td>{fmt_int(len(segments))}</td></tr>
            <tr><td>Segments retenus</td><td>{segments["segment"].nunique()}</td></tr>
            <tr><td>Silhouette</td><td>{cluster_metrics.get("silhouette", 0):.4f}</td></tr>
            <tr><td>Davies-Bouldin</td><td>{cluster_metrics.get("davies_bouldin", 0):.4f}</td></tr>
          </tbody>
        </table>
      </article>
    </section>

    <section class="card">
      <h2>Comparaison des modeles fraude</h2>
      <table>
        <thead><tr><th>Modele</th><th>Statut</th><th>Accuracy</th><th>Precision</th><th>Recall</th><th>F1</th><th>ROC-AUC</th></tr></thead>
        <tbody>{table_rows(fraud_model_rows)}</tbody>
      </table>
      <div class="decision">
        <strong>Decision.</strong> Random Forest, XGBoost et LightGBM donnent le meilleur compromis sur l'echantillon de comparaison. Le Random Forest est conserve pour la demo car il est robuste, interpretable par importance de variables et deja sauvegarde dans les artefacts de production.
      </div>
    </section>

    <section class="card">
      <h2>Comparaison des algorithmes de clustering</h2>
      <table>
        <thead><tr><th>Modele</th><th>Clusters trouves</th><th>Bruit</th><th>Silhouette</th><th>Davies-Bouldin</th></tr></thead>
        <tbody>{table_rows(clustering_rows)}</tbody>
      </table>
      <div class="decision">
        <strong>Decision.</strong> Gaussian Mixture obtient une bonne silhouette, mais K-Means a ete retenu pour la lisibilite, la stabilite et la facilite d'explication des segments metiers dans un contexte entreprise.
      </div>
    </section>

    <section class="card">
      <h2>Risque par type de transaction</h2>
      <table>
        <thead><tr><th>Type</th><th>Transactions</th><th>Fraudes</th><th>Taux fraude</th><th>Montant median</th></tr></thead>
        <tbody>{table_rows(type_rows)}</tbody>
      </table>
      <div class="decision">
        <strong>Decision.</strong> Appliquer un controle renforce sur {top_rate['type']}: taux de fraude {fmt_pct(top_rate['taux_fraude'], 4)}, environ {top_rate['taux_fraude'] / second_rate['taux_fraude']:.1f} fois {second_rate['type']}. {max_frauds['type']} porte le plus grand nombre de cas ({fmt_int(max_frauds['fraudes'])}). Les types {zero_types} n'ont aucun cas historique: monitoring plutot que blocage automatique.
      </div>
    </section>

    <section class="card">
      <h2>Profils clients</h2>
      <table>
        <thead><tr><th>Segment</th><th>Profil</th><th>Clients</th><th>Revenu median</th><th>Depense moyenne</th><th>Reponse campagne</th></tr></thead>
        <tbody>{table_rows(segment_rows)}</tbody>
      </table>
      <div class="decision">
        <strong>Decision.</strong> Le segment premium compte {fmt_int(premium['clients'])} clients ({fmt_pct(premium['clients'] / len(segments), 1)}) mais depense {premium['depense_moyenne'] / global_spend:.1f} fois la moyenne et repond a {fmt_pct(premium['reponse_campagne'], 1)} contre {fmt_pct(global_response, 1)} globalement. Les dormants ({fmt_int(dormant['clients'])} clients) gardent une forte depense moyenne: campagne de reactivation selective.
      </div>
    </section>

    <section class="grid two">
      <article class="card">
        <h2>Usage operationnel</h2>
        <ul>
          <li>Saisie manuelle d'une transaction pour decision immediate.</li>
          <li>Import CSV de gros volume avec controle de schema.</li>
          <li>Export d'un fichier score contenant probabilite, risque et action recommandee.</li>
          <li>Les 25 plus grosses fraudes representent {fmt_pct(top25_share, 1)} du montant frauduleux total: priorisation par montant recommandee.</li>
        </ul>
      </article>
      <article class="card">
        <h2>Actions recommandees</h2>
        <ul>
          <li>Fraude: revue prioritaire sur TRANSFER/CASH_OUT et sur les montants au-dessus de {fmt_int(p95_amount)}.</li>
          <li>Premium: fidelisation, avantages et service prioritaire.</li>
          <li>Dormants: reactivation controlee, car leur valeur reste elevee.</li>
          <li>Digitaux/promotions: campagnes web ciblees, avec controle du cout promotionnel.</li>
          <li>Economes: automatisation marketing a faible cout.</li>
        </ul>
      </article>
    </section>

    <footer>Rapport genere le {generated_at} - Fraud Intelligence</footer>
  </main>
</body>
</html>
"""


def main() -> None:
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(build_report(), encoding="utf-8")
    print(f"Rapport HTML genere: {REPORT_PATH}")


if __name__ == "__main__":
    main()
