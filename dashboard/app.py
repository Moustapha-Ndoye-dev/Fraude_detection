from __future__ import annotations

import json
import sys
from io import BytesIO
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from ml_project.config import PATHS
from ml_project.features.customers import (
    customer_segment_profile,
    label_customer_segments,
    segment_by_label,
)
from ml_project.serving.fraud_scoring import (
    expected_fraud_schema,
    score_fraud_dataframe,
    summarize_scoring,
    template_fraud_csv,
    validate_fraud_input,
)


st.set_page_config(
    page_title="Fraud & Customer Intelligence",
    layout="wide",
    initial_sidebar_state="expanded",
)

from react_widgets import react_hero, react_insight, react_kpis


def fmt_int(value: float | int) -> str:
    return f"{value:,.0f}".replace(",", " ")


def fmt_money(value: float | int) -> str:
    return f"{value:,.0f}".replace(",", " ")


def fmt_pct(value: float, digits: int = 2) -> str:
    return f"{value * 100:.{digits}f}%"


def page_header(
    title: str,
    subtitle: str,
    *,
    eyebrow: str = "Fraud & Customer Intelligence",
    chips: list[str] | None = None,
) -> None:
    react_hero(title, subtitle, eyebrow=eyebrow, chips=chips or [])


REPORT_PAGES = {
    "introduction": {
        "menu": "1 · Introduction et contexte",
        "eyebrow": "Rapport d'analyse · M2 CDSD",
        "title": "Fraud & Customer Intelligence",
        "subtitle": (
            "Projet de machine learning couvrant la detection de fraude bancaire, "
            "la segmentation client et une premiere architecture MLOps. Ce tableau de bord "
            "restitue les resultats du rapport final: plus d'un million de transactions analysees, "
            "modeles compares automatiquement, profils marketing actionnables et scoring operationnel."
        ),
        "chips": ["Detection de fraude", "Segmentation client", "Scoring CSV", "MLOps"],
    },
    "fraude": {
        "menu": "2 · Detection de fraude bancaire",
        "eyebrow": "Chapitre 3 · Analyse des transactions",
        "title": "Detection de fraude bancaire",
        "subtitle": (
            "Analyse exploratoire des transactions, comparaison des modeles supervises "
            "et lecture metier des signaux de risque. Les fraudes historiques se concentrent "
            "sur les virements et retraits: ce chapitre aide a prioriser les controles "
            "par type, periode et montant."
        ),
        "chips": ["EDA transactions", "Modele retenu", "Metriques F1 / Recall", "Priorisation risque"],
    },
    "scoring": {
        "menu": "3 · Scoring operationnel",
        "eyebrow": "Chapitre 5 · Mise en production",
        "title": "Scoring en conditions reelles",
        "subtitle": (
            "Demonstration du service de scoring: saisie unitaire d'une transaction ou import "
            "CSV en volume. Chaque ligne est validee, scoree et traduite en niveau de risque "
            "et action recommandee pour les equipes conformite et operations."
        ),
        "chips": ["API FastAPI", "Validation schema", "Export CSV score", "Seuil configurable"],
    },
    "segmentation": {
        "menu": "4 · Segmentation client",
        "eyebrow": "Chapitre 4 · Marketing analytique",
        "title": "Segmentation intelligente des clients",
        "subtitle": (
            "Clustering non supervise de la base client pour identifier des profils homogenes: "
            "premium, dormants, sensibles aux promotions et economes. Chaque segment ouvre "
            "des recommandations distinctes en fidelisation, reactivation et automatisation marketing."
        ),
        "chips": ["Clustering", "Profils metiers", "Silhouette", "Actions marketing"],
    },
}


def report_header(page_key: str) -> None:
    section = REPORT_PAGES[page_key]
    page_header(
        section["title"],
        section["subtitle"],
        eyebrow=section["eyebrow"],
        chips=section["chips"],
    )


def business_note(title: str, body: str) -> None:
    react_insight(title, body)


@st.cache_data(show_spinner=False)
def load_json(path: str) -> dict:
    file_path = Path(path)
    if not file_path.exists():
        return {}
    return json.loads(file_path.read_text(encoding="utf-8"))


@st.cache_data(show_spinner="Chargement des transactions...")
def load_fraud_data() -> pd.DataFrame:
    if not PATHS.fraud_data.exists():
        return pd.DataFrame()
    return pd.read_csv(
        PATHS.fraud_data,
        sep=";",
        usecols=[
            "step",
            "type",
            "amount",
            "oldbalanceOrg",
            "newbalanceOrig",
            "oldbalanceDest",
            "newbalanceDest",
            "isFraud",
        ],
    )


@st.cache_data(show_spinner="Chargement des segments clients...")
def load_customer_segments() -> pd.DataFrame:
    path = PATHS.processed_dir / "customer_segments.csv"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


@st.cache_data(show_spinner=False)
def load_k_scores() -> pd.DataFrame:
    path = PATHS.report_dir / "customer_k_scores.csv"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


@st.cache_resource(show_spinner=False)
def load_fraud_model():
    path = PATHS.model_dir / "fraud_pipeline.joblib"
    if not path.exists():
        return None
    return joblib.load(path)


def scoring_ready() -> bool:
    return (PATHS.model_dir / "fraud_pipeline.joblib").exists()


def missing_asset(message: str) -> None:
    st.error(message)
    st.info(
        "En production Streamlit, les fichiers de donnees et les modeles doivent etre presents dans le depot GitHub "
        "ou fournis par un stockage externe. Le dashboard reste disponible, mais cette page attend ces artefacts."
    )


def artifact_status() -> dict[str, bool]:
    return {
        "Scoring fraude": (PATHS.model_dir / "fraud_pipeline.joblib").exists(),
        "Segmentation clients": (PATHS.model_dir / "customer_clustering.joblib").exists(),
        "Indicateurs fraude": (PATHS.report_dir / "fraud_metrics.json").exists(),
        "Fichier segments": (PATHS.processed_dir / "customer_segments.csv").exists(),
    }


def fraud_type_summary(fraud_df: pd.DataFrame) -> pd.DataFrame:
    return (
        fraud_df.groupby("type")
        .agg(transactions=("isFraud", "size"), fraudes=("isFraud", "sum"), montant_median=("amount", "median"))
        .assign(taux_fraude=lambda frame: frame["fraudes"] / frame["transactions"])
        .reset_index()
        .sort_values("taux_fraude", ascending=False)
    )


def fraud_step_summary(fraud_df: pd.DataFrame) -> pd.DataFrame:
    return (
        fraud_df.groupby("step")
        .agg(transactions=("isFraud", "size"), fraudes=("isFraud", "sum"))
        .assign(taux_fraude=lambda frame: frame["fraudes"] / frame["transactions"])
        .reset_index()
    )


def fraud_executive_insight(fraud_df: pd.DataFrame, fraud_metrics: dict) -> str:
    total_transactions = len(fraud_df)
    total_frauds = int(fraud_df["isFraud"].sum())
    summary = fraud_type_summary(fraud_df)
    risky = summary[summary["fraudes"] > 0]
    risky_types = " et ".join(risky["type"].astype(str).tolist())
    risky_share = risky["fraudes"].sum() / total_frauds if total_frauds else 0
    recall = fraud_metrics.get("recall", 0)
    detected_estimate = round(total_frauds * recall)
    missed_estimate = max(total_frauds - detected_estimate, 0)
    return (
        f"Sur {fmt_int(total_transactions)} transactions, {fmt_int(total_frauds)} fraudes sont observees "
        f"({fmt_pct(total_frauds / total_transactions, 3)}). {fmt_pct(risky_share, 1)} des fraudes historiques "
        f"sont concentrees sur {risky_types}. Avec un recall de {fmt_pct(recall, 2)}, le modele retrouve environ "
        f"{fmt_int(detected_estimate)} fraudes et en laisserait environ {fmt_int(missed_estimate)} hors alerte "
        "sur un volume comparable: la decision prioritaire est donc de traiter vite les alertes critiques, puis "
        "d'analyser les cas manques pour ajuster le seuil."
    )


def executive_business_insight(fraud_df: pd.DataFrame, segments: pd.DataFrame, fraud_metrics: dict) -> str:
    total_frauds = int(fraud_df["isFraud"].sum())
    summary = fraud_type_summary(fraud_df)
    profile = label_customer_segments(customer_segment_profile(segments))
    premium = segment_by_label(profile, "Clients premium")
    premium_share = premium["clients"] / len(segments)
    precision = fraud_metrics.get("precision", 0)
    top_fraud_type = summary.iloc[0]
    return (
        f"Decision dirigeant: concentrer les controles fraude sur {top_fraud_type['type']} et CASH_OUT, car ces deux types "
        f"portent {fmt_int(total_frauds)} fraudes observees. La precision historique de {fmt_pct(precision, 2)} limite "
        "la perte de temps analyste sur les alertes. Cote client, le segment premium ne represente que "
        f"{fmt_pct(premium_share, 1)} de la base mais depense en moyenne {fmt_money(premium['depense_moyenne'])}; "
        "il doit etre protege par des actions de fidelisation plutot que des campagnes de masse."
    )


def fraud_type_business_insight(summary: pd.DataFrame) -> str:
    top_rate = summary.iloc[0]
    max_frauds = summary.sort_values("fraudes", ascending=False).iloc[0]
    second_rate = summary.iloc[1]
    ratio = top_rate["taux_fraude"] / second_rate["taux_fraude"] if second_rate["taux_fraude"] else 0
    zero_types = ", ".join(summary.loc[summary["fraudes"] == 0, "type"].astype(str).tolist())
    return (
        f"Decision: appliquer un controle renforce sur {top_rate['type']}. Son taux de fraude est "
        f"{fmt_pct(top_rate['taux_fraude'], 4)}, soit environ {ratio:.1f} fois le taux de {second_rate['type']}. "
        f"{max_frauds['type']} porte le plus grand nombre de cas ({fmt_int(max_frauds['fraudes'])} fraudes), donc il doit "
        "aussi alimenter la file analyste. Les types "
        f"{zero_types} n'ont aucun cas de fraude dans l'historique: ne pas les bloquer par regle fixe, mais les garder "
        "en monitoring pour detecter un changement de comportement."
    )


def fraud_timeline_business_insight(step_summary: pd.DataFrame) -> str:
    by_count = step_summary.sort_values("fraudes", ascending=False)
    peak = by_count.iloc[0]
    significant = step_summary[step_summary["transactions"] >= 1000].sort_values("taux_fraude", ascending=False).iloc[0]
    active_steps = int((step_summary["fraudes"] > 0).sum())
    return (
        f"Pic brut: le step {int(peak['step'])} concentre {fmt_int(peak['fraudes'])} fraudes sur "
        f"{fmt_int(peak['transactions'])} transactions. Sur les fenetres avec au moins 1 000 transactions, le step "
        f"{int(significant['step'])} est le plus sensible ({fmt_int(significant['fraudes'])} fraudes, "
        f"{fmt_pct(significant['taux_fraude'], 3)}). Decision: declencher une alerte operationnelle lorsqu'une fenetre "
        f"depasse 20 fraudes ou franchit 1% de taux de fraude. Les fraudes apparaissent sur {active_steps} steps, donc "
        "la surveillance doit etre continue, pas seulement ponctuelle."
    )


def fraud_amount_business_insight(fraud_only: pd.DataFrame) -> str:
    median_amount = fraud_only["amount"].median()
    p95_amount = fraud_only["amount"].quantile(0.95)
    top25_sum = fraud_only.sort_values("amount", ascending=False).head(25)["amount"].sum()
    total_amount = fraud_only["amount"].sum()
    return (
        f"Le montant median d'une fraude est {fmt_money(median_amount)}, mais les 5% les plus eleves commencent autour de "
        f"{fmt_money(p95_amount)}. Les 25 plus grosses fraudes representent {fmt_pct(top25_sum / total_amount, 1)} du montant "
        "frauduleux total. Decision: creer une file prioritaire 'impact financier' pour les alertes a tres gros montant, "
        "meme si la probabilite n'est pas la seule variable de tri."
    )


def scoring_business_insight(summary: dict) -> str:
    manual_review = summary.get("manual_review_transactions", 0)
    return (
        f"Avec le seuil actuel ({summary['threshold']:.2f}), le fichier genere {fmt_int(summary['predicted_frauds'])} alertes "
        f"sur {fmt_int(summary['rows'])} lignes, soit {fmt_pct(summary['predicted_fraud_rate'], 2)} du volume. "
        f"{fmt_int(summary['critical_transactions'])} transactions sont critiques et {fmt_int(manual_review)} sont en revue manuelle. "
        "Decision: traiter les critiques en premier; si la capacite analyste est inferieure au nombre d'alertes, relever le seuil "
        "ou prioriser par montant."
    )


def risk_distribution_insight(risk_counts: pd.DataFrame) -> str:
    counts = dict(zip(risk_counts["risk_band"], risk_counts["transactions"]))
    high_priority = counts.get("HIGH", 0) + counts.get("CRITICAL", 0)
    low = counts.get("LOW", 0)
    return (
        f"La file prioritaire contient {fmt_int(high_priority)} transactions HIGH/CRITICAL; {fmt_int(low)} lignes sont LOW et "
        "peuvent rester en validation automatique. Decision: dimensionner l'equipe analyste sur HIGH/CRITICAL, pas sur le volume "
        "total du fichier."
    )


def customer_executive_insight(profile: pd.DataFrame, segments: pd.DataFrame) -> str:
    premium = segment_by_label(profile, "Clients premium")
    dormant = segment_by_label(profile, "Clients dormants")
    global_spend = segments["Total_Spend"].mean()
    global_response = segments["Response"].mean()
    return (
        f"Le segment premium compte {fmt_int(premium['clients'])} clients ({fmt_pct(premium['clients'] / len(segments), 1)}) "
        f"avec une depense moyenne de {fmt_money(premium['depense_moyenne'])}, soit {premium['depense_moyenne'] / global_spend:.1f} fois "
        f"la moyenne. Son taux de reponse est {fmt_pct(premium['reponse_campagne'], 1)} contre {fmt_pct(global_response, 1)} globalement. "
        f"Les dormants restent importants: {fmt_int(dormant['clients'])} clients, depense moyenne {fmt_money(dormant['depense_moyenne'])}, "
        "mais recence la plus haute. Decision: fideliser les premium et reactiver les dormants avec une offre selective."
    )


def customer_profile_business_insight(profile: pd.DataFrame, segments: pd.DataFrame) -> str:
    premium = segment_by_label(profile, "Clients premium")
    dormant = segment_by_label(profile, "Clients dormants")
    digital = segment_by_label(profile, "Digitaux et promotions")
    low_value = segment_by_label(profile, "Clients economes")
    return (
        f"Plan d'action: 1) Premium: {fmt_int(premium['clients'])} clients, reponse {fmt_pct(premium['reponse_campagne'], 1)}; "
        "programme fidelite et service prioritaire. 2) Dormants: "
        f"{fmt_int(dormant['clients'])} clients avec forte depense moyenne ({fmt_money(dormant['depense_moyenne'])}); campagne de reactivation "
        "a cout controle. 3) Digitaux/promotions: "
        f"{fmt_int(digital['clients'])} clients, {digital['achats_web_moyens']:.1f} achats web et {digital['achats_promo_moyens']:.1f} achats promo moyens; "
        "campagnes digitales ciblees. 4) Economes: "
        f"{fmt_int(low_value['clients'])} clients ({fmt_pct(low_value['clients'] / len(segments), 1)}) a faible depense; automatiser les communications."
    )


def spend_business_insight(profile: pd.DataFrame) -> str:
    value = profile.assign(valeur_segment=profile["clients"] * profile["depense_moyenne"])
    top_value = value.sort_values("valeur_segment", ascending=False).iloc[0]
    premium = segment_by_label(profile, "Clients premium")
    low_value = segment_by_label(profile, "Clients economes")
    return (
        f"Le plus gros potentiel de chiffre d'affaires vient du segment {int(top_value['segment'])} ({top_value['label_metier']}): "
        f"{fmt_pct(top_value['valeur_segment'] / value['valeur_segment'].sum(), 1)} de la depense totale estimee. "
        f"Le premium depense {premium['depense_moyenne'] / low_value['depense_moyenne']:.1f} fois plus qu'un client econome. "
        "Decision: suivre la retention et la valeur par segment, pas seulement le nombre de clients."
    )


def response_business_insight(profile: pd.DataFrame, segments: pd.DataFrame) -> str:
    best = profile.sort_values("reponse_campagne", ascending=False).iloc[0]
    global_response = segments["Response"].mean()
    return (
        f"Le segment {int(best['segment'])} ({best['label_metier']}) repond a {fmt_pct(best['reponse_campagne'], 1)}, "
        f"soit {best['reponse_campagne'] / global_response:.1f} fois la moyenne globale. Decision: utiliser ce segment pour les campagnes "
        "a objectif conversion; pour les segments sous la moyenne, reduire la pression commerciale et tester des messages differents."
    )


def cluster_choice_business_insight(k_scores: pd.DataFrame, selected_k: int) -> str:
    if k_scores.empty:
        return "Le nombre de segments doit etre valide par les equipes metier avec les performances de campagne."
    best = k_scores.sort_values("silhouette", ascending=False).iloc[0]
    selected = k_scores.loc[k_scores["k"] == selected_k]
    selected_score = selected.iloc[0]["silhouette"] if not selected.empty else 0
    return (
        f"Le meilleur score silhouette statistique est obtenu avec k={int(best['k'])} ({best['silhouette']:.4f}), mais k={selected_k} "
        f"est retenu avec une silhouette de {selected_score:.4f} car il produit des groupes plus actionnables commercialement. "
        "Decision: garder k=4 pour la demo entreprise, puis confirmer ce choix par les taux de conversion reels."
    )


def render_sidebar() -> str:
    st.sidebar.title("Rapport interactif")
    st.sidebar.caption("Fraud & Customer Intelligence — restitution des analyses")
    page = st.sidebar.radio(
        "Sections du rapport",
        [section["menu"] for section in REPORT_PAGES.values()],
    )
    st.sidebar.divider()
    st.sidebar.caption("Disponibilite des livrables")
    st.sidebar.write(f"Moteur de scoring: {'Pret' if scoring_ready() else 'Modele manquant'}")
    for name, ok in artifact_status().items():
        st.sidebar.write(f"{name}: {'Pret' if ok else 'Manquant'}")
    report_path = PATHS.report_dir / "rapport_final.html"
    if report_path.exists():
        st.sidebar.download_button(
            "Telecharger le rapport HTML",
            data=report_path.read_bytes(),
            file_name="rapport_final.html",
            mime="text/html",
            width="stretch",
        )
    return page


def render_executive_page(fraud_df: pd.DataFrame, segments: pd.DataFrame, fraud_metrics: dict) -> None:
    if fraud_df.empty:
        missing_asset("Donnees de fraude indisponibles: fichier detection_fraude.csv absent du deploiement.")
        return
    if segments.empty:
        missing_asset("Segments clients indisponibles: fichier data/processed/customer_segments.csv absent du deploiement.")
        return
    total_transactions = len(fraud_df)
    total_frauds = int(fraud_df["isFraud"].sum())
    profile = label_customer_segments(customer_segment_profile(segments))
    report_header("introduction")

    react_kpis(
        [
            {"label": "Transactions analysees", "value": fmt_int(total_transactions), "detail": "Historique complet"},
            {
                "label": "Taux de fraude",
                "value": fmt_pct(total_frauds / total_transactions, 3),
                "detail": f"{fmt_int(total_frauds)} cas observes",
                "tone": "risk",
            },
            {
                "label": "Recall modele",
                "value": fmt_pct(fraud_metrics.get("recall", 0), 2),
                "detail": "Fraudes retrouvees",
                "tone": "good",
            },
            {
                "label": "Clients segmentes",
                "value": fmt_int(len(segments)),
                "detail": f"{profile['segment'].nunique()} profils metiers",
            },
        ]
    )

    left, right = st.columns(2)
    with left:
        st.subheader("Risque fraude par type")
        type_summary = fraud_type_summary(fraud_df)
        st.bar_chart(type_summary.set_index("type")["taux_fraude"])

    with right:
        st.subheader("Segments clients")
        st.bar_chart(profile.set_index("segment")["clients"])

    business_note(
        "Lecture executive.",
        executive_business_insight(fraud_df, segments, fraud_metrics),
    )


def render_fraud_page(fraud_df: pd.DataFrame, fraud_metrics: dict) -> None:
    if fraud_df.empty:
        missing_asset("Donnees de fraude indisponibles: fichier detection_fraude.csv absent du deploiement.")
        return
    total_transactions = len(fraud_df)
    total_frauds = int(fraud_df["isFraud"].sum())
    fraud_only = fraud_df[fraud_df["isFraud"] == 1]
    report_header("fraude")

    react_kpis(
        [
            {"label": "Transactions", "value": fmt_int(total_transactions), "detail": "Base analysee"},
            {"label": "Fraudes observees", "value": fmt_int(total_frauds), "detail": "Cas confirmes", "tone": "risk"},
            {"label": "Taux historique", "value": fmt_pct(total_frauds / total_transactions, 3), "detail": "Risque global"},
            {"label": "F1-score", "value": fmt_pct(fraud_metrics.get("f1", 0), 2), "detail": "Precision / recall", "tone": "good"},
            {"label": "ROC-AUC", "value": f"{fraud_metrics.get('roc_auc', 0):.4f}", "detail": "Separation modele", "tone": "good"},
        ]
    )

    left, right = st.columns(2)
    with left:
        st.subheader("Fraude par type de transaction")
        summary = fraud_type_summary(fraud_df)
        display = summary.copy()
        display["taux_fraude"] = display["taux_fraude"].map(lambda value: f"{value * 100:.4f}%")
        display["montant_median"] = display["montant_median"].map(fmt_money)
        st.dataframe(display, width="stretch", hide_index=True)
        business_note(
            "Decision par type.",
            fraud_type_business_insight(summary),
        )

    with right:
        st.subheader("Fraudes dans le temps")
        step_summary = fraud_step_summary(fraud_df)
        step_data = step_summary.set_index("step")["fraudes"]
        st.line_chart(step_data)
        business_note(
            "Decision temporelle.",
            fraud_timeline_business_insight(step_summary),
        )

    st.subheader("Transactions frauduleuses les plus importantes")
    st.dataframe(fraud_only.sort_values("amount", ascending=False).head(25), width="stretch", hide_index=True)
    business_note(
        "Decision par impact financier.",
        fraud_amount_business_insight(fraud_only),
    )


def risk_business_text(risk_band: str, probability: float) -> str:
    if risk_band == "CRITICAL":
        return f"Risque critique ({fmt_pct(probability, 2)}). Bloquer temporairement ou envoyer en revue analyste."
    if risk_band == "HIGH":
        return f"Risque eleve ({fmt_pct(probability, 2)}). Controler avant validation."
    if risk_band == "MEDIUM":
        return f"Risque moyen ({fmt_pct(probability, 2)}). Valider avec surveillance renforcee."
    return f"Risque faible ({fmt_pct(probability, 2)}). Validation automatique possible."


def render_scored_results(scored: pd.DataFrame, threshold: float, delimiter: str, export_name: str) -> None:
    summary = summarize_scoring(scored, threshold=threshold)
    react_kpis(
        [
            {"label": "Lignes scorees", "value": fmt_int(summary["rows"]), "detail": "Transactions controlees"},
            {
                "label": "Alertes fraude",
                "value": fmt_int(summary["predicted_frauds"]),
                "detail": fmt_pct(summary["predicted_fraud_rate"], 2),
                "tone": "risk",
            },
            {"label": "Risque moyen", "value": fmt_pct(summary["average_probability"], 2), "detail": "Probabilite moyenne"},
            {"label": "Critiques", "value": fmt_int(summary["critical_transactions"]), "detail": "Priorite analyste", "tone": "attention"},
        ]
    )

    business_note(
        "Decision de seuil.",
        scoring_business_insight(summary),
    )

    risk_order = ["LOW", "MEDIUM", "HIGH", "CRITICAL"]
    risk_counts = (
        scored["risk_band"]
        .value_counts()
        .reindex(risk_order, fill_value=0)
        .rename_axis("risk_band")
        .reset_index(name="transactions")
    )
    st.subheader("Repartition des niveaux de risque")
    st.bar_chart(risk_counts.set_index("risk_band")["transactions"])
    business_note(
        "Priorisation de la file analyste.",
        risk_distribution_insight(risk_counts),
    )

    st.subheader("Transactions les plus risquees")
    top_risk = scored.sort_values("fraud_probability", ascending=False).head(50)
    st.dataframe(top_risk, width="stretch", hide_index=True)

    output = BytesIO()
    scored.to_csv(output, index=False, sep=delimiter)
    st.download_button(
        "Exporter le fichier score",
        data=output.getvalue(),
        file_name=export_name,
        mime="text/csv",
        width="stretch",
    )


def render_operational_scoring_page() -> None:
    report_header("scoring")
    model = load_fraud_model()
    if model is None:
        st.error("Le service de scoring n'est pas pret: le modele fraude est introuvable.")
        return

    react_kpis(
        [
            {"label": "Service de scoring", "value": "Pret", "detail": "Transaction et batch CSV", "tone": "good"},
            {"label": "Controle qualite", "value": "Strict", "detail": "Schema et types verifies"},
            {"label": "Sortie metier", "value": "Action", "detail": "Validation, revue ou blocage"},
        ]
    )

    transaction_tab, csv_tab, schema_tab = st.tabs(["Saisie transaction", "Import CSV", "Format attendu"])

    with transaction_tab:
        st.subheader("Saisir une transaction")
        threshold = st.slider("Seuil de decision fraude", 0.0, 1.0, 0.5, 0.01, key="single_threshold")
        with st.form("single_transaction_form"):
            col1, col2, col3 = st.columns(3)
            with col1:
                step = st.number_input("Step", min_value=0, value=1, step=1)
                tx_type = st.selectbox("Type de transaction", ["TRANSFER", "CASH_OUT", "PAYMENT", "CASH_IN", "DEBIT"])
                amount = st.number_input("Montant", min_value=0.0, value=181.0, step=100.0)
            with col2:
                oldbalance_org = st.number_input("Solde emetteur avant", min_value=0.0, value=181.0, step=100.0)
                newbalance_orig = st.number_input("Solde emetteur apres", min_value=0.0, value=0.0, step=100.0)
                name_orig = st.text_input("Identifiant emetteur", value="C1305486145")
            with col3:
                oldbalance_dest = st.number_input("Solde destinataire avant", min_value=0.0, value=0.0, step=100.0)
                newbalance_dest = st.number_input("Solde destinataire apres", min_value=0.0, value=0.0, step=100.0)
                name_dest = st.text_input("Identifiant destinataire", value="C553264065")
            submitted = st.form_submit_button("Scorer la transaction", type="primary", width="stretch")

        if submitted:
            payload = pd.DataFrame(
                [
                    {
                        "step": step,
                        "type": tx_type,
                        "amount": amount,
                        "nameOrig": name_orig,
                        "oldbalanceOrg": oldbalance_org,
                        "newbalanceOrig": newbalance_orig,
                        "nameDest": name_dest,
                        "oldbalanceDest": oldbalance_dest,
                        "newbalanceDest": newbalance_dest,
                        "isFlaggedFraud": 0,
                    }
                ]
            )
            validation = validate_fraud_input(payload, strict=True)
            if not validation.is_valid:
                st.error("Transaction rejetee: certaines valeurs ne sont pas conformes.")
                for error in validation.errors:
                    st.write(f"- {error}")
            else:
                scored = score_fraud_dataframe(model, validation.dataframe, threshold=threshold)
                result = scored.iloc[0]
                react_kpis(
                    [
                        {
                            "label": "Decision",
                            "value": "Alerte" if int(result["predicted_isFraud"]) else "Acceptee",
                            "detail": str(result["recommended_action"]),
                            "tone": "risk" if int(result["predicted_isFraud"]) else "good",
                        },
                        {
                            "label": "Probabilite fraude",
                            "value": fmt_pct(float(result["fraud_probability"]), 2),
                            "detail": "Score modele",
                        },
                        {"label": "Niveau de risque", "value": str(result["risk_band"]), "detail": "Priorisation metier"},
                    ]
                )
                business_note("Interpretation.", risk_business_text(str(result["risk_band"]), float(result["fraud_probability"])))
                st.dataframe(scored, width="stretch", hide_index=True)

    with csv_tab:
        st.subheader("Importer un fichier de transactions")
        left, right = st.columns(2)
        with left:
            delimiter = st.selectbox("Separateur CSV", [";", ","], index=0, key="csv_delimiter")
            strict = st.checkbox("Rejeter les colonnes non attendues", value=True, key="csv_strict")
        with right:
            threshold = st.slider("Seuil de decision fraude", 0.0, 1.0, 0.5, 0.01, key="csv_threshold")

        uploaded_file = st.file_uploader("Importer un CSV de transactions", type=["csv"], key="transaction_csv")
        if uploaded_file is None:
            st.warning("Aucun fichier importe. Le fichier doit respecter le format attendu.")
        else:
            file_bytes = uploaded_file.getvalue()
            try:
                input_df = pd.read_csv(BytesIO(file_bytes), sep=delimiter)
            except Exception as exc:
                st.error(f"Lecture CSV impossible: {exc}")
                return

            validation = validate_fraud_input(input_df, strict=strict)
            if not validation.is_valid:
                st.error("Fichier rejete: le schema ou les valeurs ne sont pas conformes.")
                for error in validation.errors:
                    st.write(f"- {error}")
                return

            for warning in validation.warnings:
                st.warning(warning)

            react_kpis(
                [
                    {"label": "Fichier accepte", "value": fmt_int(len(validation.dataframe)), "detail": "Lignes valides", "tone": "good"},
                    {"label": "Colonnes controlees", "value": fmt_int(len(validation.dataframe.columns)), "detail": "Schema attendu"},
                    {"label": "Pret a scorer", "value": "Oui", "detail": uploaded_file.name},
                ]
            )

            st.subheader("Apercu des donnees validees")
            st.dataframe(validation.dataframe.head(20), width="stretch", hide_index=True)
            business_note(
                "Decision qualite.",
                f"Le fichier contient {fmt_int(len(validation.dataframe))} lignes conformes et {fmt_int(len(validation.dataframe.columns))} colonnes controlees. "
                "Il peut etre score sans retraitement manuel. Si le fichier avait une colonne obligatoire manquante ou un type invalide, il serait rejete avant prediction pour eviter une decision metier fausse.",
            )

            if st.button("Valider et scorer le fichier", type="primary", width="stretch"):
                with st.spinner("Scoring du fichier en cours..."):
                    scored = score_fraud_dataframe(model, validation.dataframe, threshold=threshold)
                render_scored_results(scored, threshold, delimiter, "fraud_scoring_results.csv")

    with schema_tab:
        st.subheader("Format CSV attendu")
        st.dataframe(expected_fraud_schema(), width="stretch", hide_index=True)
        st.download_button(
            "Telecharger un modele CSV",
            data=template_fraud_csv(),
            file_name="template_scoring_fraude.csv",
            mime="text/csv",
            width="stretch",
        )
        business_note(
            "Decision d'exploitation.",
            "Ce modele CSV fixe un contrat de donnees unique entre les equipes metier et la solution de scoring. En production, tout fichier non conforme doit etre retourne a l'emetteur avant scoring.",
        )


def render_customer_page(segments: pd.DataFrame, clustering_metrics: dict, k_scores: pd.DataFrame) -> None:
    if segments.empty:
        missing_asset("Segments clients indisponibles: fichier data/processed/customer_segments.csv absent du deploiement.")
        return
    profile = label_customer_segments(customer_segment_profile(segments))
    report_header("segmentation")
    react_kpis(
        [
            {"label": "Clients", "value": fmt_int(len(segments)), "detail": "Base CRM analysee"},
            {"label": "Segments", "value": fmt_int(profile["segment"].nunique()), "detail": "Profils actionnables"},
            {"label": "Silhouette", "value": f"{clustering_metrics.get('silhouette', 0):.4f}", "detail": "Qualite separation", "tone": "good"},
            {"label": "Davies-Bouldin", "value": f"{clustering_metrics.get('davies_bouldin', 0):.4f}", "detail": "Compacite clusters"},
        ]
    )

    st.subheader("Profils metiers")
    display = profile.copy()
    display["revenu_median"] = display["revenu_median"].map(fmt_money)
    display["depense_moyenne"] = display["depense_moyenne"].map(fmt_money)
    display["reponse_campagne"] = display["reponse_campagne"].map(lambda value: fmt_pct(value, 1))
    st.dataframe(display, width="stretch", hide_index=True)
    business_note(
        "Plan d'action segment.",
        customer_profile_business_insight(profile, segments),
    )

    left, right = st.columns(2)
    with left:
        st.subheader("Depense moyenne")
        st.bar_chart(profile.set_index("segment")["depense_moyenne"])
        business_note(
            "Decision valeur.",
            spend_business_insight(profile),
        )

    with right:
        st.subheader("Reponse campagne")
        st.bar_chart(profile.set_index("segment")["reponse_campagne"])
        business_note(
            "Decision campagne.",
            response_business_insight(profile, segments),
        )

    st.subheader("Projection revenu / depense")
    plot_data = segments.sample(min(1500, len(segments)), random_state=42)
    st.scatter_chart(plot_data, x="Income", y="Total_Spend", color="segment")
    business_note(
        "Decision ciblage.",
        "La projection revenu/depense sert a isoler les clients a fort potentiel: revenu eleve et depense elevee pour fidelisation, revenu eleve mais depense faible pour stimulation, faible revenu/faible depense pour campagnes automatisees a faible cout.",
    )

    if not k_scores.empty:
        st.subheader("Comparaison des nombres de clusters")
        st.dataframe(k_scores, width="stretch", hide_index=True)
        business_note(
            "Decision clustering.",
            cluster_choice_business_insight(k_scores, profile["segment"].nunique()),
        )


PAGE_BY_MENU = {section["menu"]: key for key, section in REPORT_PAGES.items()}


page = render_sidebar()
page_key = PAGE_BY_MENU[page]
fraud_metrics = load_json(str(PATHS.report_dir / "fraud_metrics.json"))
clustering_metrics = load_json(str(PATHS.report_dir / "customer_clustering_metrics.json"))

if page_key in {"introduction", "fraude"}:
    fraud_data = load_fraud_data()
else:
    fraud_data = pd.DataFrame()

if page_key in {"introduction", "segmentation"}:
    customer_segments = load_customer_segments()
else:
    customer_segments = pd.DataFrame()

if page_key == "introduction":
    render_executive_page(fraud_data, customer_segments, fraud_metrics)
elif page_key == "fraude":
    render_fraud_page(fraud_data, fraud_metrics)
elif page_key == "scoring":
    render_operational_scoring_page()
elif page_key == "segmentation":
    render_customer_page(customer_segments, clustering_metrics, load_k_scores())
