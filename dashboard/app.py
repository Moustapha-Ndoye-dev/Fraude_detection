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


def page_header(title: str, subtitle: str) -> None:
    react_hero(
        title,
        subtitle,
        chips=["Production", "Scoring CSV", "Lecture business"],
    )


def show_metric(label: str, value: str, help_text: str | None = None) -> None:
    st.metric(label, value, help=help_text)


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


def customer_profile(segments: pd.DataFrame) -> pd.DataFrame:
    return (
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
        .reset_index()
        .sort_values("segment")
    )


def segment_labels(profile: pd.DataFrame) -> pd.DataFrame:
    profile = profile.copy()
    profile["label_metier"] = "Clients economes"
    profile.loc[profile["depense_moyenne"].idxmax(), "label_metier"] = "Clients premium"
    profile.loc[profile["recence_moyenne"].idxmax(), "label_metier"] = "Clients dormants"
    web_promo_score = profile["achats_web_moyens"].rank() + profile["achats_promo_moyens"].rank()
    profile.loc[web_promo_score.idxmax(), "label_metier"] = "Digitaux et promotions"
    return profile


def render_sidebar() -> str:
    st.sidebar.title("Fraud Intelligence")
    st.sidebar.caption("Pilotage industriel des risques et des segments clients")
    page = st.sidebar.radio(
        "Navigation",
        [
            "Synthese executive",
            "Risque fraude",
            "Scoring operationnel",
            "Segments clients",
            "MLOps & exploitation",
        ],
    )
    st.sidebar.divider()
    st.sidebar.caption("Statut solution")
    st.sidebar.write(f"Service de scoring: {'OK' if scoring_ready() else 'MODELE MANQUANT'}")
    for name, ok in artifact_status().items():
        st.sidebar.write(f"{name}: {'OK' if ok else 'MANQUANT'}")
    return page


def render_executive_page(fraud_df: pd.DataFrame, segments: pd.DataFrame, fraud_metrics: dict) -> None:
    page_header(
        "Synthese executive",
        "Vue de pilotage pour dirigeants, risque, conformite et marketing.",
    )
    if fraud_df.empty:
        missing_asset("Donnees de fraude indisponibles: fichier detection_fraude.csv absent du deploiement.")
        return
    if segments.empty:
        missing_asset("Segments clients indisponibles: fichier data/processed/customer_segments.csv absent du deploiement.")
        return
    total_transactions = len(fraud_df)
    total_frauds = int(fraud_df["isFraud"].sum())
    profile = segment_labels(customer_profile(segments))

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
        "Lecture business.",
        "La solution priorise les transactions suspectes, reduit la charge analyste et transforme les segments clients en actions marketing ciblees.",
    )


def render_fraud_page(fraud_df: pd.DataFrame, fraud_metrics: dict) -> None:
    page_header(
        "Risque fraude",
        "Analyse operationnelle des transactions, metriques modele et signaux de fraude a prioriser.",
    )
    if fraud_df.empty:
        missing_asset("Donnees de fraude indisponibles: fichier detection_fraude.csv absent du deploiement.")
        return
    total_transactions = len(fraud_df)
    total_frauds = int(fraud_df["isFraud"].sum())
    fraud_only = fraud_df[fraud_df["isFraud"] == 1]

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
            "Lecture business.",
            "TRANSFER et CASH_OUT concentrent le risque. Ces operations doivent etre surveillees avec un seuil plus strict.",
        )

    with right:
        st.subheader("Fraudes dans le temps")
        step_data = fraud_step_summary(fraud_df).set_index("step")["fraudes"]
        st.line_chart(step_data)
        business_note(
            "Lecture business.",
            "Ce graphique aide a reperer les periodes ou le risque augmente et peut declencher un controle renforce.",
        )

    st.subheader("Transactions frauduleuses les plus importantes")
    st.dataframe(fraud_only.sort_values("amount", ascending=False).head(25), width="stretch", hide_index=True)
    business_note(
        "Lecture business.",
        "Cette liste sert de file de travail pour les analystes. Les montants les plus eleves sont a traiter en priorite.",
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
        "Lecture business.",
        "Ces indicateurs permettent de dimensionner la charge de revue et de prioriser les controles.",
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
        "Lecture business.",
        "LOW peut etre automatise. MEDIUM demande une surveillance. HIGH et CRITICAL alimentent la file de revue prioritaire.",
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
    page_header(
        "Scoring operationnel",
        "Saisissez une transaction ou importez un fichier CSV pour obtenir le risque et l'action recommandee.",
    )
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
                "Lecture business.",
                "Le controle qualite evite de produire des predictions sur un fichier incomplet ou mal formate.",
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
            "Lecture business.",
            "Le modele CSV donne aux equipes metiers un format unique et reduit les erreurs d'import.",
        )


def render_customer_page(segments: pd.DataFrame, clustering_metrics: dict, k_scores: pd.DataFrame) -> None:
    page_header(
        "Segments clients",
        "Profils marketing actionnables pour la fidelisation, la reactivation et les campagnes ciblees.",
    )
    if segments.empty:
        missing_asset("Segments clients indisponibles: fichier data/processed/customer_segments.csv absent du deploiement.")
        return
    profile = segment_labels(customer_profile(segments))
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
        "Lecture business.",
        "Chaque segment doit recevoir une action differente: fidelisation, reactivation, offres ciblees ou campagnes economes.",
    )

    left, right = st.columns(2)
    with left:
        st.subheader("Depense moyenne")
        st.bar_chart(profile.set_index("segment")["depense_moyenne"])
        business_note(
            "Lecture business.",
            "Les segments avec depense elevee sont prioritaires pour la fidelisation.",
        )

    with right:
        st.subheader("Reponse campagne")
        st.bar_chart(profile.set_index("segment")["reponse_campagne"])
        business_note(
            "Lecture business.",
            "Le taux de reponse indique les segments ou les campagnes ont le plus de chances de convertir.",
        )

    st.subheader("Projection revenu / depense")
    plot_data = segments.sample(min(1500, len(segments)), random_state=42)
    st.scatter_chart(plot_data, x="Income", y="Total_Spend", color="segment")
    business_note(
        "Lecture business.",
        "La projection revenu / depense aide a distinguer les clients a fort potentiel des clients a faible valeur actuelle.",
    )

    if not k_scores.empty:
        st.subheader("Comparaison des nombres de clusters")
        st.dataframe(k_scores, width="stretch", hide_index=True)
        business_note(
            "Lecture business.",
            "Le nombre de segments doit rester lisible et actionnable pour les equipes marketing.",
        )


def render_mlops_page() -> None:
    page_header(
        "MLOps & exploitation",
        "Vue cible pour passer d'un prototype a un service exploitable.",
    )
    react_kpis(
        [
            {"label": "Scoring", "value": "Operationnel", "detail": "Transaction et batch CSV", "tone": "good"},
            {"label": "Interface", "value": "Dashboard", "detail": "Pilotage metier"},
            {"label": "Controle", "value": "Schema strict", "detail": "Rejet des fichiers non conformes"},
            {"label": "Deploiement", "value": "Application", "detail": "Streamlit Cloud"},
        ]
    )

    st.subheader("Chaine industrielle")
    architecture = pd.DataFrame(
        [
            ["Ingestion", "CSV, core banking, CRM", "Controle schema, qualite, volumetrie"],
            ["Feature engineering", "Variables de solde, montant, canal", "Transformation reproductible"],
            ["Scoring", "Moteur de scoring", "Probabilite, niveau de risque, action"],
            ["Pilotage", "Dashboard executif", "KPIs, alertes, segments"],
            ["Monitoring", "Drift, performance, erreurs", "Detection de degradation"],
            ["Reentrainement", "Pipeline planifie", "Mise a jour controlee du modele"],
        ],
        columns=["Couche", "Implementation", "Role"],
    )
    st.dataframe(architecture, width="stretch", hide_index=True)

    left, right = st.columns(2)
    with left:
        st.subheader("Controles production")
        for control in [
            "Fichiers CSV rejetes si colonnes manquantes ou types invalides.",
            "Journalisation des volumes scores et taux de risque.",
            "Seuil de fraude configurable selon le cout metier.",
            "Suivi des faux positifs et faux negatifs apres retour analyste.",
            "Surveillance de la derive des montants, types et soldes.",
        ]:
            st.write(control)
    with right:
        st.subheader("Usage metier reel")
        for usage in [
            "Fraude critique: blocage temporaire ou revue prioritaire.",
            "Fraude elevee: file analyste.",
            "Risque moyen: monitoring renforce.",
            "Risque faible: validation automatique.",
            "Segments clients: ciblage CRM et personnalisation marketing.",
        ]:
            st.write(usage)


page = render_sidebar()
fraud_metrics = load_json(str(PATHS.report_dir / "fraud_metrics.json"))
clustering_metrics = load_json(str(PATHS.report_dir / "customer_clustering_metrics.json"))

if page in {"Synthese executive", "Risque fraude"}:
    fraud_data = load_fraud_data()
else:
    fraud_data = pd.DataFrame()

if page in {"Synthese executive", "Segments clients"}:
    customer_segments = load_customer_segments()
else:
    customer_segments = pd.DataFrame()

if page == "Synthese executive":
    render_executive_page(fraud_data, customer_segments, fraud_metrics)
elif page == "Risque fraude":
    render_fraud_page(fraud_data, fraud_metrics)
elif page == "Scoring operationnel":
    render_operational_scoring_page()
elif page == "Segments clients":
    render_customer_page(customer_segments, clustering_metrics, load_k_scores())
elif page == "MLOps & exploitation":
    render_mlops_page()
