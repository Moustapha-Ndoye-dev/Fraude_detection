from __future__ import annotations

import json
import sys
from io import BytesIO
from pathlib import Path

import altair as alt
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


CHART_COLORS = ["#0F766E", "#1D4ED8", "#B45309", "#BE123C", "#6D28D9", "#047857"]


st.set_page_config(
    page_title="Fraud & Customer Intelligence",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)


def theme_tokens(mode: str) -> dict[str, str | list[str]]:
    if mode == "Sombre":
        return {
            "bg": "#0F172A",
            "surface": "#111827",
            "surface_alt": "#1F2937",
            "border": "#334155",
            "text": "#F8FAFC",
            "muted": "#CBD5E1",
            "primary": "#2DD4BF",
            "primary_text": "#042F2E",
            "success_bg": "#052E16",
            "success_border": "#16A34A",
            "warning_bg": "#431407",
            "warning_border": "#EA580C",
            "code_bg": "#020617",
            "sidebar_bg": "#020617",
            "sidebar_text": "#F8FAFC",
            "chart_colors": ["#2DD4BF", "#60A5FA", "#F59E0B", "#FB7185", "#A78BFA", "#34D399"],
        }
    if mode == "Clair":
        return {
            "bg": "#F4F6F8",
            "surface": "#FFFFFF",
            "surface_alt": "#F8FAFC",
            "border": "#D8DEE8",
            "text": "#111827",
            "muted": "#475569",
            "primary": "#0F766E",
            "primary_text": "#FFFFFF",
            "success_bg": "#ECFDF5",
            "success_border": "#86EFAC",
            "warning_bg": "#FFF7ED",
            "warning_border": "#FDBA74",
            "code_bg": "#E2E8F0",
            "sidebar_bg": "#0B1220",
            "sidebar_text": "#F9FAFB",
            "chart_colors": ["#0F766E", "#1D4ED8", "#B45309", "#BE123C", "#6D28D9", "#047857"],
        }
    return {
        "bg": "var(--background-color, #F4F6F8)",
        "surface": "var(--secondary-background-color, #FFFFFF)",
        "surface_alt": "var(--secondary-background-color, #F8FAFC)",
        "border": "rgba(148, 163, 184, 0.55)",
        "text": "var(--text-color, #111827)",
        "muted": "rgba(100, 116, 139, 0.95)",
        "primary": "var(--primary-color, #0F766E)",
        "primary_text": "#FFFFFF",
        "success_bg": "rgba(16, 185, 129, 0.14)",
        "success_border": "rgba(16, 185, 129, 0.65)",
        "warning_bg": "rgba(249, 115, 22, 0.14)",
        "warning_border": "rgba(249, 115, 22, 0.65)",
        "code_bg": "rgba(148, 163, 184, 0.18)",
        "sidebar_bg": "var(--secondary-background-color, #0B1220)",
        "sidebar_text": "var(--text-color, #F9FAFB)",
        "chart_colors": ["#0F766E", "#2563EB", "#B45309", "#BE123C", "#7C3AED", "#059669"],
    }


def build_css(tokens: dict[str, str | list[str]]) -> str:
    return f"""
<style>
    html, body, [data-testid="stAppViewContainer"] {{
        background: {tokens["bg"]};
        color: {tokens["text"]};
    }}
    [data-testid="stHeader"] {{
        background: transparent !important;
    }}
    .block-container {{
        padding-top: 1rem;
        padding-bottom: 2rem;
        max-width: 1400px;
        color: {tokens["text"]} !important;
    }}
    [data-testid="stSidebar"] {{
        background: {tokens["sidebar_bg"]};
        border-right: 1px solid {tokens["border"]};
    }}
    [data-testid="stSidebar"] * {{
        color: {tokens["sidebar_text"]} !important;
    }}
    [data-testid="stSidebar"] [data-baseweb="select"] > div,
    [data-testid="stSidebar"] [data-baseweb="input"] > div,
    [data-testid="stSidebar"] input,
    [data-testid="stSidebar"] textarea {{
        background: {tokens["surface"]} !important;
        color: {tokens["text"]} !important;
        border-color: {tokens["border"]} !important;
    }}
    [data-testid="stSidebar"] [data-baseweb="select"] *,
    [data-testid="stSidebar"] [data-baseweb="input"] *,
    [data-testid="stSidebar"] input,
    [data-testid="stSidebar"] textarea {{
        color: {tokens["text"]} !important;
    }}
    [data-testid="stSidebar"] [role="listbox"] *,
    div[data-baseweb="popover"] * {{
        color: {tokens["text"]} !important;
    }}
    div[data-baseweb="popover"] {{
        background: {tokens["surface"]} !important;
    }}
    h1, h2, h3, h4, p, li, label, span {{
        letter-spacing: 0;
    }}
    .block-container h1,
    .block-container h2,
    .block-container h3,
    .block-container h4,
    .block-container p,
    .block-container li,
    .block-container label,
    .block-container span,
    .block-container [data-testid="stMarkdownContainer"],
    .block-container [data-testid="stMarkdownContainer"] * {{
        color: {tokens["text"]} !important;
    }}
    .block-container [data-testid="stCaptionContainer"],
    .block-container small {{
        color: {tokens["muted"]} !important;
    }}
    .block-container input,
    .block-container textarea,
    .block-container [data-baseweb="select"] > div,
    .block-container [data-baseweb="input"] > div {{
        background: {tokens["surface"]} !important;
        color: {tokens["text"]} !important;
        border-color: {tokens["border"]} !important;
    }}
    .block-container [data-testid="stWidgetLabel"] *,
    .block-container [data-testid="stFileUploader"] *,
    .block-container [data-testid="stSelectbox"] *,
    .block-container [data-testid="stSlider"] *,
    .block-container [data-testid="stCheckbox"] *,
    .block-container [data-testid="stRadio"] * {{
        color: {tokens["text"]} !important;
    }}
    .block-container [data-testid="stFileUploaderDropzone"] {{
        background: {tokens["surface"]} !important;
        border: 1px dashed {tokens["border"]} !important;
    }}
    .block-container [data-testid="stAlert"] *,
    .block-container [data-testid="stDataFrame"] *,
    .block-container [data-testid="stTable"] * {{
        color: {tokens["text"]} !important;
    }}
    .block-container code,
    .block-container pre {{
        color: {tokens["text"]} !important;
        background: {tokens["code_bg"]} !important;
    }}
    .topbar, .metric-card {{
        background: {tokens["surface"]};
        border: 1px solid {tokens["border"]};
        border-radius: 8px;
        box-shadow: 0 1px 2px rgba(15, 23, 42, 0.08);
    }}
    .topbar {{
        padding: 1rem 1.15rem;
        margin-bottom: 1rem;
    }}
    .page-title {{
        font-size: 1.85rem;
        line-height: 1.2;
        font-weight: 800;
        color: {tokens["text"]} !important;
        margin-bottom: 0.2rem;
    }}
    .page-subtitle {{
        color: {tokens["muted"]} !important;
        font-size: 1rem;
        line-height: 1.45;
    }}
    .metric-card {{
        padding: 0.95rem 1rem;
        min-height: 112px;
    }}
    .metric-label {{
        color: {tokens["muted"]} !important;
        font-size: 0.78rem;
        font-weight: 760;
        text-transform: uppercase;
    }}
    .metric-value {{
        color: {tokens["text"]} !important;
        font-size: 1.65rem;
        font-weight: 820;
        margin-top: 0.2rem;
    }}
    .metric-caption {{
        color: {tokens["muted"]} !important;
        font-size: 0.86rem;
        margin-top: 0.25rem;
    }}
    .callout {{
        background: {tokens["success_bg"]};
        border: 1px solid {tokens["success_border"]};
        border-left: 5px solid {tokens["primary"]};
        border-radius: 8px;
        padding: 1rem;
        color: {tokens["text"]} !important;
    }}
    .warning-box {{
        background: {tokens["warning_bg"]};
        border: 1px solid {tokens["warning_border"]};
        border-left: 5px solid {tokens["warning_border"]};
        border-radius: 8px;
        padding: 1rem;
        color: {tokens["text"]} !important;
    }}
    div[data-testid="stDataFrame"] {{
        border: 1px solid {tokens["border"]};
        border-radius: 8px;
    }}
    div.stButton > button, div.stDownloadButton > button {{
        border-radius: 6px;
        font-weight: 760;
        background: {tokens["surface"]} !important;
        color: {tokens["text"]} !important;
        border: 1px solid {tokens["border"]} !important;
    }}
    div.stButton > button *,
    div.stDownloadButton > button * {{
        color: {tokens["text"]} !important;
    }}
    div.stButton > button[kind="primary"],
    div.stDownloadButton > button[kind="primary"],
    button[data-testid="baseButton-primary"] {{
        color: {tokens["primary_text"]} !important;
        background: {tokens["primary"]} !important;
        border-color: {tokens["primary"]} !important;
    }}
    div.stButton > button[kind="primary"] *,
    div.stDownloadButton > button[kind="primary"] *,
    button[data-testid="baseButton-primary"] * {{
        color: {tokens["primary_text"]} !important;
    }}
</style>
"""


ACTIVE_THEME_MODE = "Auto Streamlit"
ACTIVE_THEME = theme_tokens(ACTIVE_THEME_MODE)


def enterprise_chart(chart: alt.Chart) -> alt.Chart:
    if ACTIVE_THEME_MODE == "Sombre":
        chart_bg = "#111827"
        text = "#F8FAFC"
        grid = "#334155"
        domain = "#475569"
    else:
        chart_bg = "#FFFFFF"
        text = "#111827"
        grid = "#E5E7EB"
        domain = "#CBD5E1"
    return (
        chart.properties(background=chart_bg)
        .configure_view(stroke=None)
        .configure_axis(
            labelColor=text,
            titleColor=text,
            gridColor=grid,
            domainColor=domain,
            tickColor=domain,
            labelFontSize=12,
            titleFontSize=13,
            titleFontWeight=700,
        )
        .configure_legend(
            labelColor=text,
            titleColor=text,
            labelFontSize=13,
            titleFontSize=14,
            titleFontWeight=700,
        )
        .configure_title(color=text, fontSize=16, fontWeight=700)
    )


def fmt_int(value: float | int) -> str:
    return f"{value:,.0f}".replace(",", " ")


def fmt_money(value: float | int) -> str:
    return f"{value:,.0f}".replace(",", " ")


def fmt_pct(value: float, digits: int = 2) -> str:
    return f"{value * 100:.{digits}f}%"


def page_header(title: str, subtitle: str) -> None:
    st.markdown(
        f"""
        <div class="topbar">
            <div class="page-title">{title}</div>
            <div class="page-subtitle">{subtitle}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def metric_card(label: str, value: str, caption: str = "") -> None:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
            <div class="metric-caption">{caption}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


@st.cache_data(show_spinner=False)
def load_json(path: str) -> dict:
    file_path = Path(path)
    if not file_path.exists():
        return {}
    return json.loads(file_path.read_text(encoding="utf-8"))


@st.cache_data(show_spinner="Chargement des transactions...")
def load_fraud_data() -> pd.DataFrame:
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
        raise FileNotFoundError("Segments introuvables. Lancer scripts/train_customer_clustering.py")
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


def artifact_status() -> dict[str, bool]:
    return {
        "Scoring fraude": (PATHS.model_dir / "fraud_pipeline.joblib").exists(),
        "Segmentation clients": (PATHS.model_dir / "customer_clustering.joblib").exists(),
        "Indicateurs fraude": (PATHS.report_dir / "fraud_metrics.json").exists(),
        "Fichier segments": (PATHS.processed_dir / "customer_segments.csv").exists(),
    }


def scoring_ready() -> bool:
    return (PATHS.model_dir / "fraud_pipeline.joblib").exists()


def business_note(title: str, body: str) -> None:
    st.markdown(
        f"""
        <div class="callout">
            <b>{title}</b><br>{body}
        </div>
        """,
        unsafe_allow_html=True,
    )


def fraud_type_summary(fraud_df: pd.DataFrame) -> pd.DataFrame:
    return (
        fraud_df.groupby("type")
        .agg(transactions=("isFraud", "size"), fraudes=("isFraud", "sum"), montant_median=("amount", "median"))
        .assign(taux_fraude=lambda x: x["fraudes"] / x["transactions"])
        .reset_index()
        .sort_values("taux_fraude", ascending=False)
    )


def fraud_step_summary(fraud_df: pd.DataFrame) -> pd.DataFrame:
    return (
        fraud_df.groupby("step")
        .agg(transactions=("isFraud", "size"), fraudes=("isFraud", "sum"))
        .assign(taux_fraude=lambda x: x["fraudes"] / x["transactions"])
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


def render_sidebar() -> tuple[str, str]:
    st.sidebar.title("Fraud Intelligence")
    st.sidebar.caption("Pilotage industriel des risques et des segments clients")
    theme_mode = st.sidebar.selectbox(
        "Apparence",
        ["Auto Streamlit", "Clair", "Sombre"],
        index=0,
        help="Auto respecte les couleurs natives Streamlit. Clair/Sombre appliquent un thème dashboard complet.",
    )
    page = st.sidebar.radio(
        "Navigation",
        [
            "Synthese executive",
            "Risque fraude",
            "Scoring opérationnel",
            "Segments clients",
            "MLOps & exploitation",
        ],
    )
    st.sidebar.divider()
    st.sidebar.caption("Statut solution")
    st.sidebar.write(f"Service de scoring: {'OK' if scoring_ready() else 'MODELE MANQUANT'}")
    for name, ok in artifact_status().items():
        st.sidebar.write(f"{name}: {'OK' if ok else 'MANQUANT'}")
    return page, theme_mode


def render_executive_page(fraud_df: pd.DataFrame, segments: pd.DataFrame, fraud_metrics: dict) -> None:
    page_header(
        "Synthese executive",
        "Vue de pilotage pour dirigeants, risque, conformité et marketing: performance modèle, risque transactionnel et valeur client.",
    )
    total_transactions = len(fraud_df)
    total_frauds = int(fraud_df["isFraud"].sum())
    profile = segment_labels(customer_profile(segments))

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        metric_card("Transactions analysees", fmt_int(total_transactions), "Historique complet")
    with c2:
        metric_card("Taux de fraude", fmt_pct(total_frauds / total_transactions, 3), f"{fmt_int(total_frauds)} cas observes")
    with c3:
        metric_card("Recall modele", fmt_pct(fraud_metrics.get("recall", 0), 2), "Fraudes retrouvees")
    with c4:
        metric_card("Clients segmentes", fmt_int(len(segments)), f"{profile['segment'].nunique()} profils metiers")

    left, right = st.columns([1.15, 1])
    with left:
        st.subheader("Risque fraude par type")
        chart_data = fraud_type_summary(fraud_df)
        chart = (
            alt.Chart(chart_data)
            .mark_bar(cornerRadiusTopLeft=3, cornerRadiusTopRight=3)
            .encode(
                x=alt.X("type:N", title="Type de transaction"),
                y=alt.Y("taux_fraude:Q", title="Taux de fraude", axis=alt.Axis(format="%")),
                color=alt.Color("type:N", scale=alt.Scale(range=CHART_COLORS), legend=None),
                tooltip=["type", alt.Tooltip("transactions:Q", format=","), alt.Tooltip("fraudes:Q", format=","), alt.Tooltip("taux_fraude:Q", format=".4%")],
            )
            .properties(height=310)
        )
        st.altair_chart(enterprise_chart(chart), width="stretch", theme=None)

    with right:
        st.subheader("Segments clients")
        chart = (
            alt.Chart(profile)
            .mark_bar(cornerRadiusTopLeft=3, cornerRadiusTopRight=3)
            .encode(
                x=alt.X("segment:N", title="Segment"),
                y=alt.Y("clients:Q", title="Clients"),
                color=alt.Color("label_metier:N", scale=alt.Scale(range=CHART_COLORS), title="Profil"),
                tooltip=["segment", "label_metier", alt.Tooltip("clients:Q", format=",")],
            )
            .properties(height=310)
        )
        st.altair_chart(enterprise_chart(chart), width="stretch", theme=None)

    st.markdown(
        """
        <div class="callout">
            <b>Lecture business.</b> La solution priorise les transactions suspectes, réduit la charge analyste,
            permet le scoring en volume via fichier CSV et transforme les segments clients en actions marketing ciblées.
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_fraud_page(fraud_df: pd.DataFrame, fraud_metrics: dict) -> None:
    page_header(
        "Risque fraude",
        "Analyse opérationnelle des transactions, métriques modèle et signaux de fraude à prioriser.",
    )
    total_transactions = len(fraud_df)
    total_frauds = int(fraud_df["isFraud"].sum())
    fraud_only = fraud_df[fraud_df["isFraud"] == 1]

    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        metric_card("Transactions", fmt_int(total_transactions))
    with c2:
        metric_card("Fraudes observees", fmt_int(total_frauds))
    with c3:
        metric_card("Taux historique", fmt_pct(total_frauds / total_transactions, 3))
    with c4:
        metric_card("F1-score", fmt_pct(fraud_metrics.get("f1", 0), 2))
    with c5:
        metric_card("ROC-AUC", f"{fraud_metrics.get('roc_auc', 0):.4f}")

    left, right = st.columns([1, 1])
    with left:
        st.subheader("Fraude par type de transaction")
        summary = fraud_type_summary(fraud_df)
        table = summary.copy()
        table["taux_fraude"] = table["taux_fraude"].map(lambda x: f"{x * 100:.4f}%")
        table["montant_median"] = table["montant_median"].map(fmt_money)
        st.dataframe(table, width="stretch", hide_index=True)
        business_note(
            "Lecture business.",
            "Les types TRANSFER et CASH_OUT concentrent le risque. En exploitation, ces opérations doivent être surveillées avec un seuil plus strict et une revue prioritaire.",
        )

    with right:
        st.subheader("Fraudes dans le temps")
        step_data = fraud_step_summary(fraud_df)
        chart = (
            alt.Chart(step_data)
            .mark_line(strokeWidth=2, color="#0F766E")
            .encode(
                x=alt.X("step:Q", title="Step"),
                y=alt.Y("fraudes:Q", title="Fraudes"),
                tooltip=["step", "transactions", "fraudes", alt.Tooltip("taux_fraude:Q", format=".4%")],
            )
            .properties(height=290)
        )
        st.altair_chart(enterprise_chart(chart), width="stretch", theme=None)
        business_note(
            "Lecture business.",
            "Ce graphique aide à repérer les périodes où le risque augmente. Une hausse soudaine peut déclencher une alerte opérationnelle ou un contrôle renforcé.",
        )

    st.subheader("Transactions frauduleuses les plus importantes")
    st.dataframe(
        fraud_only.sort_values("amount", ascending=False).head(25),
        width="stretch",
        hide_index=True,
    )
    business_note(
        "Lecture business.",
        "Cette liste sert de file de travail pour les analystes: les montants les plus élevés sont les dossiers à traiter en priorité.",
    )


def render_scoring_page() -> None:
    page_header(
        "Scoring opérationnel",
        "Importez un fichier normalise, validez strictement son schema, scorez en volume et récupérez un fichier enrichi avec probabilité, niveau de risque et action recommandée.",
    )
    online = scoring_ready()
    c1, c2, c3 = st.columns(3)
    with c1:
        metric_card("Service de scoring", "OK" if online else "INDISPONIBLE", "Transaction et CSV")
    with c2:
        metric_card("Schema", "Strict", "Colonnes et types contrôlés")
    with c3:
        metric_card("Sortie", "CSV scoré", "Probabilité, risque, action")

    left, right = st.columns([1, 1])
    with left:
        st.subheader("Schema attendu")
        st.dataframe(expected_fraud_schema(), width="stretch", hide_index=True)
        st.download_button(
            "Télécharger un modèle CSV",
            data=template_fraud_csv(),
            file_name="template_scoring_fraude.csv",
            mime="text/csv",
            width="stretch",
        )

    with right:
        st.subheader("Paramètres de scoring")
        delimiter = st.selectbox("Séparateur CSV", [";", ","], index=0)
        threshold = st.slider("Seuil de décision fraude", min_value=0.0, max_value=1.0, value=0.5, step=0.01)
        strict = st.checkbox("Rejeter les colonnes non attendues", value=True)
        mode = st.radio("Moteur de scoring", ["Service de scoring"], horizontal=True)
        uploaded_file = st.file_uploader("Importer un CSV de transactions", type=["csv"])

    if uploaded_file is None:
        st.markdown(
            """
            <div class="warning-box">
                Aucun fichier importé. Le fichier doit respecter le schema attendu. En mode strict,
                toute colonne manquante ou non prévue bloque le scoring.
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    file_bytes = uploaded_file.getvalue()
    try:
        input_df = pd.read_csv(BytesIO(file_bytes), sep=delimiter)
    except Exception as exc:
        st.error(f"Lecture CSV impossible: {exc}")
        return

    validation = validate_fraud_input(input_df, strict=strict)
    if not validation.is_valid:
        st.error("Fichier rejeté: le schema ou les valeurs ne sont pas conformes.")
        for error in validation.errors:
            st.write(f"- {error}")
        return

    for warning in validation.warnings:
        st.warning(warning)

    scored = None
    api_result = None
    if st.button("Valider et scorer le fichier", type="primary", width="stretch"):
        if False:
            pass
        else:
            model = load_fraud_model()
            if model is None:
                st.error("Service de scoring indisponible: modèle fraude introuvable.")
                return
            scored = score_fraud_dataframe(model, validation.dataframe, threshold=threshold)
            api_result = {"summary": summarize_scoring(scored, threshold=threshold), "warnings": validation.warnings}

        summary = api_result["summary"]
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            metric_card("Lignes scorées", fmt_int(summary["rows"]))
        with c2:
            metric_card("Fraudes prédites", fmt_int(summary["predicted_frauds"]), fmt_pct(summary["predicted_fraud_rate"], 2))
        with c3:
            metric_card("Risque moyen", fmt_pct(summary["average_probability"], 2))
        with c4:
            metric_card("Critiques", fmt_int(summary["critical_transactions"]), "Blocage temporaire")

        st.subheader("Transactions les plus risquées")
        st.dataframe(
            scored.sort_values("fraud_probability", ascending=False).head(50),
            width="stretch",
            hide_index=True,
        )
        output = BytesIO()
        scored.to_csv(output, index=False, sep=delimiter)
        st.download_button(
            "Exporter le CSV scoré",
            data=output.getvalue(),
            file_name="fraud_scoring_results.csv",
            mime="text/csv",
            width="stretch",
        )


def risk_business_text(risk_band: str, probability: float) -> str:
    if risk_band == "CRITICAL":
        return (
            f"Risque critique ({fmt_pct(probability, 2)}). La transaction doit être bloquée temporairement "
            "ou envoyée immédiatement en revue analyste."
        )
    if risk_band == "HIGH":
        return f"Risque élevé ({fmt_pct(probability, 2)}). La transaction doit être contrôlée avant validation."
    if risk_band == "MEDIUM":
        return f"Risque moyen ({fmt_pct(probability, 2)}). La transaction peut passer avec surveillance renforcée."
    return f"Risque faible ({fmt_pct(probability, 2)}). La transaction peut être validée automatiquement."


def render_scored_results(scored: pd.DataFrame, threshold: float, delimiter: str, export_name: str) -> None:
    summary = summarize_scoring(scored, threshold=threshold)
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        metric_card("Lignes scorées", fmt_int(summary["rows"]))
    with c2:
        metric_card("Alertes fraude", fmt_int(summary["predicted_frauds"]), fmt_pct(summary["predicted_fraud_rate"], 2))
    with c3:
        metric_card("Risque moyen", fmt_pct(summary["average_probability"], 2))
    with c4:
        metric_card("Critiques", fmt_int(summary["critical_transactions"]), "Action immédiate")

    business_note(
        "Lecture business.",
        "Ces indicateurs permettent de dimensionner la charge de revue: plus le nombre d'alertes et de cas critiques augmente, plus l'équipe conformité doit prioriser les contrôles.",
    )

    risk_order = ["LOW", "MEDIUM", "HIGH", "CRITICAL"]
    risk_counts = (
        scored["risk_band"]
        .value_counts()
        .reindex(risk_order, fill_value=0)
        .rename_axis("risk_band")
        .reset_index(name="transactions")
    )
    st.subheader("Répartition des niveaux de risque")
    chart = (
        alt.Chart(risk_counts)
        .mark_bar(cornerRadiusTopLeft=3, cornerRadiusTopRight=3)
        .encode(
            x=alt.X("risk_band:N", title="Niveau de risque", sort=risk_order),
            y=alt.Y("transactions:Q", title="Transactions"),
            color=alt.Color("risk_band:N", scale=alt.Scale(range=CHART_COLORS), legend=None),
            tooltip=["risk_band", alt.Tooltip("transactions:Q", format=",")],
        )
        .properties(height=280)
    )
    st.altair_chart(enterprise_chart(chart), width="stretch", theme=None)
    business_note(
        "Lecture business.",
        "LOW peut être automatisé. MEDIUM demande une surveillance. HIGH et CRITICAL alimentent la file de revue prioritaire.",
    )

    st.subheader("Transactions les plus risquées")
    top_risk = scored.sort_values("fraud_probability", ascending=False).head(50)
    st.dataframe(top_risk, width="stretch", hide_index=True)
    business_note(
        "Lecture business.",
        "Cette table est la liste opérationnelle à transmettre aux analystes. Elle contient la probabilité, le niveau de risque et l'action recommandée.",
    )

    output = BytesIO()
    scored.to_csv(output, index=False, sep=delimiter)
    st.download_button(
        "Exporter le fichier scoré",
        data=output.getvalue(),
        file_name=export_name,
        mime="text/csv",
        width="stretch",
    )


def render_operational_scoring_page() -> None:
    page_header(
        "Scoring opérationnel",
        "Saisissez une transaction ou importez un fichier CSV. Le système contrôle les champs, calcule le risque et propose une action métier.",
    )
    model = load_fraud_model()
    if model is None:
        st.error("Le service de scoring n'est pas prêt: le modèle fraude est introuvable.")
        return

    c1, c2, c3 = st.columns(3)
    with c1:
        metric_card("Service de scoring", "Prêt", "Transaction seule ou fichier CSV")
    with c2:
        metric_card("Contrôle qualité", "Strict", "Schéma et valeurs vérifiés")
    with c3:
        metric_card("Sortie métier", "Action", "Validation, surveillance ou revue")

    transaction_tab, csv_tab, schema_tab = st.tabs(["Saisie transaction", "Import CSV", "Format attendu"])

    with transaction_tab:
        st.subheader("Saisir une transaction")
        threshold = st.slider("Seuil de décision fraude", 0.0, 1.0, 0.5, 0.01, key="single_threshold")
        with st.form("single_transaction_form"):
            col1, col2, col3 = st.columns(3)
            with col1:
                step = st.number_input("Step", min_value=0, value=1, step=1)
                tx_type = st.selectbox("Type de transaction", ["TRANSFER", "CASH_OUT", "PAYMENT", "CASH_IN", "DEBIT"])
                amount = st.number_input("Montant", min_value=0.0, value=181.0, step=100.0)
            with col2:
                oldbalance_org = st.number_input("Solde émetteur avant", min_value=0.0, value=181.0, step=100.0)
                newbalance_orig = st.number_input("Solde émetteur après", min_value=0.0, value=0.0, step=100.0)
                name_orig = st.text_input("Identifiant émetteur", value="C1305486145")
            with col3:
                oldbalance_dest = st.number_input("Solde destinataire avant", min_value=0.0, value=0.0, step=100.0)
                newbalance_dest = st.number_input("Solde destinataire après", min_value=0.0, value=0.0, step=100.0)
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
                st.error("Transaction rejetée: certaines valeurs ne sont pas conformes.")
                for error in validation.errors:
                    st.write(f"- {error}")
            else:
                scored = score_fraud_dataframe(model, validation.dataframe, threshold=threshold)
                result = scored.iloc[0]
                c1, c2, c3 = st.columns(3)
                with c1:
                    metric_card("Décision", "Alerte" if int(result["predicted_isFraud"]) else "Acceptée")
                with c2:
                    metric_card("Probabilité fraude", fmt_pct(float(result["fraud_probability"]), 2))
                with c3:
                    metric_card("Niveau de risque", str(result["risk_band"]), str(result["recommended_action"]))
                business_note("Interprétation.", risk_business_text(str(result["risk_band"]), float(result["fraud_probability"])))
                st.dataframe(scored, width="stretch", hide_index=True)

    with csv_tab:
        st.subheader("Importer un fichier de transactions")
        left, right = st.columns([1, 1])
        with left:
            delimiter = st.selectbox("Séparateur CSV", [";", ","], index=0, key="csv_delimiter")
            strict = st.checkbox("Rejeter les colonnes non attendues", value=True, key="csv_strict")
        with right:
            threshold = st.slider("Seuil de décision fraude", 0.0, 1.0, 0.5, 0.01, key="csv_threshold")
        uploaded_file = st.file_uploader("Importer un CSV de transactions", type=["csv"], key="transaction_csv")

        if uploaded_file is None:
            st.markdown(
                """
                <div class="warning-box">
                    Aucun fichier importé. Le fichier doit respecter le format attendu. Si une colonne obligatoire manque,
                    le fichier est rejeté avant scoring.
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            file_bytes = uploaded_file.getvalue()
            try:
                input_df = pd.read_csv(BytesIO(file_bytes), sep=delimiter)
            except Exception as exc:
                st.error(f"Lecture CSV impossible: {exc}")
                return

            validation = validate_fraud_input(input_df, strict=strict)
            if not validation.is_valid:
                st.error("Fichier rejeté: le schéma ou les valeurs ne sont pas conformes.")
                for error in validation.errors:
                    st.write(f"- {error}")
                return

            for warning in validation.warnings:
                st.warning(warning)

            c1, c2, c3 = st.columns(3)
            with c1:
                metric_card("Fichier accepté", fmt_int(len(validation.dataframe)), "lignes valides")
            with c2:
                metric_card("Colonnes contrôlées", fmt_int(len(validation.dataframe.columns)))
            with c3:
                metric_card("Prêt à scorer", "Oui", uploaded_file.name)

            st.subheader("Aperçu des données validées")
            st.dataframe(validation.dataframe.head(20), width="stretch", hide_index=True)
            business_note(
                "Lecture business.",
                "Cette étape évite de produire des prédictions sur un fichier incomplet ou mal formaté. Le contrôle qualité se fait avant toute décision métier.",
            )

            if st.button("Valider et scorer le fichier", type="primary", width="stretch"):
                with st.spinner("Scoring du fichier en cours..."):
                    scored = score_fraud_dataframe(model, validation.dataframe, threshold=threshold)
                render_scored_results(scored, threshold, delimiter, "fraud_scoring_results.csv")

    with schema_tab:
        st.subheader("Format CSV attendu")
        st.dataframe(expected_fraud_schema(), width="stretch", hide_index=True)
        st.download_button(
            "Télécharger un modèle CSV",
            data=template_fraud_csv(),
            file_name="template_scoring_fraude.csv",
            mime="text/csv",
            width="stretch",
        )
        business_note(
            "Lecture business.",
            "Le modèle CSV donne aux équipes métiers un format unique. Cela réduit les erreurs d'import et garantit que les résultats sont comparables d'un fichier à l'autre.",
        )


def render_customer_page(segments: pd.DataFrame, clustering_metrics: dict, k_scores: pd.DataFrame) -> None:
    page_header(
        "Segments clients",
        "Profils marketing actionnables pour la fidélisation, la réactivation et les campagnes ciblées.",
    )
    profile = segment_labels(customer_profile(segments))
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        metric_card("Clients", fmt_int(len(segments)))
    with c2:
        metric_card("Segments", fmt_int(profile["segment"].nunique()))
    with c3:
        metric_card("Silhouette", f"{clustering_metrics.get('silhouette', 0):.4f}")
    with c4:
        metric_card("Davies-Bouldin", f"{clustering_metrics.get('davies_bouldin', 0):.4f}")

    st.subheader("Profils métiers")
    display = profile.copy()
    display["revenu_median"] = display["revenu_median"].map(fmt_money)
    display["depense_moyenne"] = display["depense_moyenne"].map(fmt_money)
    display["reponse_campagne"] = display["reponse_campagne"].map(lambda x: fmt_pct(x, 1))
    st.dataframe(display, width="stretch", hide_index=True)
    business_note(
        "Lecture business.",
        "Chaque segment doit recevoir une action différente: fidélisation pour les premium, réactivation pour les dormants, offres ciblées pour les clients digitaux et promotions.",
    )

    left, right = st.columns([1, 1])
    with left:
        st.subheader("Dépense moyenne")
        chart = (
            alt.Chart(profile)
            .mark_bar(cornerRadiusTopLeft=3, cornerRadiusTopRight=3)
            .encode(
                x=alt.X("segment:N", title="Segment"),
                y=alt.Y("depense_moyenne:Q", title="Dépense moyenne"),
                color=alt.Color("label_metier:N", scale=alt.Scale(range=CHART_COLORS), title="Profil"),
                tooltip=["segment", "label_metier", alt.Tooltip("depense_moyenne:Q", format=",.0f")],
            )
            .properties(height=300)
        )
        st.altair_chart(enterprise_chart(chart), width="stretch", theme=None)
        business_note(
            "Lecture business.",
            "Les segments avec dépense élevée sont prioritaires pour la fidélisation, car une petite amélioration de rétention peut générer un fort impact business.",
        )

    with right:
        st.subheader("Réponse campagne")
        chart = (
            alt.Chart(profile)
            .mark_bar(cornerRadiusTopLeft=3, cornerRadiusTopRight=3)
            .encode(
                x=alt.X("segment:N", title="Segment"),
                y=alt.Y("reponse_campagne:Q", title="Taux de réponse", axis=alt.Axis(format="%")),
                color=alt.Color("label_metier:N", scale=alt.Scale(range=CHART_COLORS), title="Profil"),
                tooltip=["segment", "label_metier", alt.Tooltip("reponse_campagne:Q", format=".1%")],
            )
            .properties(height=300)
        )
        st.altair_chart(enterprise_chart(chart), width="stretch", theme=None)
        business_note(
            "Lecture business.",
            "Le taux de réponse indique où les campagnes ont le plus de chances de convertir. Un segment peu réactif demande un message différent ou une pression marketing plus faible.",
        )

    st.subheader("Projection revenu / dépense")
    plot_data = segments.sample(min(1500, len(segments)), random_state=42)
    chart = (
        alt.Chart(plot_data)
        .mark_circle(size=58, opacity=0.72)
        .encode(
            x=alt.X("Income:Q", title="Revenu"),
            y=alt.Y("Total_Spend:Q", title="Dépense totale"),
            color=alt.Color("segment:N", scale=alt.Scale(range=CHART_COLORS), title="Segment"),
            tooltip=["ID", "segment", alt.Tooltip("Income:Q", format=",.0f"), alt.Tooltip("Total_Spend:Q", format=",.0f")],
        )
        .properties(height=420)
    )
    st.altair_chart(enterprise_chart(chart), width="stretch", theme=None)
    business_note(
        "Lecture business.",
        "La projection revenu / dépense aide à distinguer les clients à fort potentiel des clients à faible valeur actuelle. Elle sert à prioriser les efforts CRM.",
    )

    if not k_scores.empty:
        st.subheader("Comparaison des nombres de clusters")
        st.dataframe(k_scores, width="stretch", hide_index=True)
        business_note(
            "Lecture business.",
            "Le nombre de segments ne se choisit pas uniquement par score statistique: il doit aussi rester lisible et actionnable par les équipes marketing.",
        )


def render_mlops_page() -> None:
    page_header(
        "MLOps & exploitation",
        "Vue cible pour passer d'un prototype à un service exploitable: scoring batch, monitoring, gouvernance et réentraînement.",
    )
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        metric_card("Scoring", "Opérationnel", "Transaction et batch CSV")
    with c2:
        metric_card("Interface", "Dashboard", "Pilotage métier")
    with c3:
        metric_card("Contrôle", "Schema strict", "Rejet des fichiers non conformes")
    with c4:
        metric_card("Déploiement", "Application", "Dashboard et service métier")

    st.subheader("Chaîne industrielle")
    architecture = pd.DataFrame(
        [
            ["Ingestion", "CSV, core banking, CRM", "Contrôle schéma, qualité, volumétrie"],
            ["Feature engineering", "Variables de solde, montant, canal", "Transformation reproductible"],
            ["Scoring", "Moteur de scoring", "Probabilité, niveau de risque, action"],
            ["Pilotage", "Dashboard exécutif", "KPIs, alertes, segments"],
            ["Monitoring", "Drift, performance, erreurs", "Détection de dégradation"],
            ["Réentraînement", "Pipeline planifié", "Mise à jour contrôlée du modèle"],
        ],
        columns=["Couche", "Implémentation", "Rôle"],
    )
    st.dataframe(architecture, width="stretch", hide_index=True)

    left, right = st.columns(2)
    with left:
        st.subheader("Contrôles production")
        st.markdown(
            """
            - Fichiers CSV rejetés si colonnes manquantes ou types invalides.
            - Journalisation des volumes scorés et taux de risque.
            - Seuil de fraude configurable selon le coût métier.
            - Suivi des faux positifs et faux négatifs après retour analyste.
            - Surveillance de la dérive des montants, types et soldes.
            """
        )
    with right:
        st.subheader("Usage métier réel")
        st.markdown(
            """
            - Fraude critique: blocage temporaire ou revue prioritaire.
            - Fraude élevée: file analyste.
            - Risque moyen: monitoring renforcé.
            - Risque faible: validation automatique.
            - Segments clients: ciblage CRM et personnalisation marketing.
            """
        )


page, theme_mode = render_sidebar()
ACTIVE_THEME_MODE = theme_mode
ACTIVE_THEME = theme_tokens(theme_mode)
CHART_COLORS = ACTIVE_THEME["chart_colors"]
st.markdown(build_css(ACTIVE_THEME), unsafe_allow_html=True)

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
elif page == "Scoring opérationnel":
    render_operational_scoring_page()
elif page == "Segments clients":
    render_customer_page(customer_segments, clustering_metrics, load_k_scores())
elif page == "MLOps & exploitation":
    render_mlops_page()
