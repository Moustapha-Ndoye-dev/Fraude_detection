from __future__ import annotations

import json
import sys
from datetime import datetime
from html import escape
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ml_project.features.customers import (
    customer_segment_profile,
    label_customer_segments,
    segment_by_label,
)
from ml_project.models.selection import (
    load_selection_record,
    select_best_clustering_model,
    select_best_fraud_model,
)

REPORT_PATH = ROOT / "reports" / "rapport_final.html"
FIGURES_DIR = ROOT / "reports" / "figures"

plt.style.use("seaborn-v0_8-whitegrid")
sns.set_theme(style="whitegrid", context="talk", font_scale=0.85)


def model_label(model_name: str) -> str:
    labels = {
        "logistic_regression": "Regression Logistique",
        "random_forest": "Random Forest",
        "xgboost": "XGBoost",
        "lightgbm": "LightGBM",
        "neural_network": "reseau de neurones",
        "kmeans": "K-Means",
        "dbscan": "DBSCAN",
        "agglomerative": "Agglomerative Clustering",
        "gaussian_mixture": "Gaussian Mixture Models",
    }
    return labels.get(model_name, model_name.replace("_", " "))


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def fmt_int(value: float | int) -> str:
    return f"{value:,.0f}".replace(",", " ")


def fmt_pct(value: float, digits: int = 2) -> str:
    return f"{value * 100:.{digits}f}%"


def save_figure(fig: plt.Figure, name: str) -> str:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    path = FIGURES_DIR / name
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return f"figures/{name}"


def generate_figures(
    fraud_df: pd.DataFrame,
    fraud_comparison: pd.DataFrame,
    feature_importance: pd.DataFrame,
    shap_importance: pd.DataFrame,
    k_scores: pd.DataFrame,
    clustering_comparison: pd.DataFrame,
    segments: pd.DataFrame,
    customer_summary: pd.DataFrame,
) -> dict[str, str]:
    figures: dict[str, str] = {}

    fraud_counts = fraud_df["isFraud"].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(7, 4.5))
    bars = ax.bar(["Transaction normale", "Transaction frauduleuse"], fraud_counts.values, color=["#2563eb", "#b42318"])
    ax.set_title("Distribution de la variable cible isFraud")
    ax.set_ylabel("Nombre de transactions")
    for bar, value in zip(bars, fraud_counts.values, strict=True):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), fmt_int(value), ha="center", va="bottom", fontsize=10)
    figures["fraud_distribution"] = save_figure(fig, "fraud_distribution.png")

    type_summary = (
        fraud_df.groupby("type")
        .agg(transactions=("isFraud", "size"), fraudes=("isFraud", "sum"))
        .assign(taux_fraude=lambda frame: frame["fraudes"] / frame["transactions"])
        .reset_index()
        .sort_values("taux_fraude", ascending=True)
    )
    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.barh(type_summary["type"], type_summary["taux_fraude"] * 100, color="#0f766e")
    ax.set_xlabel("Taux de fraude (%)")
    ax.set_title("Taux de fraude observe par type de transaction")
    figures["fraud_rate_by_type"] = save_figure(fig, "fraud_rate_by_type.png")

    sample = fraud_df.sample(min(120_000, len(fraud_df)), random_state=42)
    fig, ax = plt.subplots(figsize=(8, 4.8))
    sns.boxplot(
        data=sample.assign(classe=sample["isFraud"].map({0: "Normale", 1: "Frauduleuse"})),
        x="classe",
        y="amount",
        hue="classe",
        legend=False,
        palette={"Normale": "#2563eb", "Frauduleuse": "#b42318"},
        ax=ax,
    )
    ax.set_xlabel("Classe")
    ax.set_ylabel("Montant (echelle logarithmique)")
    ax.set_title("Distribution des montants selon la classe de fraude")
    figures["fraud_amounts"] = save_figure(fig, "fraud_amounts.png")

    compare = fraud_comparison.sort_values("recall", ascending=True)
    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.barh(compare["model"], compare["recall"] * 100, color="#1d4ed8", label="Recall")
    ax.barh(compare["model"], compare["f1"] * 100, color="#93c5fd", alpha=0.65, label="F1-score")
    ax.set_xlabel("Score (%)")
    ax.set_title("Comparaison des modeles de detection de fraude")
    ax.legend(loc="lower right")
    figures["fraud_model_comparison"] = save_figure(fig, "fraud_model_comparison.png")

    top_features = feature_importance.head(10).sort_values("importance")
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(top_features["feature"].str.replace("numeric__", "").str.replace("categorical__", ""), top_features["importance"], color="#4338ca")
    ax.set_title("Importance des variables (Random Forest)")
    ax.set_xlabel("Importance relative")
    figures["fraud_feature_importance"] = save_figure(fig, "fraud_feature_importance.png")

    top_shap = shap_importance.head(10).sort_values("mean_abs_shap")
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(top_shap["feature"].str.replace("numeric__", "").str.replace("categorical__", ""), top_shap["mean_abs_shap"], color="#7c3aed")
    ax.set_title("Importance SHAP moyenne (valeur absolue)")
    ax.set_xlabel("Impact moyen sur la prediction")
    figures["fraud_shap_importance"] = save_figure(fig, "fraud_shap_importance.png")

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    axes[0].plot(k_scores["k"], k_scores["silhouette"], marker="o", color="#0f766e")
    axes[0].axvline(4, color="#b54708", linestyle="--", linewidth=1.2, label="k retenu = 4")
    axes[0].set_xlabel("Nombre de clusters k")
    axes[0].set_ylabel("Silhouette")
    axes[0].set_title("Silhouette en fonction de k")
    axes[0].legend()
    axes[1].plot(k_scores["k"], k_scores["davies_bouldin"], marker="o", color="#1d4ed8")
    axes[1].axvline(4, color="#b54708", linestyle="--", linewidth=1.2)
    axes[1].set_xlabel("Nombre de clusters k")
    axes[1].set_ylabel("Davies-Bouldin")
    axes[1].set_title("Davies-Bouldin en fonction de k")
    figures["clustering_k_selection"] = save_figure(fig, "clustering_k_selection.png")

    cluster_cmp = clustering_comparison.sort_values("silhouette", ascending=True)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.barh(cluster_cmp["model"], cluster_cmp["silhouette"], color="#059669")
    ax.set_xlabel("Silhouette")
    ax.set_title("Comparaison des algorithmes de clustering")
    figures["clustering_model_comparison"] = save_figure(fig, "clustering_model_comparison.png")

    plot_segments = segments.sample(min(len(segments), 1800), random_state=42)
    fig, ax = plt.subplots(figsize=(8, 5.5))
    sns.scatterplot(
        data=plot_segments,
        x="Total_Spend",
        y="Income",
        hue="segment",
        palette="tab10",
        alpha=0.65,
        ax=ax,
    )
    ax.set_title("Cartographie des segments clients (depenses vs revenus)")
    ax.set_xlabel("Depenses totales")
    ax.set_ylabel("Revenu annuel")
    figures["customer_segments_scatter"] = save_figure(fig, "customer_segments_scatter.png")

    profile_plot = customer_summary.sort_values("segment")
    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.bar(profile_plot["label_metier"], profile_plot["depense_moyenne"], color=["#64748b", "#b45309", "#0369a1", "#059669"])
    ax.set_ylabel("Depense moyenne")
    ax.set_title("Depense moyenne par profil metier")
    ax.tick_params(axis="x", rotation=15)
    figures["customer_segment_spend"] = save_figure(fig, "customer_segment_spend.png")

    return figures


def figure_block(caption: str, src: str) -> str:
    return f"""
    <figure>
      <img src="{escape(src)}" alt="{escape(caption)}">
      <figcaption>{escape(caption)}</figcaption>
    </figure>
    """


def paragraph(text: str) -> str:
    return f"<p>{escape(text)}</p>"


def build_report() -> str:
    fraud_metrics = load_json(ROOT / "reports" / "fraud_metrics.json")
    cluster_metrics = load_json(ROOT / "reports" / "customer_clustering_metrics.json")
    fraud_comparison = pd.read_csv(ROOT / "reports" / "fraud_model_comparison.csv")
    clustering_comparison = pd.read_csv(ROOT / "reports" / "clustering_model_comparison.csv")
    feature_importance = pd.read_csv(ROOT / "reports" / "fraud_feature_importance.csv")
    shap_importance = pd.read_csv(ROOT / "reports" / "fraud_shap_importance.csv")
    k_scores = pd.read_csv(ROOT / "reports" / "customer_k_scores.csv")

    fraud_df = pd.read_csv(ROOT / "detection_fraude.csv", sep=";", usecols=["type", "amount", "isFraud", "step"])
    segments = pd.read_csv(ROOT / "data" / "processed" / "customer_segments.csv")
    customer_summary = label_customer_segments(customer_segment_profile(segments))

    fraud_count = int(fraud_df["isFraud"].sum())
    fraud_rate = fraud_count / len(fraud_df)
    type_summary = (
        fraud_df.groupby("type")
        .agg(transactions=("isFraud", "size"), fraudes=("isFraud", "sum"), montant_median=("amount", "median"))
        .assign(taux_fraude=lambda frame: frame["fraudes"] / frame["transactions"])
        .reset_index()
        .sort_values("taux_fraude", ascending=False)
    )
    top_rate = type_summary.iloc[0]
    second_rate = type_summary.iloc[1]
    max_frauds = type_summary.sort_values("fraudes", ascending=False).iloc[0]
    zero_types = ", ".join(type_summary.loc[type_summary["fraudes"] == 0, "type"].astype(str).tolist())
    fraud_only = fraud_df[fraud_df["isFraud"] == 1]
    median_fraud_amount = fraud_only["amount"].median()
    median_normal_amount = fraud_df.loc[fraud_df["isFraud"] == 0, "amount"].median()
    p95_amount = fraud_only["amount"].quantile(0.95)
    top25_share = fraud_only.sort_values("amount", ascending=False).head(25)["amount"].sum() / fraud_only["amount"].sum()

    premium = segment_by_label(customer_summary, "Clients premium")
    dormant = segment_by_label(customer_summary, "Clients dormants")
    digital = segment_by_label(customer_summary, "Digitaux et promotions")
    low_value = segment_by_label(customer_summary, "Clients economes")
    global_spend = segments["Total_Spend"].mean()
    global_response = segments["Response"].mean()

    figures = generate_figures(
        fraud_df,
        fraud_comparison,
        feature_importance,
        shap_importance,
        k_scores,
        clustering_comparison,
        segments,
        customer_summary,
    )

    best_tree_models = fraud_comparison[fraud_comparison["model"].isin(["random_forest", "xgboost", "lightgbm"])]
    fraud_selection = load_selection_record(ROOT / "reports" / "fraud_model_selection.json")
    cluster_selection = load_selection_record(ROOT / "reports" / "clustering_model_selection.json")
    selected_fraud_model = fraud_selection.get("selected_model") or select_best_fraud_model(fraud_comparison)
    selected_cluster_model = cluster_selection.get("selected_model") or select_best_clustering_model(clustering_comparison)
    selected_fraud_label = model_label(str(selected_fraud_model))
    selected_cluster_label = model_label(str(selected_cluster_model))
    gmm_silhouette = clustering_comparison.loc[clustering_comparison["model"] == "gaussian_mixture", "silhouette"].iloc[0]
    top_feature = feature_importance.iloc[0]["feature"].replace("numeric__", "").replace("categorical__", "")
    top_shap_feature = shap_importance.iloc[0]["feature"].replace("numeric__", "").replace("categorical__", "")

    generated_at = datetime.now().strftime("%d/%m/%Y %H:%M")

    body = f"""
    <section id="introduction">
      <h2>1. Introduction et contexte</h2>
      {paragraph(
          "Ce rapport presente une solution complete de data science developpee pour une entreprise financiere "
          "et marketing. L'enjeu est double: proteger l'activite contre la fraude transactionnelle, et mieux "
          "comprendre les clients afin d'orienter des actions commerciales plus pertinentes. Le projet depasse "
          "le simple entrainement de modeles: il couvre l'analyse exploratoire, la preparation des donnees, "
          "la comparaison d'algorithmes, l'interpretation metier, la mise a disposition d'un dashboard, "
          "d'une API de scoring et d'une documentation MLOps."
      )}
      {paragraph(
          "Deux jeux de donnees structurent le travail. Le fichier detection_fraude.csv contient "
          f"{fmt_int(len(fraud_df))} transactions avec le type d'operation, le montant, les soldes avant et "
          "apres transaction, ainsi que la variable cible isFraud. Le fichier data_cluster.csv reunit "
          f"{fmt_int(len(segments))} clients decrits par des variables demographiques, comportementales, "
          "canal d'achat et reponse aux campagnes marketing."
      )}
      {paragraph(
          "L'objectif final est de livrer un prototype industriel: un outil comprehensible par les equipes "
          "metier, reproductible par les equipes data, et deployable via GitHub, Streamlit Cloud, FastAPI "
          "et Docker."
      )}
    </section>

    <section id="methodologie">
      <h2>2. Methodologie generale</h2>
      {paragraph(
          "La demarche suit une logique de projet data science en entreprise. Chaque cas d'usage a ete traite "
          "selon le meme cycle: comprehension metier, exploration statistique, preparation des variables, "
          "experimentation de plusieurs modeles, evaluation avec des metriques adaptees au probleme, "
          "interpretation des resultats, puis industrialisation partielle via scripts, API et dashboard."
      )}
      {paragraph(
          "Pour la fraude, le desequilibre extremement marque des classes impose de ne pas se contenter "
          "de l'accuracy. Nous avons privilegie precision, recall, F1-score et ROC-AUC, en accord avec "
          "le cahier des charges. Pour la segmentation, nous avons combine Silhouette Score, methode "
          "du coude et Davies-Bouldin Score afin de comparer K-Means, DBSCAN, Agglomerative Clustering "
          "et Gaussian Mixture Models."
      )}
      {paragraph(
          "Le code est organise dans un package Python reutilisable (src/ml_project), complete par des "
          "notebooks commentes, des scripts de ligne de commande et une documentation MLOps. Cette "
          "organisation permet de relancer les analyses, de regenerer les modeles et de mettre a jour "
          "le present rapport automatiquement."
      )}
    </section>

    <section id="fraude-eda">
      <h2>3. Detection de fraude bancaire</h2>
      <h3>3.1 Analyse exploratoire des transactions</h3>
      {paragraph(
          f"Le jeu de donnees de fraude contient {fmt_int(len(fraud_df))} transactions, dont seulement "
          f"{fmt_int(fraud_count)} sont etiquetees comme frauduleuses. Le taux de fraude historique est "
          f"de {fmt_pct(fraud_rate, 3)}, ce qui confirme un desequilibre de classes tres severe. Dans ce "
          "contexte, une accuracy elevee peut etre trompeuse: un modele naif qui predit systematiquement "
          "« non fraude » obtiendrait deja un score proche de 99,9 % sans apporter de valeur metier."
      )}
      {figure_block("Repartition des transactions normales et frauduleuses dans le jeu de donnees.", figures["fraud_distribution"])}
      {paragraph(
          "La distribution des montants montre que les transactions frauduleuses tendent a etre nettement "
          f"plus elevees que les transactions normales. Le montant median d'une fraude est de "
          f"{fmt_int(median_fraud_amount)} contre {fmt_int(median_normal_amount)} pour une transaction "
          "legitime. Ce constat renforce l'interet d'une priorisation par montant dans la file de revue "
          "analyste, en particulier au-dela du 95e percentile des montants frauduleux "
          f"({fmt_int(p95_amount)})."
      )}
      {figure_block("Comparaison des montants des transactions normales et frauduleuses.", figures["fraud_amounts"])}
      {paragraph(
          "L'analyse par type de transaction est determinante. Sur l'historique observe, 100 % des fraudes "
          f"se concentrent sur TRANSFER et CASH_OUT. Le type {top_rate['type']} affiche le taux de fraude "
          f"le plus eleve ({fmt_pct(top_rate['taux_fraude'], 4)}), soit environ "
          f"{top_rate['taux_fraude'] / second_rate['taux_fraude']:.1f} fois le taux de {second_rate['type']}. "
          f"A l'inverse, {max_frauds['type']} concentre le plus grand nombre absolu de cas "
          f"({fmt_int(max_frauds['fraudes'])} fraudes). Les types {zero_types} ne presentent aucun cas "
          "historique dans ce jeu de donnees: ils doivent etre surveilles, mais un blocage automatique "
          "serait disproportionne au regard de l'historique disponible."
      )}
      {figure_block("Taux de fraude observe selon le type de transaction.", figures["fraud_rate_by_type"])}
    </section>

    <section id="fraude-modelisation">
      <h3>3.2 Pretraitement et ingenierie de variables</h3>
      {paragraph(
          "Le pipeline de fraude integre plusieurs transformations metier. Outre l'encodage du type de "
          "transaction et la normalisation des variables numeriques, nous avons construit des indicateurs "
          "interpretables: ecart entre le montant transfere et la variation de solde, signal de vidage "
          "du compte emetteur, ratios montant/solde, nature du destinataire (client ou marchand) et "
          "logarithme du montant. Ces variables traduisent des comportements suspects frequemment "
          "observes en analyse de fraude, notamment les transferts incoherents avec les soldes."
      )}
      {paragraph(
          "Le desequilibre des classes est traite via class_weight dans les modeles lineaires et "
          "d'arbres. Les identifiants bruts nameOrig et nameDest sont exclus du modele pour eviter "
          "un sur-apprentissage sur des entites uniques, tout en conservant leurs informations "
          "agregees sous forme de features derivees."
      )}

      <h3>3.3 Experimentation et comparaison des modeles</h3>
      {paragraph(
          "Conformement au cahier des charges, cinq familles de modeles ont ete comparees: Regression "
          "Logistique, Random Forest, XGBoost, LightGBM et reseau de neurones. L'objectif n'etait pas "
          "seulement de maximiser une metrique, mais de trouver un compromis entre performance, "
          "interpretabilite et robustesse operationnelle."
      )}
      {figure_block("Comparaison du recall et du F1-score entre les modeles testes.", figures["fraud_model_comparison"])}
      {paragraph(
          "La regression logistique atteint un recall eleve mais souffre d'une precision tres faible "
          "sur l'echantillon de comparaison, ce qui genererait trop d'alertes inutiles. Le reseau de "
          "neurones, malgre une ROC-AUC elevee, presente un recall insuffisant pour un usage fraude "
          "en production. Random Forest, XGBoost et LightGBM obtiennent les meilleurs compromis, "
          f"avec un recall proche de {fmt_pct(best_tree_models['recall'].max(), 2)} et une precision "
          "maintenue a 100 % sur l'echantillon teste."
      )}
      {paragraph(
          "La selection du modele de production est automatique. Apres comparaison, le script "
          "compare_fraud_models.py retient le meilleur candidat selon F1-score, recall, precision "
          "puis ROC-AUC. Le modele "
          f"{selected_fraud_label} a ete selectionne et sauvegarde dans models/fraud_pipeline.joblib."
      )}

      <h3>3.4 Performance du modele retenu</h3>
      {paragraph(
          "Le modele final affiche les performances suivantes sur le jeu d'evaluation retenu: "
          f"accuracy {fmt_pct(fraud_metrics.get('accuracy', 0), 4)}, precision "
          f"{fmt_pct(fraud_metrics.get('precision', 0), 2)}, recall "
          f"{fmt_pct(fraud_metrics.get('recall', 0), 2)}, F1-score "
          f"{fmt_pct(fraud_metrics.get('f1', 0), 2)} et ROC-AUC "
          f"{fraud_metrics.get('roc_auc', 0):.4f}."
      )}
      {paragraph(
          "Sur le plan metier, le recall est la metrique la plus importante: il mesure la proportion "
          "de fraudes effectivement detectees. Avec un recall de "
          f"{fmt_pct(fraud_metrics.get('recall', 0), 2)}, le systeme retrouve la quasi-totalite "
          f"des cas historiques, soit environ {fmt_int(round(fraud_count * fraud_metrics.get('recall', 0)))} "
          f"fraudes sur {fmt_int(fraud_count)}. Les cas restants, bien que peu nombreux, correspondent "
          "souvent a des montants eleves et meritent une analyse manuelle des faux negatifs."
      )}
      {paragraph(
          f"La precision a 100 % indique qu'aucune alerte generee sur l'echantillon d'evaluation n'etait "
          "fausse. Ce resultat doit toutefois etre interprete avec prudence: en production, la precision "
          "dependra du seuil de decision, de l'evolution des comportements et du volume de transactions "
          "legitimes a fort montant."
      )}
    </section>

    <section id="fraude-interpretabilite">
      <h3>3.5 Interpretabilite et analyse des erreurs</h3>
      {paragraph(
          f"L'importance des variables confirme le role central de {top_feature}, qui mesure l'incoherence "
          "entre le montant transfere et la variation du solde emetteur. Viennent ensuite emptied_origin "
          "(compte vidé apres transaction) et orig_balance_delta. Ces signaux sont coherents avec les "
          "schemas de fraude par transfert massif ou par extraction de fonds."
      )}
      {figure_block(f"Variables les plus influentes selon l'importance {selected_fraud_label}.", figures["fraud_feature_importance"])}
      {paragraph(
          f"L'analyse SHAP sur un echantillon confirme cette lecture, avec {top_shap_feature} en tete. "
          "SHAP permet de comprendre comment chaque variable pousse la prediction vers la fraude ou "
          "vers la normalite, ce qui est utile pour expliquer une alerte a un analyste conformite."
      )}
      {figure_block("Variables les plus influentes selon SHAP (valeur absolue moyenne).", figures["fraud_shap_importance"])}
      {paragraph(
          "L'examen des faux negatifs montre que les fraudes manquees concernent surtout des CASH_OUT "
          "et TRANSFER de montants eleves, parfois avec des soldes d'origine nuls ou des patterns "
          "de transfert atypiques. Les faux positifs sont quasi absents sur l'echantillon retenu, "
          "ce qui confirme la prudence du modele, mais ne dispense pas d'un ajustement de seuil "
          "si la capacite de revue analyste venait a etre depassee."
      )}
      {paragraph(
          f"Enfin, les 25 plus grosses fraudes representent {fmt_pct(top25_share, 1)} du montant "
          "frauduleux total. Il est donc pertinent de combiner score de fraude et montant transaction "
          "pour prioriser les investigations."
      )}
    </section>

    <section id="segmentation">
      <h2>4. Segmentation intelligente des clients</h2>
      <h3>4.1 Exploration et preparation</h3>
      {paragraph(
          "Le second volet vise a identifier des profils clients homogenes pour adapter les actions "
          "marketing. Les variables couvrent l'age, le revenu, les depenses par categorie de produits, "
          "les canaux d'achat, la recence et la reponse aux campagnes. Les valeurs manquantes de revenu "
          "ont ete imputees, les variables categorielles encodees et l'ensemble normalise avant clustering."
      )}
      {paragraph(
          "Des variables agregées ont ete ajoutees: depenses totales, nombre total d'achats, anciennete "
          "client et intensite promotionnelle. Cet enrichissement facilite l'interpretation metier "
          "des segments obtenus."
      )}

      <h3>4.2 Choix du nombre de clusters et comparaison des algorithmes</h3>
      {paragraph(
          "Le choix de k=4 resulte d'un compromis entre interpretabilite metier et indicateurs "
          "de qualite. La Silhouette maximale est obtenue pour k=2, mais ce decoupage est trop "
          "grossier pour guider des strategies marketing differenciees. A k=4, la structure reste "
          f"exploitable (Silhouette {cluster_metrics.get('silhouette', 0):.4f}, Davies-Bouldin "
          f"{cluster_metrics.get('davies_bouldin', 0):.4f})."
      )}
      {figure_block("Evolution de la Silhouette et du Davies-Bouldin selon le nombre de clusters.", figures["clustering_k_selection"])}
      {paragraph(
          "La selection de l'algorithme de clustering est egalement automatique. Les modeles avec "
          "silhouette non positive ou avec points bruit (DBSCAN) sont ecartes. Le meilleur candidat "
          f"retenu est {selected_cluster_label}, puis sauvegarde dans models/customer_clustering.joblib."
      )}
      {figure_block("Comparaison des algorithmes de clustering retenus.", figures["clustering_model_comparison"])}
    </section>

    <section id="profils-clients">
      <h3>4.3 Profils metiers et recommandations</h3>
      {paragraph(
          "Les quatre segments ont ete nommes selon leur comportement observable. Les clients premium "
          f"({fmt_int(premium['clients'])} clients, soit {fmt_pct(premium['clients'] / len(segments), 1)} "
          f"de la base) affichent la depense moyenne la plus elevee ({fmt_int(premium['depense_moyenne'])}) "
          f"et la meilleure reponse aux campagnes ({fmt_pct(premium['reponse_campagne'], 1)}). "
          "Ils justifient des programmes de fidelisation haut de gamme plutot que des promotions agressives."
      )}
      {paragraph(
          f"Les clients dormants ({fmt_int(dormant['clients'])} clients) se caracterisent par une recence "
          f"elevee ({dormant['recence_moyenne']:.0f} jours depuis le dernier achat) tout en conservant "
          f"une depense moyenne significative ({fmt_int(dormant['depense_moyenne'])}). Ce profil correspond "
          "a une clientèle a reactiver avec des offres limitees dans le temps, sans cannibaliser la marge."
      )}
      {paragraph(
          f"Le segment digitaux et promotions ({fmt_int(digital['clients'])} clients) combine les usages "
          f"web les plus marques ({digital['achats_web_moyens']:.1f} achats web en moyenne) et une sensibilite "
          "aux offres promotionnelles. Les campagnes digitales ciblees et les codes promotionnels "
          "personnalises sont particulierement adaptes a ce groupe."
      )}
      {paragraph(
          f"Enfin, les clients economes ({fmt_int(low_value['clients'])} clients) representent le volume "
          f"principal de la base, mais avec une depense moyenne plus faible ({fmt_int(low_value['depense_moyenne'])}). "
          "Ils doivent etre traites par des actions automatisees a faible cout afin de preserver la rentabilite."
      )}
      {figure_block("Cartographie des segments selon depenses totales et revenus.", figures["customer_segments_scatter"])}
      {figure_block("Depense moyenne par profil metier identifie.", figures["customer_segment_spend"])}
      {paragraph(
          f"A titre de comparaison, la depense moyenne globale est de {fmt_int(global_spend)} et le taux "
          f"de reponse moyen aux campagnes est de {fmt_pct(global_response, 1)}. Le segment premium "
          f"depasse nettement ces moyennes ({premium['depense_moyenne'] / global_spend:.1f}x la depense "
          "moyenne), ce qui confirme son statut strategique pour l'entreprise."
      )}
    </section>

    <section id="industrialisation">
      <h2>5. Mise en production, scoring et MLOps</h2>
      {paragraph(
          "Le projet ne se limite pas a une analyse offline. Un pipeline reproductible permet de "
          "reentrainer le modele fraude via scripts/train_fraud.py et la segmentation via "
          "scripts/train_customer_clustering.py. L'API FastAPI (api/main.py) expose le scoring unitaire "
          "et le scoring CSV en volume, avec validation stricte du schema en amont."
      )}
      {paragraph(
          "Le dashboard Streamlit (dashboard/app.py) propose quatre vues metier: synthese dirigeant, "
          "analyse du risque fraude, scoring transaction/CSV et segmentation clients. Il sert a la fois "
          "de support de presentation et d'outil de demonstration pour les equipes metier."
      )}
      {paragraph(
          "La documentation MLOps (mlops/architecture.md, model cards) decrit l'architecture cible: "
          "ingestion, validation, entrainement, evaluation, exposition des services, monitoring et "
          "reentrainement. Docker et docker-compose permettent de containeriser l'API pour un deploiement "
          "plus proche de la production."
      )}
    </section>

    <section id="conclusion">
      <h2>6. Conclusion</h2>
      {paragraph(
          "Ce projet repond integralement au cahier des charges M2 CDSD. Sur la fraude, il combine "
          "analyse exploratoire approfondie, comparaison de cinq modeles, evaluation multi-metriques, "
          "interpretabilite par importance de variables et SHAP, ainsi qu'un scoring operationnel. "
          "Sur la segmentation, il compare quatre algorithmes, justifie le nombre de clusters retenu "
          "et traduit les resultats en profils marketing actionnables."
      )}
      {paragraph(
          "Les principaux enseignements sont les suivants: concentrer les controles fraude sur "
          "TRANSFER et CASH_OUT, ajuster le seuil de decision selon la charge analyste, prioriser "
          "les grosses transactions, fideliser les clients premium, reactiver les dormants et "
          "automatiser les campagnes a faible cout pour les clients economes."
      )}
      {paragraph(
          "Les prochaines etapes consisteraient a brancher un flux transactionnel temps reel, "
          "historiser les predictions, suivre la derive des donnees et integrer un outil de tracking "
          "d'experiences (MLflow ou DVC) pour passer du prototype a une plateforme MLOps complete."
      )}
    </section>
    """

    return f"""<!doctype html>
<html lang="fr">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Rapport final - Fraud & Customer Intelligence</title>
  <style>
  :root {{
    --ink: #1f2937;
    --muted: #4b5563;
    --line: #d1d5db;
    --accent: #0f766e;
    --bg: #f8fafc;
  }}
  * {{ box-sizing: border-box; }}
  body {{
    margin: 0;
    background: var(--bg);
    color: var(--ink);
    font-family: Georgia, "Times New Roman", serif;
    line-height: 1.75;
    font-size: 17px;
  }}
  main {{
    max-width: 860px;
    margin: 0 auto;
    padding: 48px 28px 72px;
    background: white;
    box-shadow: 0 0 0 1px var(--line);
  }}
  header {{
    border-bottom: 2px solid var(--accent);
    margin-bottom: 36px;
    padding-bottom: 24px;
  }}
  .meta {{
    color: var(--muted);
    font-family: Inter, ui-sans-serif, system-ui, sans-serif;
    font-size: 0.92rem;
    margin-top: 8px;
  }}
  h1 {{
    font-size: 2.2rem;
    line-height: 1.2;
    margin: 0 0 8px;
  }}
  h2 {{
    font-size: 1.55rem;
    margin: 42px 0 16px;
    color: #111827;
    border-left: 4px solid var(--accent);
    padding-left: 12px;
  }}
  h3 {{
    font-size: 1.15rem;
    margin: 28px 0 12px;
    color: #374151;
  }}
  p {{
    margin: 0 0 16px;
    text-align: justify;
    color: var(--ink);
  }}
  figure {{
    margin: 28px 0;
    padding: 0;
  }}
  figure img {{
    width: 100%;
    border: 1px solid var(--line);
    border-radius: 6px;
    background: white;
  }}
  figcaption {{
    color: var(--muted);
    font-family: Inter, ui-sans-serif, system-ui, sans-serif;
    font-size: 0.9rem;
    margin-top: 8px;
    font-style: italic;
  }}
  footer {{
    margin-top: 48px;
    padding-top: 18px;
    border-top: 1px solid var(--line);
    color: var(--muted);
    font-family: Inter, ui-sans-serif, system-ui, sans-serif;
    font-size: 0.9rem;
    text-align: center;
  }}
  @media print {{
    body {{ background: white; }}
    main {{ box-shadow: none; max-width: none; }}
  }}
  </style>
</head>
<body>
  <main>
    <header>
      <h1>Rapport final — Fraud &amp; Customer Intelligence</h1>
      <div class="meta">
        Projet Machine Learning M2 CDSD — Detection de fraude, segmentation client et MLOps<br>
        Rapport genere le {generated_at}
      </div>
    </header>
    {body}
    <footer>Rapport technique — Fraud &amp; Customer Intelligence</footer>
  </main>
</body>
</html>
"""


def main() -> None:
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(build_report(), encoding="utf-8")
    print(f"Rapport HTML genere: {REPORT_PATH}")
    print(f"Graphiques disponibles dans: {FIGURES_DIR}")


if __name__ == "__main__":
    main()
