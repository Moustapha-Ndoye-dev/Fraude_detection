# Fraude Detection

Projet de Machine Learning pour deux cas d'usage:

- detection de fraude bancaire avec apprentissage supervise;
- segmentation client avec clustering non supervise;
- industrialisation via une approche MLOps.

## Architecture

```text
.
|-- api/                    # API FastAPI pour servir les predictions fraude
|-- config/                 # Parametres projet
|-- dashboard/              # Application Streamlit
|-- data/                   # Donnees traitees utiles au dashboard
|-- mlops/                  # Architecture, model cards, monitoring
|-- models/                 # Artefacts modeles sauvegardes
|-- notebooks/              # Notebooks d'analyse et d'experimentation
|-- reports/                # Metriques, figures et exports
|-- scripts/                # Commandes reproductibles
|-- src/ml_project/         # Package Python principal
`-- tests/                  # Tests unitaires
```

## Livrables

- Rapport final narratif: `RAPPORT_FINAL.md` (texte + graphiques)
- Rapport HTML: `reports/rapport_final.html` (generer via `python scripts/generate_html_report.py`)
- Dashboard Streamlit: `dashboard/app.py`
- Presentation finale: `PRESENTATION_ENTREPRISE.md`
- Depot GitHub deployable: architecture projet, dependances, modeles, donnees utiles, scripts et documentation
- API FastAPI: `api/main.py`
- Documentation MLOps: `mlops/`
- Notebooks commentes: `notebooks/`
- Comparaisons modeles: `reports/fraud_model_comparison.csv` et `reports/clustering_model_comparison.csv`
- Interpretabilite fraude: `reports/fraud_feature_importance.csv`, echantillons FP/FN et note SHAP

Les fichiers fournis par le cahier des charges restent lisibles a la racine:

- `detection_fraude.csv`
- `data_cluster.csv`
- `projet_machine_learning_m2CDSD.docx` (cahier des charges de reference)

Les artefacts necessaires au deploiement Streamlit Cloud sont versionnes:

- `models/fraud_pipeline.joblib`
- `models/customer_clustering.joblib`
- `data/processed/customer_segments.csv`
- `reports/customer_k_scores.csv`

## Installation

```bash
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Verifier l'environnement:

```bash
python scripts/check_environment.py
```

## Analyse rapide des donnees

Le script suivant fonctionne sans dependances externes:

```bash
python scripts/profile_data.py
```

## Entrainement fraude

Comparaison puis selection automatique du meilleur modele :

```bash
python scripts/compare_fraud_models.py --nrows 200000
```

Entrainement avec le meilleur modele identifie (`--model best` par defaut) :

```bash
python scripts/train_fraud.py
```

Entrainement manuel :

```bash
python scripts/train_fraud.py --model random_forest --nrows 200000
```

Baseline interpretable:

```bash
python scripts/train_fraud.py --model logistic_regression --nrows 200000
```

Comparaison des modeles demandes dans le cahier des charges:

```bash
python scripts/compare_fraud_models.py --nrows 200000
```

Le script compare les modeles, selectionne automatiquement le meilleur et sauvegarde
`models/fraud_pipeline.joblib` sur l'integralite des transactions.

Le script compare:

- Regression Logistique;
- Random Forest;
- XGBoost;
- LightGBM;
- reseau de neurones.

Pour XGBoost et LightGBM:

```bash
python -m pip install -r requirements-optional.txt
```

Sorties:

- `models/fraud_pipeline.joblib`
- `reports/fraud_metrics.json`
- `reports/fraud_model_comparison.csv`
- `reports/fraud_model_selection.json`

Interpretabilite, importance des variables et analyse FP/FN:

```bash
python scripts/interpret_fraud_model.py
```

Sorties:

- `reports/fraud_feature_importance.csv`
- `reports/fraud_shap_importance.csv`
- `reports/fraud_false_positives_sample.csv`
- `reports/fraud_false_negatives_sample.csv`
- `reports/fraud_shap_note.json`

## Segmentation client

Recherche du nombre de clusters:

```bash
python scripts/train_customer_clustering.py --search-k
```

Comparaison puis selection automatique du meilleur algorithme :

```bash
python scripts/compare_clustering_models.py
```

Entrainement avec le meilleur algorithme identifie :

```bash
python scripts/train_customer_clustering.py
```

Entrainement manuel :

```bash
python scripts/train_customer_clustering.py --model kmeans --clusters 4
```

Le script compare les algorithmes, selectionne automatiquement le meilleur
(silhouette, Davies-Bouldin, sans bruit DBSCAN) et sauvegarde les artefacts.

Sorties:

- `models/customer_clustering.joblib`
- `data/processed/customer_segments.csv`
- `reports/customer_clustering_metrics.json`
- `reports/clustering_model_comparison.csv`
- `reports/clustering_model_selection.json`

## API

```bash
uvicorn api.main:app --reload
```

Endpoints:

- `GET /health`
- `POST /predict/fraud`
- `GET /schema/fraud`
- `POST /score/fraud/csv/summary`
- `POST /score/fraud/csv`

Scoring CSV en volume:

```bash
curl -X POST "http://127.0.0.1:8000/score/fraud/csv/summary?delimiter=;&threshold=0.5&strict=true" \
  -F "file=@transactions_a_scorer.csv"
```

Le fichier est rejete si les colonnes attendues ou les types ne sont pas conformes.

Scoring batch en terminal:

```bash
python scripts/score_fraud_csv.py transactions_a_scorer.csv --output data/processed/fraud_scoring_results.csv
```

## Dashboard

```bash
streamlit run dashboard/app.py
```

Sur Streamlit Community Cloud, utiliser:

```text
Main file path: dashboard/app.py
```

Le dashboard contient quatre sections calquees sur le rapport final:

- `1 · Introduction et contexte`;
- `2 · Detection de fraude bancaire`;
- `3 · Scoring operationnel`;
- `4 · Segmentation client`.

Le design utilise des composants React embarques dans Streamlit pour les headers, les KPI et les blocs de lecture business. Le bundle de production est deja versionne dans:

```text
dashboard/components/fraud_widgets/dist/fraud-widgets.js
```

Pour modifier ces composants React:

```bash
cd dashboard/components/fraud_widgets
npm install
npm run build
```

Generer le rapport HTML:

```bash
python scripts/generate_html_report.py
```

## Deploiement Streamlit Cloud

1. Connecter le depot GitHub `Moustapha-Ndoye-dev/Fraude_detection`.
2. Choisir la branche `main`.
3. Mettre `dashboard/app.py` dans `Main file path`.
4. Attendre le redeploiement apres chaque `git push`.

Si l'ancienne interface apparait encore, ouvrir `Manage app`, puis lancer `Reboot app` ou `Clear cache`.

## Strategie d'analyse

Fraude:

- analyser le desequilibre de `isFraud`;
- comparer les types de transactions;
- surveiller precision, recall, F1 et ROC-AUC;
- ajuster le seuil selon le cout metier des faux positifs et faux negatifs;
- expliquer le modele avec importances et SHAP.

Segmentation:

- imputer `Income`;
- encoder `Education` et `Marital_Status`;
- normaliser les variables numeriques;
- comparer K-Means, DBSCAN, Agglomerative Clustering et Gaussian Mixture;
- nommer les segments selon leur interpretation metier.

## MLOps

La documentation MLOps est dans `mlops/`:

- `architecture.md`
- `model_card_fraud.md`
- `model_card_clustering.md`

Le projet est pret pour une evolution vers MLflow, DVC, Docker et CI/CD.
