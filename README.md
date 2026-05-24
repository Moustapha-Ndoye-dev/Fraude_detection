# Fraude Detection

Projet de Machine Learning pour deux cas d'usage:

- detection de fraude bancaire avec apprentissage supervise;
- segmentation client avec clustering non supervise;
- industrialisation via une approche MLOps.

## Architecture

```text
.
├── api/                    # API FastAPI pour servir les predictions fraude
├── config/                 # Parametres projet
├── dashboard/              # Application Streamlit
├── data/                   # Donnees brutes/traitees (hors Git)
├── mlops/                  # Architecture, model cards, monitoring
├── models/                 # Artefacts modeles sauvegardes
├── notebooks/              # Notebooks d'analyse et d'experimentation
├── reports/                # Metriques, figures et exports
├── scripts/                # Commandes reproductibles
├── src/ml_project/         # Package Python principal
└── tests/                  # Tests unitaires
```

Les fichiers fournis restent lisibles a la racine:

- `detection_fraude.csv`
- `data_cluster.csv`
- `projet_machine_learning_m2CDSD.docx`

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

Baseline Random Forest:

```bash
python scripts/train_fraud.py --model random_forest --nrows 200000
```

Baseline interpretable:

```bash
python scripts/train_fraud.py --model logistic_regression --nrows 200000
```

Sorties:

- `models/fraud_pipeline.joblib`
- `reports/fraud_metrics.json`

## Segmentation client

Recherche du nombre de clusters:

```bash
python scripts/train_customer_clustering.py --search-k
```

Entrainement K-Means:

```bash
python scripts/train_customer_clustering.py --model kmeans --clusters 4
```

Sorties:

- `models/customer_clustering.joblib`
- `data/processed/customer_segments.csv`
- `reports/customer_clustering_metrics.json`

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
Main file path: streamlit_app.py
```

Le dashboard contient:

- une synthese executive;
- une vue risque fraude;
- un outil de scoring operationnel avec saisie transaction et import CSV;
- une vue segmentation client;
- une page MLOps et exploitation.

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
