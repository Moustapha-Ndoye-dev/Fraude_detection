# Checklist de conformite au cahier des charges

Reference officielle: `projet_machine_learning_m2CDSD.docx` (Exercice 1, Exercice 2, MLOps, livrables).

Statut global: **conforme** — tous les points du cahier des charges sont couverts par le depot.

## Exercice 1 - Detection de fraude bancaire

| Exigence | Statut | Livrable |
| --- | --- | --- |
| Analyse exploratoire des transactions, montants, comportements suspects | Fait | `notebooks/01_eda_fraude.ipynb`, `scripts/profile_data.py`, dashboard |
| Visualisations fraude | Fait | `dashboard/app.py`, `reports/rapport_final.html` |
| Pretraitement: encodage, normalisation, desequilibre | Fait | `src/ml_project/features/fraud.py`, `src/ml_project/models/fraud.py` |
| Regression Logistique | Fait | `scripts/compare_fraud_models.py`, `reports/fraud_model_comparison.csv` |
| Random Forest | Fait | `scripts/train_fraud.py`, `models/fraud_pipeline.joblib` |
| XGBoost | Fait | `scripts/compare_fraud_models.py`, dependance optionnelle `requirements-optional.txt` |
| LightGBM | Fait | `scripts/compare_fraud_models.py`, dependance optionnelle `requirements-optional.txt` |
| Reseau de neurones | Fait | `src/ml_project/models/fraud.py`, `reports/fraud_model_comparison.csv` |
| Evaluation Accuracy, Precision, Recall, F1, ROC-AUC | Fait | `reports/fraud_metrics.json`, `reports/fraud_model_comparison.csv` |
| Importance des variables | Fait | `scripts/interpret_fraud_model.py`, `reports/fraud_feature_importance.csv` |
| Analyse faux positifs / faux negatifs | Fait | `reports/fraud_false_positives_sample.csv`, `reports/fraud_false_negatives_sample.csv` |
| SHAP | Fait | `scripts/interpret_fraud_model.py`, `reports/fraud_shap_importance.csv`, `reports/fraud_shap_note.json` |

## Exercice 2 - Segmentation client

| Exigence | Statut | Livrable |
| --- | --- | --- |
| Analyse exploratoire clients, revenus, achats | Fait | `notebooks/03_eda_segmentation.ipynb` |
| Pretraitement: encodage, normalisation, valeurs manquantes, PCA eventuelle | Fait | `src/ml_project/features/customers.py`, `src/ml_project/models/clustering.py` (PCA non retenue, normalisation suffisante) |
| K-Means | Fait | `scripts/train_customer_clustering.py`, `models/customer_clustering.joblib` |
| DBSCAN | Fait | `scripts/compare_clustering_models.py` |
| Agglomerative Clustering | Fait | `scripts/compare_clustering_models.py` |
| Gaussian Mixture Models | Fait | `scripts/compare_clustering_models.py` |
| Silhouette Score | Fait | `reports/customer_k_scores.csv`, `reports/clustering_model_comparison.csv` |
| Elbow / choix de k | Fait | `reports/customer_k_scores.csv`, `notebooks/04_clustering_clients.ipynb` |
| Davies-Bouldin Score | Fait | `reports/customer_clustering_metrics.json`, `reports/clustering_model_comparison.csv` |
| Profils premium, digitaux, promotions, dormants | Fait | `dashboard/app.py`, `data/processed/customer_segments.csv` |
| Recommandations business | Fait | `dashboard/app.py`, `RAPPORT_FINAL.md`, `reports/rapport_final.html` |

## Partie commune - MLOps

| Exigence | Statut | Livrable |
| --- | --- | --- |
| Pipeline donnees | Fait | `src/ml_project/data/`, `src/ml_project/features/` |
| Validation schema | Fait | `src/ml_project/data/loaders.py`, `src/ml_project/serving/fraud_scoring.py` |
| Versioning donnees/modeles/parametres | Fait | GitHub, `models/`, `reports/`, `config/settings.yaml` |
| Deploiement FastAPI | Fait | `api/main.py` (FastAPI retenu plutot que Flask) |
| Deploiement Streamlit | Fait | `dashboard/app.py` (Streamlit retenu plutot que Dash) |
| Docker | Fait | `Dockerfile`, `docker-compose.yml` |
| Monitoring et drift | Documente | `mlops/architecture.md`, `notebooks/05_mlops_synthese.ipynb` |
| CI/CD simplifie | Documente | `mlops/architecture.md`, `README.md` |

## Livrables finaux

| Livrable demande | Statut | Fichier |
| --- | --- | --- |
| Notebooks structures et commentes | Fait | `notebooks/*.ipynb` |
| Analyses detaillees | Fait | `RAPPORT_FINAL.md`, notebooks |
| Visualisations professionnelles | Fait | `dashboard/app.py`, `reports/rapport_final.html` |
| Rapport technique | Fait | `RAPPORT_FINAL.md` |
| Rapport HTML | Fait | `reports/rapport_final.html` |
| Interpretation metier | Fait | dashboard, rapport, presentation |
| Architecture MLOps | Fait | `mlops/architecture.md` |
| Dashboard interactif | Fait | `dashboard/app.py` |
| Presentation finale | Fait | `PRESENTATION_ENTREPRISE.md` |
| Depot GitHub propre | Fait | `README.md`, structure projet, dependances, scripts |
