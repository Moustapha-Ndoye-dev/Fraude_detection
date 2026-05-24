# Architecture MLOps

## Objectif

Industrialiser deux cas d'usage:

- detection de fraude bancaire en classification supervisee;
- segmentation client en clustering non supervise.

## Flux cible

1. Ingestion des fichiers CSV fournis.
2. Validation du schema et controles qualite.
3. Preparation des variables et feature engineering.
4. Entrainement des modeles.
5. Evaluation et generation des rapports.
6. Sauvegarde des artefacts dans `models/` et `reports/`.
7. Exposition via API FastAPI et dashboard Streamlit.
8. Monitoring des donnees, performances et derives.

## Versionning

- Code: Git/GitHub.
- Donnees: DVC ou stockage objet versionne.
- Modeles: MLflow Model Registry ou artefacts versionnes.
- Parametres: fichiers YAML et logs d'experiences.

## Deploiement

- API: FastAPI + Uvicorn.
- Dashboard: Streamlit.
- Conteneurisation: Docker.
- CI/CD: tests, lint, build image, deploiement.

## Monitoring

Fraude:

- taux de prediction positive;
- precision, recall et F1 si labels disponibles;
- derive des montants, types et soldes;
- suivi des faux positifs/faux negatifs.

Clustering:

- repartition des segments;
- stabilite des centroïdes;
- derive des revenus, depenses et canaux;
- coherence metier des profils.
