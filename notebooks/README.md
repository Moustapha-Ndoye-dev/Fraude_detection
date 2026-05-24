# Notebooks fournis

Ordre conseille:

1. `01_eda_fraude.ipynb` - exploration des transactions, desequilibre, montants et types.
2. `02_modelisation_fraude.ipynb` - preprocessing, modeles, evaluation, interpretation.
3. `03_eda_segmentation.ipynb` - exploration des clients, revenus, depenses et canaux.
4. `04_clustering_clients.ipynb` - K-Means, DBSCAN, clustering hierarchique, GMM.
5. `05_mlops_synthese.ipynb` - architecture, monitoring et demo des artefacts.

Ces notebooks sont volontairement relies aux scripts reproductibles du projet. Les analyses peuvent donc etre relancees depuis les notebooks ou directement depuis le terminal.

Commandes utiles:

```bash
python scripts/compare_fraud_models.py --nrows 200000
python scripts/interpret_fraud_model.py
python scripts/compare_clustering_models.py
python scripts/generate_html_report.py
```
