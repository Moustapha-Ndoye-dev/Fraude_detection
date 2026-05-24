# Rapports

Ce dossier recoit les sorties d'analyse:

- metriques JSON;
- tableaux CSV;
- graphiques dans `figures/`;
- rapport final HTML narratif: `rapport_final.html`;
- comparaisons de modeles fraude et clustering;
- importance des variables, SHAP et analyse FP/FN.

Generer le rapport complet (texte + graphiques):

```bash
python scripts/generate_html_report.py
```

Les figures sont produites automatiquement dans `reports/figures/`.
