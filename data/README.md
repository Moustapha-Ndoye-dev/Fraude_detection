# Donnees

Les fichiers fournis sont conserves a la racine du projet par defaut:

- `detection_fraude.csv`
- `data_cluster.csv`
- `projet_machine_learning_m2CDSD.docx`

Les chemins peuvent etre modifies via `.env` ou `config/settings.yaml`.

Bonnes pratiques:

- garder les donnees brutes hors Git;
- placer les exports nettoyes dans `data/processed/`;
- versionner la logique de preparation, pas les gros fichiers generes.
