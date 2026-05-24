# Model Card - Detection de fraude

## Usage

Classifier une transaction comme normale ou frauduleuse a partir des variables de transaction.

## Donnees

Source: `detection_fraude.csv`.

Cible: `isFraud`.

## Risques

- Fort desequilibre des classes.
- Accuracy peu informative seule.
- Cout metier asymetrique: un faux negatif est plus grave qu'un faux positif.
- Les identifiants clients ne doivent pas etre utilises tels quels pour generaliser.

## Metriques prioritaires

1. Recall fraude.
2. Precision fraude.
3. F1-score.
4. ROC-AUC.

## Recommandation

Comparer une baseline interpretable avec un modele arbre, puis ajuster le seuil de decision selon le cout metier.
