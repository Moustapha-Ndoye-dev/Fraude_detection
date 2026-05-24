# Model Card - Segmentation client

## Usage

Segmenter les clients selon leur profil demographique, leurs depenses et leurs canaux d'achat.

## Donnees

Source: `data_cluster.csv`.

Pas de cible supervisee pour l'entrainement.

## Evaluation

- Silhouette score.
- Davies-Bouldin score.
- Elbow method.
- Lecture metier des segments.

## Risques

- Segments statistiques non actionnables.
- Sensibilite aux variables d'echelle et aux valeurs aberrantes.
- Revenus manquants a imputer proprement.

## Recommandation

Choisir le nombre de segments en combinant scores numeriques et interpretation marketing.
