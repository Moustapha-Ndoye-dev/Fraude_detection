# Rapport final - Fraud & Customer Intelligence

## 1. Contexte et objectif

Ce projet met en place une solution de machine learning exploitable par une entreprise pour deux besoins metiers:

- detecter automatiquement les transactions potentiellement frauduleuses;
- segmenter les clients afin de guider des actions marketing differenciees.

L'objectif n'est pas seulement de produire des modeles, mais de livrer une solution complete: donnees analysees, modeles entraines, visualisations, dashboard, scoring en volume, documentation MLOps, presentation et depot GitHub deployable.

## 2. Donnees utilisees

Deux jeux de donnees sont exploites:

- `detection_fraude.csv`: transactions financieres avec montant, type de transaction, soldes avant/apres et variable cible `isFraud`;
- `data_cluster.csv`: donnees clients pour segmentation marketing.

Statistiques principales:

- transactions analysees: 1 048 575;
- fraudes observees: 1 142;
- taux de fraude historique: 0,109%;
- clients segmentes: 2 240.

## 3. Detection de fraude

Le modele de detection de fraude est entraine sous forme de pipeline reproductible. Il integre la preparation des variables, l'encodage des categories et le classifieur.

Resultats du modele:

| Metrique | Valeur |
| --- | ---: |
| Accuracy | 99,9976% |
| Precision | 100,00% |
| Recall | 97,81% |
| F1-score | 98,89% |
| ROC-AUC | 0,9981 |

Interpretation metier:

- le taux de fraude est tres faible, donc l'accuracy seule n'est pas suffisante;
- le recall est prioritaire car une fraude non detectee coute cher a l'entreprise;
- les types `TRANSFER` et `CASH_OUT` concentrent le risque;
- le seuil de decision doit rester configurable selon le cout des faux positifs et faux negatifs.

## 4. Segmentation client

La segmentation client repose sur un clustering non supervise. Les variables client sont preparees, normalisees et exploitees pour obtenir des groupes actionnables.

Resultats:

| Indicateur | Valeur |
| --- | ---: |
| Clients segmentes | 2 240 |
| Nombre de segments retenu | 4 |
| Silhouette | 0,1764 |
| Davies-Bouldin | 2,0898 |

Profils metiers:

- clients premium;
- clients dormants;
- clients economes;
- clients digitaux et sensibles aux promotions.

Interpretation metier:

- les clients premium sont prioritaires pour la fidelisation;
- les clients dormants doivent recevoir des campagnes de reactivation;
- les clients sensibles aux promotions sont utiles pour les campagnes ciblees;
- les segments doivent etre suivis dans le temps pour mesurer leur stabilite.

## 5. Dashboard

Le dashboard Streamlit est disponible dans:

```text
dashboard/app.py
```

Il contient:

- une page `01 - Synthese dirigeant` pour les indicateurs dirigeants;
- une page `02 - Analyse du risque fraude` pour analyser les fraudes par type, periode et montant;
- une page `03 - Scoring transaction et CSV` pour saisir une transaction ou importer un CSV;
- une page `04 - Segmentation clients` pour visualiser les profils clients.

La partie industrialisation est livree dans le depot sous forme de scripts, API, Dockerfile et documentation, mais elle n'est plus affichee comme page descriptive dans le dashboard metier.

Une version HTML du rapport est disponible dans:

```text
reports/rapport_final.html
```

Le dashboard utilise des composants React embarques pour les headers, KPI et lectures business, tout en gardant les formulaires, tableaux et uploads CSV en Streamlit natif pour la stabilite.

## 6. Scoring operationnel

La solution permet deux modes d'utilisation:

- saisie manuelle d'une transaction;
- import CSV en volume.

Le fichier CSV est controle avant scoring:

- colonnes obligatoires;
- types attendus;
- colonnes non attendues rejetables;
- valeurs numeriques controlees.

La sortie contient:

- probabilite de fraude;
- prediction binaire;
- niveau de risque;
- action recommandee;
- export CSV score.

## 7. MLOps et industrialisation

Le projet inclut une base MLOps:

- architecture documentee dans `mlops/architecture.md`;
- model card fraude dans `mlops/model_card_fraud.md`;
- model card segmentation dans `mlops/model_card_clustering.md`;
- scripts reproductibles d'entrainement et de scoring;
- API FastAPI pour exposer le scoring;
- Dockerfile et docker-compose pour industrialisation.

Architecture cible:

```text
Sources donnees
    -> Validation schema
    -> Feature engineering
    -> Entrainement modele
    -> Evaluation
    -> Sauvegarde modele
    -> API / Dashboard
    -> Monitoring
    -> Reentrainement
```

## 8. Depot GitHub

Le depot contient une architecture propre:

- `src/` pour le package Python;
- `scripts/` pour les commandes reproductibles;
- `dashboard/` pour l'application Streamlit;
- `api/` pour le service FastAPI;
- `models/` pour les modeles necessaires a la demo;
- `reports/` pour les metriques;
- `mlops/` pour la documentation industrielle;
- `README.md` pour le lancement et le deploiement.

Le fichier principal a utiliser sur Streamlit Cloud est:

```text
dashboard/app.py
```

## 9. Limites et ameliorations possibles

Limites:

- les donnees semblent etre un jeu d'etude et non un flux transactionnel temps reel;
- le taux de fraude tres faible impose un suivi attentif des faux positifs;
- les segments doivent etre valides avec les equipes marketing avant exploitation commerciale.

Ameliorations:

- ajout d'un suivi de drift;
- historisation des predictions;
- monitoring des performances apres retour analyste;
- integration MLflow ou DVC;
- deploiement API separe pour un usage SI reel;
- authentification et gestion des droits utilisateurs.

## 10. Conclusion

Le projet repond au cahier des charges: detection de fraude, segmentation client, visualisations, interpretation metier, dashboard operationnel, base MLOps, presentation finale et depot GitHub deployable.

La solution peut etre presentee comme un prototype industriel: elle transforme des donnees brutes en outil d'aide a la decision pour les equipes risque, conformite, data et marketing.
