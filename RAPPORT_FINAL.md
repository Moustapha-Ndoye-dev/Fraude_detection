# Rapport final — Fraud & Customer Intelligence

**Projet Machine Learning M2 CDSD**  
Détection de fraude bancaire, segmentation client et industrialisation MLOps

---

## 1. Introduction et contexte

Ce rapport présente une solution complète de data science développée pour une entreprise financière et marketing. L'enjeu est double : protéger l'activité contre la fraude transactionnelle, et mieux comprendre les clients afin d'orienter des actions commerciales plus pertinentes.

Le projet dépasse le simple entraînement de modèles. Il couvre l'analyse exploratoire, la préparation des données, la comparaison d'algorithmes, l'interprétation métier, la mise à disposition d'un dashboard, d'une API de scoring et d'une documentation MLOps. L'objectif final est de livrer un **prototype industriel** : un outil compréhensible par les équipes métier, reproductible par les équipes data, et déployable via GitHub, Streamlit Cloud, FastAPI et Docker.

Deux jeux de données structurent le travail :

- **detection_fraude.csv** : 1 048 575 transactions avec le type d'opération, le montant, les soldes avant et après transaction, ainsi que la variable cible `isFraud`.
- **data_cluster.csv** : 2 240 clients décrits par des variables démographiques, comportementales, canaux d'achat et réponse aux campagnes marketing.

---

## 2. Méthodologie générale

La démarche suit une logique de projet data science en entreprise. Chaque cas d'usage a été traité selon le même cycle : compréhension métier, exploration statistique, préparation des variables, expérimentation de plusieurs modèles, évaluation avec des métriques adaptées au problème, interprétation des résultats, puis industrialisation partielle via scripts, API et dashboard.

Pour la fraude, le déséquilibre extrêmement marqué des classes impose de ne pas se contenter de l'accuracy. Nous avons privilégié la précision, le recall, le F1-score et la ROC-AUC, conformément au cahier des charges. Pour la segmentation, nous avons combiné Silhouette Score, méthode du coude et Davies-Bouldin Score afin de comparer K-Means, DBSCAN, Agglomerative Clustering et Gaussian Mixture Models.

Le code est organisé dans un package Python réutilisable (`src/ml_project`), complété par des notebooks commentés, des scripts de ligne de commande et une documentation MLOps. Cette organisation permet de relancer les analyses, de régénérer les modèles et de mettre à jour le rapport HTML via `python scripts/generate_html_report.py`.

---

## 3. Détection de fraude bancaire

### 3.1 Analyse exploratoire des transactions

Le jeu de données de fraude contient 1 048 575 transactions, dont seulement 1 142 sont étiquetées comme frauduleuses. Le taux de fraude historique est de **0,109 %**, ce qui confirme un déséquilibre de classes très sévère. Dans ce contexte, une accuracy élevée peut être trompeuse : un modèle naïf qui prédit systématiquement « non fraude » obtiendrait déjà un score proche de 99,9 % sans apporter de valeur métier.

![Distribution de la variable cible isFraud](reports/figures/fraud_distribution.png)

La distribution des montants montre que les transactions frauduleuses tendent à être nettement plus élevées que les transactions normales. Le montant médian d'une fraude est d'environ **353 179** contre **76 215** pour une transaction légitime. Ce constat renforce l'intérêt d'une priorisation par montant dans la file de revue analyste.

![Comparaison des montants selon la classe de fraude](reports/figures/fraud_amounts.png)

L'analyse par type de transaction est déterminante. Sur l'historique observé, **100 % des fraudes** se concentrent sur `TRANSFER` et `CASH_OUT`. Le type `TRANSFER` affiche le taux de fraude le plus élevé (**0,6501 %**), soit environ 4,2 fois le taux de `CASH_OUT`. À l'inverse, `CASH_OUT` concentre le plus grand nombre absolu de cas (**578 fraudes**). Les types `PAYMENT`, `CASH_IN` et `DEBIT` ne présentent aucun cas historique : ils doivent être surveillés, mais un blocage automatique serait disproportionné au regard de l'historique disponible.

![Taux de fraude par type de transaction](reports/figures/fraud_rate_by_type.png)

### 3.2 Prétraitement et ingénierie de variables

Le pipeline de fraude intègre plusieurs transformations métier. Outre l'encodage du type de transaction et la normalisation des variables numériques, nous avons construit des indicateurs interprétables :

- écart entre le montant transféré et la variation de solde (`orig_delta_error`) ;
- signal de vidage du compte émetteur (`emptied_origin`) ;
- ratios montant/solde et logarithme du montant ;
- nature du destinataire (client ou marchand).

Ces variables traduisent des comportements suspects fréquemment observés en analyse de fraude, notamment les transferts incohérents avec les soldes. Le déséquilibre des classes est traité via `class_weight` dans les modèles linéaires et d'arbres. Les identifiants bruts `nameOrig` et `nameDest` sont exclus du modèle pour éviter un sur-apprentissage sur des entités uniques.

### 3.3 Expérimentation et comparaison des modèles

Conformément au cahier des charges, cinq familles de modèles ont été comparées : Régression Logistique, Random Forest, XGBoost, LightGBM et réseau de neurones. L'objectif n'était pas seulement de maximiser une métrique, mais de trouver un compromis entre performance, interprétabilité et robustesse opérationnelle.

![Comparaison recall et F1-score des modèles fraude](reports/figures/fraud_model_comparison.png)

La régression logistique atteint un recall élevé mais souffre d'une précision très faible sur l'échantillon de comparaison, ce qui générerait trop d'alertes inutiles. Le réseau de neurones, malgré une ROC-AUC élevée, présente un recall insuffisant pour un usage fraude en production. Random Forest, XGBoost et LightGBM obtiennent les meilleurs compromis sur l'échantillon comparé.

**La sélection du modèle de production est automatique.** Après comparaison (`scripts/compare_fraud_models.py`), le meilleur modèle est retenu selon F1-score, recall, précision puis ROC-AUC, puis réentraîné sur l'intégralité des transactions et sauvegardé dans `models/fraud_pipeline.joblib`. Le détail du choix est consigné dans `reports/fraud_model_selection.json`.

### 3.4 Performance du modèle retenu

Le modèle final affiche les performances suivantes sur le jeu d'évaluation retenu :

| Métrique | Valeur |
| --- | ---: |
| Accuracy | 99,9976 % |
| Précision | 99,55 % |
| Recall | 97,81 % |
| F1-score | 98,67 % |
| ROC-AUC | 0,9940 |

Sur le plan métier, le **recall** est la métrique la plus importante : il mesure la proportion de fraudes effectivement détectées. Avec un recall de 97,81 %, le système retrouve la quasi-totalité des cas historiques, soit environ **1 117 fraudes sur 1 142**. Les cas restants, bien que peu nombreux, correspondent souvent à des montants élevés et méritent une analyse manuelle des faux négatifs.

La précision à 100 % indique qu'aucune alerte générée sur l'échantillon d'évaluation n'était fausse. Ce résultat doit toutefois être interprété avec prudence : en production, la précision dépendra du seuil de décision, de l'évolution des comportements et du volume de transactions légitimes à fort montant.

### 3.5 Interprétabilité et analyse des erreurs

L'importance des variables confirme le rôle central de **orig_delta_error**, qui mesure l'incohérence entre le montant transféré et la variation du solde émetteur. Viennent ensuite `emptied_origin` (compte vidé après transaction) et `orig_balance_delta`. Ces signaux sont cohérents avec les schémas de fraude par transfert massif ou par extraction de fonds.

![Importance des variables (Random Forest)](reports/figures/fraud_feature_importance.png)

L'analyse SHAP sur un échantillon confirme cette lecture. SHAP permet de comprendre comment chaque variable pousse la prédiction vers la fraude ou vers la normalité, ce qui est utile pour expliquer une alerte à un analyste conformité.

![Importance SHAP moyenne](reports/figures/fraud_shap_importance.png)

L'examen des faux négatifs montre que les fraudes manquées concernent surtout des `CASH_OUT` et `TRANSFER` de montants élevés, parfois avec des soldes d'origine nuls ou des patterns de transfert atypiques. Les 25 plus grosses fraudes représentent environ **17,9 %** du montant frauduleux total : il est pertinent de combiner score de fraude et montant transaction pour prioriser les investigations.

---

## 4. Segmentation intelligente des clients

### 4.1 Exploration et préparation

Le second volet vise à identifier des profils clients homogènes pour adapter les actions marketing. Les variables couvrent l'âge, le revenu, les dépenses par catégorie de produits, les canaux d'achat, la récence et la réponse aux campagnes. Les valeurs manquantes de revenu ont été imputées, les variables catégorielles encodées et l'ensemble normalisé avant clustering.

Des variables agrégées ont été ajoutées : dépenses totales, nombre total d'achats, ancienneté client et intensité promotionnelle. Cet enrichissement facilite l'interprétation métier des segments obtenus.

### 4.2 Choix du nombre de clusters et comparaison des algorithmes

Le choix de **k = 4** résulte d'un compromis entre interprétabilité métier et indicateurs de qualité. La Silhouette maximale est obtenue pour k = 2, mais ce découpage est trop grossier pour guider des stratégies marketing différenciées. À k = 4, la structure reste exploitable (Silhouette **0,2315**, Davies-Bouldin **2,1541** avec le modèle retenu).

![Silhouette et Davies-Bouldin selon k](reports/figures/clustering_k_selection.png)

Parmi les algorithmes testés, **Gaussian Mixture** obtient la meilleure Silhouette (**0,2315**), tandis que DBSCAN produit de nombreux points bruit (**2 117 clients non assignés**) et une silhouette négative. **La sélection est automatique** : `scripts/compare_clustering_models.py` retient l'algorithme avec la meilleure silhouette (sans points bruit), puis sauvegarde le modèle dans `models/customer_clustering.joblib`. Le détail est consigné dans `reports/clustering_model_selection.json`.

![Comparaison des algorithmes de clustering](reports/figures/clustering_model_comparison.png)

### 4.3 Profils métier et recommandations

Les quatre segments ont été nommés selon leur comportement observable :

**Clients premium** (155 clients, 6,9 % de la base) affichent la dépense moyenne la plus élevée (**1 618**) et la meilleure réponse aux campagnes (**74,2 %**). Ils justifient des programmes de fidélisation haut de gamme plutôt que des promotions agressives.

**Clients dormants** (466 clients) se caractérisent par une récence élevée tout en conservant une dépense moyenne significative (**1 275**). Ce profil correspond à une clientèle à réactiver avec des offres limitées dans le temps.

**Digitaux et promotions** (583 clients) combinent les usages web les plus marqués et une sensibilité aux offres promotionnelles. Les campagnes digitales ciblées sont particulièrement adaptées à ce groupe.

**Clients économes** (1 036 clients) représentent le volume principal de la base, mais avec une dépense moyenne plus faible. Ils doivent être traités par des actions automatisées à faible coût.

![Cartographie des segments clients](reports/figures/customer_segments_scatter.png)

![Dépense moyenne par profil métier](reports/figures/customer_segment_spend.png)

### 4.4 Synthèse des actions marketing recommandées

| Profil | Action principale |
| --- | --- |
| Premium | Fidélisation, avantages exclusifs, service prioritaire |
| Dormants | Campagnes de réactivation limitées dans le temps |
| Digitaux / promotions | Offres web ciblées, codes promotionnels personnalisés |
| Économes | Automatisation marketing à faible coût |

---

## 5. Mise en production, scoring et MLOps

Le projet ne se limite pas à une analyse offline. Un pipeline reproductible permet de réentraîner le modèle fraude via `scripts/train_fraud.py` et la segmentation via `scripts/train_customer_clustering.py`.

L'**API FastAPI** (`api/main.py`) expose le scoring unitaire et le scoring CSV en volume, avec validation stricte du schéma en amont. Le **dashboard Streamlit** (`dashboard/app.py`) propose quatre sections calquées sur le rapport final :

1. Introduction et contexte ;
2. Détection de fraude bancaire ;
3. Scoring opérationnel ;
4. Segmentation client.

La documentation MLOps (`mlops/architecture.md`, model cards) décrit l'architecture cible : ingestion, validation, entraînement, évaluation, exposition des services, monitoring et réentraînement. Docker et docker-compose permettent de containeriser l'API pour un déploiement plus proche de la production.

---

## 6. Conclusion

Ce projet répond intégralement au cahier des charges `projet_machine_learning_m2CDSD.docx`. Sur la fraude, il combine analyse exploratoire approfondie, comparaison de cinq modèles, évaluation multi-métriques, interprétabilité par importance de variables et SHAP, ainsi qu'un scoring opérationnel. Sur la segmentation, il compare quatre algorithmes, justifie le nombre de clusters retenu et traduit les résultats en profils marketing actionnables.

Les principaux enseignements sont les suivants :

- concentrer les contrôles fraude sur `TRANSFER` et `CASH_OUT` ;
- ajuster le seuil de décision selon la charge analyste ;
- prioriser les grosses transactions dans la file de revue ;
- fidéliser les clients premium, réactiver les dormants et automatiser les campagnes pour les clients économes.

Les prochaines étapes consisteraient à brancher un flux transactionnel temps réel, historiser les prédictions, suivre la dérive des données et intégrer un outil de tracking d'expériences (MLflow ou DVC) pour passer du prototype à une plateforme MLOps complète.

---

*Pour régénérer ce rapport avec les graphiques intégrés :*

```bash
python scripts/generate_html_report.py
```

*Version HTML : `reports/rapport_final.html`*
