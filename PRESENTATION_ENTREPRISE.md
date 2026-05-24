# Presentation entreprise

## 1. Message executif

Ce projet met en place une solution complete de valorisation des donnees pour une entreprise financiere et marketing:

- detection automatique des transactions frauduleuses;
- segmentation intelligente des clients;
- tableau de bord de pilotage;
- service de scoring;
- premiere architecture MLOps pour industrialiser le systeme.

## 2. Resultats principaux

Detection de fraude:

- transactions analysees: 1 048 575;
- fraudes observees: 1 142;
- taux de fraude: 0,109%;
- precision modele: 1,000;
- recall modele: 0,978;
- F1-score: 0,989;
- ROC-AUC: 0,998.

Segmentation client:

- clients segmentes: 2 240;
- nombre de segments retenu: 4;
- profils metiers: clients premium, clients dormants, clients digitaux/promotions, clients economes.

## 3. Demonstration conseillee

1. Ouvrir le terminal et lancer:

```bash
python scripts/show_results.py
```

2. Ouvrir le dashboard:

```bash
python -m streamlit run dashboard/app.py
```

3. Montrer la page `01 - Synthese dirigeant`.

4. Montrer la page `02 - Analyse du risque fraude`.

5. Montrer la page `04 - Segmentation clients`.

6. Aller dans `03 - Scoring transaction et CSV`.

7. Saisir une transaction manuellement et lire la decision, la probabilite et l'action recommandee.

8. Importer un CSV normalise, verifier le controle du schema, scorer le fichier puis exporter le fichier score.

## 4. Exploitation en vraie vie

Dans une entreprise, ce projet ne serait pas seulement un notebook. Il deviendrait un service integre au systeme d'information.

Pour la fraude:

- chaque nouvelle transaction est envoyee au service de scoring;
- le service retourne une probabilite de fraude;
- les transactions a fort risque sont bloquees ou envoyees en revue analyste;
- les analystes confirment ou rejettent les alertes;
- ces retours servent a reentrainer le modele.
- les fichiers CSV de gros volume peuvent etre importes, valides, scores, puis renvoyes avec un niveau de risque et une action recommandee.

Pour le marketing:

- les clients sont segmentes regulierement;
- chaque segment recoit une strategie differente;
- les clients premium recoivent des avantages de fidelisation;
- les clients dormants recoivent des campagnes de reactivation;
- les clients sensibles aux promotions recoivent des offres ciblees;
- les performances des campagnes sont suivies par segment.

## 5. Architecture cible

```text
Sources donnees
    -> Ingestion et validation
    -> Feature engineering
    -> Entrainement modele
    -> Evaluation
    -> Model registry
    -> Service de scoring
    -> Dashboard metier
    -> Monitoring
    -> Reentrainement
```

## 6. Points de vigilance

- Ne pas juger le modele fraude uniquement avec l'accuracy, car les classes sont tres desequilibrees.
- Prioriser le recall pour reduire les fraudes manquees.
- Ajuster le seuil de decision selon le cout metier.
- Rejeter les fichiers non conformes avant scoring pour eviter les predictions invalides.
- Surveiller la derive des donnees.
- Valider les segments avec les equipes marketing.
- Documenter les modeles avant tout deploiement.

## 7. Phrase de conclusion

La solution proposee transforme deux jeux de donnees bruts en systeme decisionnel exploitable: un moteur de detection de fraude, une segmentation client actionnable et une base MLOps pour passer du prototype a la production.

## 8. Livrables remis

- Rapport final: `RAPPORT_FINAL.md`
- Rapport HTML: `reports/rapport_final.html`
- Checklist de conformite: `CHECKLIST_CONFORMITE.md`
- Notebooks commentes: `notebooks/`
- Dashboard Streamlit: `dashboard/app.py`
- Presentation entreprise: `PRESENTATION_ENTREPRISE.md`
- Depot GitHub propre et deployable: `README.md`, `requirements.txt`, `src/`, `scripts/`, `api/`, `dashboard/`, `mlops/`, `models/`, `reports/`
- Main file path Streamlit Cloud: `dashboard/app.py`
