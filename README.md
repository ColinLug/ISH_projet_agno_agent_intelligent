# ISH_projet_agno_agent_intelligent

## Contexte

Dans le cadre du cours IA avancée de l'Université de Lausanne, ce projet à pour objectif d'évaluer dans quelle mesure les grands modèles de langage (LLMs) sont capables de rendre un texte donné plus complexe tout en en conservant le sens original. Plus précisément, nous voulons construire un système multi-agents qui teste si les LLMs peuvent produire un texte cohérent à des niveaux de complexité spécifiques.

## Fonctionnalité

- Architecture agentique et pipeline Agno
- Test de complexité
- Test manuelle préalable
- Analyse des résultats

## Structure des données

- data_sampled : dossier contenant les données sources
    - OSE_adv_ele.csv (189 lignes)
    - OSE_adv_int-csv (189 lignes)
    - swipe.csv (1233 lignes)
    - vikidia.csv (1233 lignes)
- data_cleaned : dossier utilisée pour la préparation et la curation des données.
    - test_preparation_data : notebook pour la préparation des données testes.


## Implémentation

L'implémentation

## Installation 

Le projet a été effectué avec le outils dans les versions suivantes :

agno==2.3.8
nltk==3.9.2
numpy==2.3.5
pandas==2.3.3
pydantic==2.12.5
spacy==3.8.11
stanza==1.11.0
textcomplexity==0.11.0
tqdm==4.66.4