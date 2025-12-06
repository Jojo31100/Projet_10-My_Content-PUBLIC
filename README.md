# Projet_10-My_Content-PUBLIC
Réalisez une application de recommandation de contenu

---

## Description

Ce projet est une application de recommandation d'articles, basée sur les clics des utilisateurs.  
Elle combine trois types de systèmes de recommandation pour proposer les articles les plus pertinents :

1. **Popularity-Based Filtering** : recommande les articles les plus populaires globalement.  
2. **Content-Based Filtering** : recommande des articles similaires à ceux déjà consultés par l'utilisateur.  
3. **Collaborative Filtering (ALS)** : recommande des articles en se basant sur les comportements d'autres utilisateurs similaires.

Le projet est déployé sous forme d'**Azure Function** pour un accès rapide via API REST.

---

## Fonctionnalités

- Chargement automatique des données depuis Azure Blob Storage :
  - Clics utilisateurs (CSV)
  - Metadata des articles (CSV)
  - Embeddings des articles (pickle)
- Sélection automatique du type de recommandation en fonction du profil utilisateur
- Calcul des recommandations avec scores associés
- Monitoring de l’utilisation mémoire et du temps d’exécution
- Journalisation claire avec `logging` pour débogage et suivi

---

## Installation

### Prérequis

- Python 3.10+
- Bibliothèques Python :
  ```bash
  pip install pandas numpy scipy scikit-learn implicit azure-storage-blob psutil

---

## Exemple d'utilisation

from recommander_main import get_recommendations

resultat = get_recommendations(user_id=12345, nbReco=5)

print(resultat)

---

## Exemple de réponse JSON

{

    "user_id": 12345,

    "modele_utilise": "content",

    "statistiques_utilisateur": {

        "clics": 10,

        "articles": 7,

        "categories": 2

    },

    "recommandations": [

        {"article_id": 101, "score": 0.92},

        {"article_id": 45, "score": 0.88},

        {"article_id": 28, "score": 0.85},

        {"article_id": 451, "score": 0.63},

        {"article_id": 1455 "score": 0.37}

    ],

    "memoire_utilisee_GiB": 0.15,

    "temps_execution": 0.23,

    "version": "1.3"

}

---

## Structure du projet

Projet_10-My_Content-PUBLIC/

│

├─ recommander/

│   └─ recommander_main.py  #Azure Function principale

│

├─ data/                    #Conteneur Azure Blob (non pushé sur GitHub)

│   ├─ articles_metadata.csv

│   ├─ articles_embeddings.pickle

│   └─ clicks

│      ├─ clicks_hour_000.csv

│      └─ ...

│

└─ README.md
