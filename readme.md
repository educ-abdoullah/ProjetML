# Système d’aide au tri radiologique

## Description du projet

Ce projet a pour objectif de concevoir un système d’aide au tri radiologique à partir de radiographies thoraciques.  
Le système combine plusieurs briques de deep learning afin d’assister l’identification de pathologies et le repérage de cas atypiques.

Le pipeline comprend :

- une **classification supervisée** sur radiographies thoraciques ;
- une **détection d’anomalies** avec un **autoencodeur convolutionnel** ;
- une **composante multimodale** fondée sur la fusion **image + métadonnées** ;
- un **suivi expérimental** avec **MLflow** ;
- un **démonstrateur applicatif** avec **Streamlit**.

L’objectif est de pouvoir charger une radiographie thoracique, obtenir des prédictions de pathologies, afficher un score d’anomalie, et comparer une prédiction **image seule** à une prédiction **multimodale**.

---

## Objectifs

Le projet répond aux contraintes suivantes :

- utiliser au moins **3 architectures profondes** pour la classification ;
- mettre en place une **détection d’anomalies** par **AE ou VAE** ;
- comparer plusieurs approches de modélisation ;
- assurer la **traçabilité expérimentale** avec **MLflow** ;
- proposer une **application testable** ;
- garder un pipeline **propre, cohérent et reproductible**.

---

## Jeux de données utilisés

### 1. ChestMNIST / MedMNIST
Ce dataset constitue la base principale de la partie supervisée et de la détection d’anomalies.

Il est utilisé pour :
- l’entraînement des modèles de classification ;
- l’analyse exploratoire des données ;
- la préparation des images ;
- l’autoencodeur convolutionnel.

La résolution retenue dans le projet est **128 × 128**, afin d’obtenir un compromis entre qualité visuelle et coût de calcul.

### 2. NIH Chest X-rays
Le dataset NIH est utilisé pour la composante multimodale.

Dans la version actuellement implémentée, la multimodalité repose sur :
- les **images thoraciques** ;
- les **métadonnées structurées** associées à l’examen :
  - âge,
  - sexe,
  - position de vue.

Cette partie constitue une preuve de concept de fusion multimodale **image + métadonnées**.

---

## Modèles utilisés

## Classification supervisée
Les modèles utilisés pour la classification sont :

- **CNN entraîné depuis zéro**
- **CNN avec transfer learning**
- **Vision Transformer (ViT)**

Les modèles sauvegardés sont :

- `best_cnn_scratch.keras`
- `best_cnn_transfer.keras`
- `best_vit.keras`

## Modèle image pour la partie NIH
Pour la composante multimodale, un modèle image spécifique NIH a également été sauvegardé :

- `nih_image_model.keras`

## Détection d’anomalies
Pour la détection d’anomalies, le projet utilise :

- un **Convolutional Autoencoder**

Modèle sauvegardé :

- `best_conv_autoencoder.keras`

Le score d’anomalie est calculé à partir de l’erreur de reconstruction entre l’image d’entrée et l’image reconstruite.

---

## Structure du projet

```text
projet_deeplearning/
│
├── app/
│   └── app.py
│
├── src/
│   ├── __init__.py
│   ├── custom_layers.py
│   ├── load_models.py
│   ├── preprocess.py
│   ├── predict.py
│   ├── anomaly.py
│   └── config.py
│
├── models/
│   ├── best_cnn_scratch.keras
│   ├── best_cnn_transfer.keras
│   ├── best_vit.keras
│   ├── nih_image_model.keras
│   └── best_conv_autoencoder.keras
│
├── notebooks/
│   ├── eda.ipynb
│   ├── preprocessing.ipynb
│   ├── supervised_models.ipynb
│   ├── detection_ae_vae.ipynb
│   ├── multimodal.ipynb
│   ├── tests.ipynb
│   └── mlflow.db
│
├── requirements.txt
└── README.md