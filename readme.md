# Système d’aide au tri radiologique

## Description du projet

Ce projet a pour objectif de concevoir un système d’aide au tri radiologique à partir de radiographies thoraciques.  
Le système combine plusieurs briques de deep learning :

- une **classification supervisée** sur images médicales
- une **détection d’anomalies** avec un **autoencodeur convolutionnel**
- une **traçabilité expérimentale** avec **MLflow**
- un **démonstrateur applicatif** avec **Streamlit**

L’idée est de pouvoir charger une radiographie thoracique, obtenir des prédictions de pathologies, ainsi qu’un score d’anomalie indiquant si l’image semble atypique ou hors distribution.

---

## Objectifs

Le projet répond aux contraintes suivantes :

- utiliser au moins **3 architectures profondes** pour la classification
- utiliser un **AE ou VAE** pour la détection d’anomalies
- mettre en place un **tracking MLflow**
- proposer une **application testable**
- garder un pipeline **propre et reproductible**

---

## Modèles utilisés

### Classification supervisée
Les modèles utilisés pour la classification sont :

- **CNN entraîné depuis zéro**
- **CNN avec transfer learning**
- **Vision Transformer (ViT)**

Les modèles sauvegardés sont :

- `best_cnn_scratch.keras`
- `best_cnn_transfer.keras`
- `best_vit.keras`

### Détection d’anomalies
Pour la détection d’anomalies, nous utilisons un :

- **Convolutional Autoencoder**

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
│   └── best_conv_autoencoder.keras
│
├── notebooks/
│   ├── eda.ipynb
│   ├── preprocessing.ipynb
│   ├── supervised_models.ipynb
│   ├── detection_ae_vae.ipynb
│   └── tests.ipynb
│
├── requirements.txt
└── README.md
```

## Lancer l’application

Depuis la racine du projet, exécuter :

```bash
streamlit run app/app.py