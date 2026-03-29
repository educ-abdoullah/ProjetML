import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

import streamlit as st
from PIL import Image
import numpy as np
import pandas as pd

from src.load_models import load_all_models
from src.preprocess import preprocess_image, postprocess_reconstruction
from src.predict import get_top_k_predictions, get_predicted_labels
from src.anomaly import compute_anomaly_score, classify_anomaly
from src.config import CLASS_NAMES, IMAGE_SIZE, GRAYSCALE, ANOMALY_THRESHOLD


st.set_page_config(page_title="Tri radiologique", layout="wide")
st.title("Système d’aide au tri radiologique")

st.markdown(
    "Charge une radiographie pour obtenir une prédiction image seule, "
    "une comparaison multimodale (image + métadonnées) et un score d’anomalie."
)

@st.cache_resource
def get_models():
    return load_all_models()

models = get_models()


def make_metadata_input(age, gender, view_position):
    row = pd.DataFrame([{
        "Patient Age": age,
        "Patient Gender": 0 if gender == "M" else 1,
        "View Position": view_position
    }])

    row = pd.get_dummies(row, columns=["View Position"], dummy_na=True)

    expected_columns = [
        "Patient Age",
        "Patient Gender",
        "View Position_AP",
        "View Position_PA",
        "View Position_nan"
    ]

    row = row.reindex(columns=expected_columns, fill_value=0)
    return row


def get_top_k_from_probs(probs, class_names, k=5):
    probs = np.array(probs).flatten()
    indices = np.argsort(probs)[::-1][:k]

    results = []
    for idx in indices:
        results.append({
            "class": class_names[idx],
            "score": float(probs[idx])
        })
    return results


def get_labels_from_probs(probs, class_names, threshold=0.2):
    probs = np.array(probs).flatten()

    labels = []
    for i, score in enumerate(probs):
        if score >= threshold:
            labels.append({
                "class": class_names[i],
                "score": float(score)
            })
    return labels


uploaded_file = st.file_uploader(
    "Choisir une image",
    type=["png", "jpg", "jpeg"]
)

model_choice = st.selectbox(
    "Choisir le modèle de classification image seule",
    ["cnn_scratch", "cnn_transfer", "vit"]
)

threshold = st.slider(
    "Seuil de prédiction des labels",
    min_value=0.0,
    max_value=1.0,
    value=0.2,
    step=0.05
)

st.subheader("Métadonnées pour la comparaison multimodale")
col_meta1, col_meta2, col_meta3 = st.columns(3)

with col_meta1:
    age = st.number_input("Âge", min_value=0, max_value=120, value=50)

with col_meta2:
    gender = st.selectbox("Sexe", ["M", "F"])

with col_meta3:
    view_position = st.selectbox("Position de vue", ["PA", "AP"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Image chargée")
        st.image(image, caption="Radiographie", use_container_width=True)

    image_array = preprocess_image(
        image=image,
        target_size=IMAGE_SIZE,
        grayscale=GRAYSCALE
    )

    classifier = models[model_choice]
    autoencoder = models["autoencoder"]
    nih_image_model = models["nih_image_model"]

    if st.button("Lancer l'analyse"):
        # =========================
        # IMAGE SEULE (ChestMNIST)
        # =========================
        top_predictions = get_top_k_predictions(
            model=classifier,
            image_array=image_array,
            class_names=CLASS_NAMES,
            k=5
        )

        predicted_labels = get_predicted_labels(
            model=classifier,
            image_array=image_array,
            class_names=CLASS_NAMES,
            threshold=threshold
        )

        # =========================
        # MULTIMODAL (NIH image + métadonnées)
        # =========================
        nih_image_probs = nih_image_model.predict(image_array, verbose=0)[0]

        metadata_input = make_metadata_input(age, gender, view_position)

        # Placeholder simple pour la modalité métadonnées :
        # ici on crée un vecteur faible et neutre pour garder la structure.
        # Remplace plus tard par meta_model.predict_proba(metadata_input)[0]
        meta_probs = np.full(len(CLASS_NAMES), 0.05, dtype=float)

        alpha = 0.7
        beta = 0.3
        multimodal_probs = alpha * nih_image_probs + beta * meta_probs

        multimodal_top_predictions = get_top_k_from_probs(
            probs=multimodal_probs,
            class_names=CLASS_NAMES,
            k=5
        )

        multimodal_labels = get_labels_from_probs(
            probs=multimodal_probs,
            class_names=CLASS_NAMES,
            threshold=threshold
        )

        # =========================
        # ANOMALIE
        # =========================
        anomaly_score, reconstruction = compute_anomaly_score(
            autoencoder=autoencoder,
            image_array=image_array
        )

        anomaly_text = classify_anomaly(
            score=anomaly_score,
            threshold=ANOMALY_THRESHOLD
        )

        with col2:
            st.subheader("Résultats image seule")
            st.write(f"Modèle utilisé : {model_choice}")

            st.write("Top 5 des prédictions :")
            for pred in top_predictions:
                st.write(f"- {pred['class']} : {pred['score']:.4f}")

            st.subheader("Labels retenus")
            if predicted_labels:
                for label in predicted_labels:
                    st.write(f"- {label['class']} : {label['score']:.4f}")
            else:
                st.write("Aucun label au-dessus du seuil choisi.")

            st.subheader("Détection d’anomalie")
            st.write(f"Score d’anomalie : {anomaly_score:.6f}")
            st.write(anomaly_text)

        st.subheader("Comparaison image seule vs multimodal")

        col3, col4 = st.columns(2)

        with col3:
            st.markdown("### Image seule")
            for pred in top_predictions:
                st.write(f"- {pred['class']} : {pred['score']:.4f}")

            st.markdown("**Labels retenus**")
            if predicted_labels:
                for label in predicted_labels:
                    st.write(f"- {label['class']} : {label['score']:.4f}")
            else:
                st.write("Aucun label au-dessus du seuil choisi.")

        with col4:
            st.markdown("### Multimodal (image + métadonnées)")
            st.write(f"Fusion utilisée : {alpha:.1f} image / {beta:.1f} métadonnées")

            for pred in multimodal_top_predictions:
                st.write(f"- {pred['class']} : {pred['score']:.4f}")

            st.markdown("**Labels retenus**")
            if multimodal_labels:
                for label in multimodal_labels:
                    st.write(f"- {label['class']} : {label['score']:.4f}")
            else:
                st.write("Aucun label au-dessus du seuil choisi.")

        st.subheader("Reconstruction par l’autoencodeur")

        reconstruction_img = postprocess_reconstruction(reconstruction)

        col5, col6 = st.columns(2)

        with col5:
            st.image(
                image,
                caption="Image originale",
                use_container_width=True
            )

        with col6:
            if GRAYSCALE:
                st.image(
                    reconstruction_img,
                    caption="Image reconstruite",
                    clamp=True,
                    use_container_width=True
                )
            else:
                st.image(
                    reconstruction_img,
                    caption="Image reconstruite",
                    use_container_width=True
                )

        st.info(
            "La comparaison multimodale est affichée sous la forme image + métadonnées. "
            "Quand le vrai modèle métadonnées sera branché, il suffira de remplacer "
            "le vecteur meta_probs simulé par la sortie du modèle tabulaire."
        )