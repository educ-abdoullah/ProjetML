import numpy as np
from src.preprocess import postprocess_reconstruction


def compute_anomaly_score(autoencoder, image_array):
    reconstruction = autoencoder.predict(image_array, verbose=0)
    score = np.mean((image_array - reconstruction) ** 2)
    return float(score), reconstruction


def get_reconstruction_image(autoencoder, image_array):
    reconstruction = autoencoder.predict(image_array, verbose=0)
    return postprocess_reconstruction(reconstruction)


def classify_anomaly(score, threshold=0.01):
    if score >= threshold:
        return "Image potentiellement atypique"
    return "Image plutôt normale"