from tensorflow import keras
from src.custom_layers import Patches, PatchEncoder


def load_cnn_scratch(model_path="models/best_cnn_scratch.keras"):
    return keras.models.load_model(model_path, compile=False)


def load_cnn_transfer(model_path="models/best_cnn_transfer.keras"):
    return keras.models.load_model(model_path, compile=False)


def load_vit(model_path="models/best_vit.keras"):
    return keras.models.load_model(
        model_path,
        custom_objects={
            "Patches": Patches,
            "PatchEncoder": PatchEncoder
        },
        compile=False
    )


def load_autoencoder(model_path="models/best_conv_autoencoder.keras"):
    return keras.models.load_model(model_path, compile=False)


def load_nih_image_model(model_path="models/nih_image_model.keras"):
    return keras.models.load_model(model_path, compile=False)


def load_all_models():
    models = {
        "cnn_scratch": load_cnn_scratch(),
        "cnn_transfer": load_cnn_transfer(),
        "vit": load_vit(),
        "autoencoder": load_autoencoder(),
        "nih_image_model": load_nih_image_model()
    }
    return models