import numpy as np
from PIL import Image


def preprocess_image(image, target_size=(128, 128), grayscale=True):
    if isinstance(image, str):
        image = Image.open(image)

    if grayscale:
        image = image.convert("L")
    else:
        image = image.convert("RGB")

    image = image.resize(target_size)
    img_array = np.array(image).astype("float32") / 255.0

    if grayscale:
        img_array = np.expand_dims(img_array, axis=-1)

    img_array = np.expand_dims(img_array, axis=0)
    return img_array


def postprocess_reconstruction(reconstruction):
    reconstruction = reconstruction[0]
    reconstruction = np.squeeze(reconstruction)
    reconstruction = np.clip(reconstruction, 0, 1)
    return reconstruction