import numpy as np


def predict_with_model(model, image_array, class_names=None):
    preds = model.predict(image_array, verbose=0)[0]

    if class_names is None:
        class_names = [f"class_{i}" for i in range(len(preds))]

    results = []
    for i, score in enumerate(preds):
        results.append({
            "class": class_names[i],
            "score": float(score)
        })

    results = sorted(results, key=lambda x: x["score"], reverse=True)
    return results


def get_top_k_predictions(model, image_array, class_names=None, k=5):
    results = predict_with_model(model, image_array, class_names)
    return results[:k]


def get_predicted_labels(model, image_array, class_names=None, threshold=0.5):
    preds = model.predict(image_array, verbose=0)[0]

    if class_names is None:
        class_names = [f"class_{i}" for i in range(len(preds))]

    labels = []
    for i, score in enumerate(preds):
        if score >= threshold:
            labels.append({
                "class": class_names[i],
                "score": float(score)
            })

    return labels