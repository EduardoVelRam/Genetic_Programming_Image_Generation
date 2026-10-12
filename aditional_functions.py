import numpy as np

def preprocess_for_knn(image):
    image = np.asarray(image, dtype=np.float32)

    if image.shape != (28, 28):
        raise ValueError(
            f"Se esperaba una imagen de 28x28, "
            f"pero se recibió {image.shape}."
        )

    image = np.nan_to_num(
        image,
        nan=0.0,
        posinf=255.0,
        neginf=0.0
    )

    image = np.clip(image, 0.0, 255.0)

    image = image / 255.0

    return image.reshape(1, -1)

def classifier_probability(image, classifier, target_class):
    X = preprocess_for_knn(image)

    probabilities = classifier.predict_proba(X)[0]

    class_index = np.where(
        classifier.classes_ == target_class
    )[0]

    if len(class_index) == 0:
        raise ValueError(
            f"La clase {target_class} no está "
            "presente en el clasificador."
        )

    return float(probabilities[class_index[0]])


def prepare_image_for_knn(image):
    image = np.asarray(image, dtype=np.float32)

    if image.shape != (28, 28):
        raise ValueError(
            f"Forma de imagen incorrecta: {image.shape}"
        )

    image = np.nan_to_num(
        image,
        nan=0.0,
        posinf=255.0,
        neginf=0.0
    )

    image = np.clip(image, 0.0, 255.0)

    return image