import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

DEBUG_DIR = Path("debug_fitness_zero")
DEBUG_DIR.mkdir(parents=True, exist_ok=True)

# Asegurarse de que la imagen esté en el rango [0, 255] y sea de tipo float32 antes de pasarla al clasificador KNN
def prepare_image_for_knn(image):
    image = np.asarray(image, dtype=np.float32)

    if image.shape != (28, 28):
        raise ValueError(
            f"Forma de imagen incorrecta: {image.shape}"
        )
    image = np.nan_to_num(image, nan=0.0, posinf=255.0, neginf=0.0) 
    image = np.clip(image, 0.0, 255.0)
    return image

def preprocess_for_knn(image):
    image = prepare_image_for_knn(image)
    image = image / 255.0
    # return image.reshape(1, -1)  # anntes
    return image.reshape(1, -1) / 255.0

def classifier_probability(image, classifier, target_class):
    X = preprocess_for_knn(image)

    probabilities = classifier.predict_proba(X)[0] # para knn
    # probabilities = classifier.predict_proba(X)

    class_index = np.where(
        classifier.classes_ == target_class
    )[0]

    if len(class_index) == 0:
        raise ValueError(
            f"La clase {target_class} no está "
            "presente en el clasificador."
        )

    return float(probabilities[class_index[0]]) # antes
    # return float(probabilities[0, class_index[0]])

# def classifier_probability(
#     image,
#     classifier,
#     target_class
# ):
#     image = np.asarray(image, dtype=np.float32)

#     if image.shape != (28, 28):
#         raise ValueError(
#             f"Se esperaba una imagen de 28x28, "
#             f"pero se recibió {image.shape}"
#         )

#     image = np.nan_to_num(
#         image,
#         nan=0.0,
#         posinf=255.0,
#         neginf=0.0
#     )

#     # Mantener los píxeles en [0, 255].
#     image = np.clip(image, 0.0, 255.0)

#     # Aplicar el mismo preprocesamiento del entrenamiento.
#     image = image.reshape(1, -1) / 255.0

#     probabilities = classifier.predict_proba(image)

#     class_indices = np.where(
#         classifier.classes_ == target_class
#     )[0]

#     if len(class_indices) == 0:
#         raise ValueError(
#             f"La clase {target_class} no existe "
#             "en el clasificador."
#         )

#     return float(probabilities[0, class_indices[0]])


def inspect_individual(individual, image, probability, target_class):
    image = np.asarray(image, dtype=np.float32)

    X = preprocess_for_knn(image)
    predicted_class = mlp1.predict(X)[0]

    print("\n=== INDIVIDUO CON FITNESS CERO ===")
    #print("Árbol:", individual)
    #print("Clase objetivo:", target_class)
    #print("Clase predicha:", predicted_class)
    #print("Probabilidad objetivo:", probability)
    #print("Mínimo:", image.min())
    print("Máximo:", image.max())
    print("Media:", image.mean())
    #print("Desviación estándar:", image.std())

    probabilities = mlp1.predict_proba(X)[0]
    print("Probabilidades por clase:", probabilities)

    plt.figure(figsize=(4, 4))
    plt.imshow(image, cmap="gray", vmin=0, vmax=255)
    plt.title(
        f"Objetivo: {target_class} | "
        f"Predicción: {predicted_class} | "
        f"P: {probability:.3f}"
    )
    plt.axis("off")

    filename = DEBUG_DIR / (
        f"fitness_zero_{abs(hash(str(individual)))}.png"
    )

    plt.savefig(filename, bbox_inches="tight", dpi=150)
    plt.close()

    print("Imagen guardada en:", filename)
