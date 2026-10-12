from pathlib import Path

import idx2numpy
import numpy as np
import joblib
from import_data import load_mnist, mnist_path
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score


# CONFIGURACIÓN
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "knn_mnist.joblib"
K_NEIGHBORS = 5

# CARGAR MNIST
X_train, y_train, X_test, y_test = load_mnist(mnist_path)

# PREPROCESAMIENTO
X_train = X_train.astype(np.float32).reshape(
    len(X_train), -1
) / 255.0

X_test = X_test.astype(np.float32).reshape(
    len(X_test), -1
) / 255.0


# ============================================================
# ENTRENAMIENTO
# ============================================================

knn = KNeighborsClassifier(
    n_neighbors=K_NEIGHBORS,
    weights="distance",
    metric="euclidean",
    n_jobs=-1
)

print("Entrenando KNN...")

knn.fit(X_train, y_train)


# ============================================================
# EVALUACIÓN
# ============================================================

y_pred = knn.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print(f"Accuracy en test: {accuracy:.4f}")


# ============================================================
# GUARDAR MODELO
# ============================================================

joblib.dump(knn, MODEL_PATH)

print(f"Modelo guardado en: {MODEL_PATH}")