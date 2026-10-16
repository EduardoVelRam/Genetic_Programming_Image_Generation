from pathlib import Path

import numpy as np
import idx2numpy
import kagglehub
import joblib
from import_data import load_mnist, mnist_path
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split


# ============================================================
# CONFIGURACIÓN
# ============================================================

RANDOM_SEED = 42

MODEL_PATH = Path(__file__).parent / "mlp_mnist.joblib"


# ============================================================
# CARGA DE MNIST
# ============================================================

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
# PREPROCESAMIENTO
# ============================================================

# MNIST: (n_imágenes, 28, 28) -> (n_imágenes, 784)

X_train = X_train.reshape(len(X_train), -1)
X_test = X_test.reshape(len(X_test), -1)

# Normalizar píxeles de [0, 255] a [0, 1].

X_train = X_train.astype(np.float32) / 255.0
X_test = X_test.astype(np.float32) / 255.0

y_train = y_train.astype(np.int64)
y_test = y_test.astype(np.int64)


# ============================================================
# ENTRENAMIENTO
# ============================================================

mlp = MLPClassifier(
    hidden_layer_sizes=(128, 64),
    activation="relu",
    solver="adam",
    alpha=1e-4,
    batch_size=128,
    learning_rate_init=1e-3,
    max_iter=100,
    early_stopping=True,
    validation_fraction=0.1,
    n_iter_no_change=10,
    random_state=RANDOM_SEED,
    verbose=True
)

print("Entrenando MLPClassifier...")

mlp.fit(X_train, y_train)


# ============================================================
# EVALUACIÓN
# ============================================================

y_pred = mlp.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print(f"\nExactitud en test: {accuracy:.4f}")
print(f"Iteraciones realizadas: {mlp.n_iter_}")

print("\nReporte de clasificación:")
print(classification_report(y_test, y_pred))


# ============================================================
# GUARDAR MODELO Y METADATOS
# ============================================================

model_data = {
    "model": mlp,
    "input_shape": (28, 28),
    "normalization": "divide_by_255",
    "classes": mlp.classes_,
    "random_seed": RANDOM_SEED,
    "test_accuracy": accuracy
}

joblib.dump(model_data, MODEL_PATH)

print(f"\nModelo guardado en: {MODEL_PATH}")