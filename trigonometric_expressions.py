import numpy as np

def create_coordinate_grid(size=28):
    x = np.linspace(-np.pi, np.pi, size, dtype=np.float32)
    y = np.linspace(-np.pi, np.pi, size, dtype=np.float32)

    X, Y = np.meshgrid(x, y)

    Z = np.sqrt(X**2 + Y**2)

    return X.astype(np.float32), Y.astype(np.float32), Z.astype(np.float32)


X_GRID, Y_GRID, Z_GRID = create_coordinate_grid(28)

# Funciones trigonométricas
def gp_sin(x):
    return np.sin(x).astype(np.float32)

def gp_cos(x):
    return np.cos(x).astype(np.float32)

def gp_tan(x):
    x = np.clip(x, -np.pi / 2 + 1e-3, np.pi / 2 - 1e-3)
    return np.tan(x).astype(np.float32)