import numpy as np
import matplotlib.pyplot as plt

def create_coordinate_grid(size=28):
    x = np.linspace(-np.pi, np.pi, size, dtype=np.float32)
    y = np.linspace(-np.pi, np.pi, size, dtype=np.float32)
    X, Y = np.meshgrid(x, y)
    Z = np.sqrt(X**2 + Y**2)
    return X.astype(np.float32), Y.astype(np.float32), Z.astype(np.float32)

X_GRID, Y_GRID, Z_GRID = create_coordinate_grid(28)

print("trigonometric_expressions imported")

# Funciones trigonométricas
def gp_sin(x):
    x = np.asarray(x, dtype=np.float32)
    return np.sin(x).astype(np.float32)

def gp_cos(x):
    x = np.asarray(x, dtype=np.float32)
    return np.cos(x).astype(np.float32)

# Función tangente con manejo de valores extremos
def gp_tan(x):
    x = np.asarray(x, dtype=np.float32)
    result = np.tan(x)
    result = np.nan_to_num(
        result,
        nan=0.0,
        posinf=1.0,
        neginf=-1.0
    )
    result = np.clip(result, -10.0, 10.0)
    return result.astype(np.float32)

FIELD = np.ndarray

# Funciones para FIELD
def field_add(a, b):
    return np.asarray(a, dtype=np.float32) + np.asarray(b, dtype=np.float32)

def field_subtract(a, b):
    return np.asarray(a, dtype=np.float32) - np.asarray(b, dtype=np.float32)

def field_multiply(a, b):
    return np.asarray(a, dtype=np.float32) * np.asarray(b, dtype=np.float32)

def field_scale(a, c):
    return np.asarray(a, dtype=np.float32) * float(c)

def field_add_constant(a, c):
    return np.asarray(a, dtype=np.float32) + float(c)

def field_divide(a, b):
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    return np.divide(a, b, out=np.zeros_like(a), where=np.abs(b) > 1e-6)

# Funciones trigonométricas para FIELD
def field_sin(a):
    return np.sin(a).astype(np.float32)

def field_cos(a):
    return np.cos(a).astype(np.float32)

def field_tan(a):
    a = np.asarray(a, dtype=np.float32)
    result = np.tan(a)
    result = np.nan_to_num(result, nan=0.0, posinf=10.0, neginf=-10.0)
    return np.clip(result, -10.0, 10.0).astype(np.float32)

# Convertir el campo matemático a una imagen normalizada
def field_to_image(field):
    field = np.asarray(field, dtype=np.float32)
    field = np.nan_to_num(field, nan=0.0, posinf=0.0, neginf=0.0)
    min_value = np.min(field)
    max_value = np.max(field)
    if max_value - min_value < 1e-8:
        return np.zeros_like(field, dtype=np.float32)
    image = (
        (field - min_value) /
        (max_value - min_value)
    ) * 255.0
    return image.astype(np.float32)

X, Y, Z = create_coordinate_grid(28)

fig, ax = plt.subplots(1, 3, figsize=(9, 3))

ax[0].imshow(X, cmap="gray")
ax[0].set_title("X")

ax[1].imshow(Y, cmap="gray")
ax[1].set_title("Y")

ax[2].imshow(Z, cmap="gray")
ax[2].set_title("Z")

plt.tight_layout()
plt.show()


# Mostrar las matrices
#print("X_GRID:")
#print(X_GRID)

#print("\nY_GRID:")
#print(Y_GRID)

#print("\nZ_GRID:")
#print(Z_GRID)

# fig = plt.figure(figsize=(8, 6))
# ax = fig.add_subplot(111, projection="3d")

# ax.plot_surface(
#     X_GRID,
#     Y_GRID,
#     Z_GRID,
#     cmap="viridis",
#     edgecolor="none"
# )

# ax.set_xlabel("X")
# ax.set_ylabel("Y")
# ax.set_zlabel("Z")
# ax.set_title(r"$Z = \sqrt{X^2 + Y^2}$")

#plt.show()

# fig, axes = plt.subplots(1, 3, figsize=(12, 4))

# axes[0].imshow(X_GRID, cmap="gray")
# axes[0].set_title("X")
# axes[0].axis("off")

# axes[1].imshow(Y_GRID, cmap="gray")
# axes[1].set_title("Y")
# axes[1].axis("off")

# axes[2].imshow(Z_GRID, cmap="gray")
# axes[2].set_title(r"$Z = \sqrt{X^2 + Y^2}$")
# axes[2].axis("off")

# plt.tight_layout()
#plt.show()

print("Todo bien hasta aquí")