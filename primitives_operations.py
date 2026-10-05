from import_data import load_mnist, mnist_path
import numpy as np
import matplotlib.pyplot as plt
import skimage
from scipy.ndimage import convolve


X_train, y_train, X_test, y_test = load_mnist(mnist_path)

# print(X_train.shape)
# print(y_train.shape)
# print(X_test.shape)
# print(y_test.shape)

# Set objective
target_class = 1

target_images = X_train[y_train == target_class]

print(target_images.shape)

# Visualize some images of the target class
fig, axes = plt.subplots(1, 5, figsize=(10, 2))

for i, ax in enumerate(axes):
    ax.imshow(target_images[i], cmap="gray")
    ax.axis("off")

plt.tight_layout()
plt.show()

# Generate a random image and visualize it alongside the target image
random_image2 = np.random.randint(0, 256, size=(28, 28), dtype=np.uint8)
# Imagen aleatoria inicial
random_image = np.random.uniform(0, 255, size=(28, 28)).astype(np.float32)

# target_image = target_images[0]
# Seleccionar una imagen objetivo
target_image = target_images[0].astype(np.float32)

fig, axes = plt.subplots(1, 3, figsize=(5, 2.5))

axes[0].imshow(random_image, cmap="gray")
axes[0].set_title("Entrada aleatoria")
axes[0].axis("off")

axes[1].imshow(random_image2, cmap="gray")
axes[1].set_title("Entrada aleatoria 2")
axes[1].axis("off")

axes[2].imshow(target_image, cmap="gray")
axes[2].set_title("Objetivo")
axes[2].axis("off")

plt.tight_layout()
plt.show()

print("Target:", target_image.shape, target_image.min(), target_image.max())
print("Random:", random_image.shape, random_image.min(), random_image.max())

# Normalize the images to the range [0, 255]
def normalize_image2(image):
    image = np.asarray(image, dtype=np.float32)

    min_val = image.min()
    max_val = image.max()

    if max_val == min_val:
        return np.zeros_like(image, dtype=np.float32)

    image = (image - min_val) / (max_val - min_val)
    image = image * 255.0

    return image

def normalize_image(image): # Clip image
    return np.clip(
        image,
        0.0,
        255.0
    ).astype(np.float32)

# Morfological operations
from skimage.morphology import (
    erosion,
    dilation,
    opening,
    closing,
    disk
)

selem = disk(1)

# Primitives
def gp_erosion(image):
    return normalize_image(
        erosion(image, footprint=selem)
    )


def gp_dilation(image):
    return normalize_image(
        dilation(image, footprint=selem)
    )


def gp_opening(image):
    return normalize_image(
        opening(image, footprint=selem)
    )


def gp_closing(image):
    return normalize_image(
        closing(image, footprint=selem)
    )

# Convolucion 3x3
average_kernel = np.ones((3, 3), dtype=np.float32) / 9.0

def gp_convolution(image):
    result = convolve(
        image,
        average_kernel,
        mode="reflect"
    )
    return normalize_image(result)

# Operaciones aritméticas
def gp_add(image_a, image_b):
    return normalize_image(image_a + image_b)


def gp_subtract(image_a, image_b):
    return normalize_image(image_a - image_b)


# def gp_multiply(image_a, image_b):
#     return normalize_image(image_a * image_b)

def gp_multiply(image_a, image_b):
    return normalize_image((image_a * image_b) / 255.0)

# Funicón fitness iniciaal
def mse(image_a, image_b):
    image_a = np.asarray(image_a, dtype=np.float32)
    image_b = np.asarray(image_b, dtype=np.float32)

    return np.mean((image_a - image_b) ** 2)

# Probar primitivas
images = {
    "Original": random_image,
    "Erosion": gp_erosion(random_image),
    "Dilation": gp_dilation(random_image),
    "Opening": gp_opening(random_image),
    "Closing": gp_closing(random_image),
    "Convolution": gp_convolution(random_image),
}

fig, axes = plt.subplots(2, 3, figsize=(8, 6))

for ax, (name, image) in zip(axes.ravel(), images.items()):
    ax.imshow(image, cmap="gray", vmin=0, vmax=255)
    ax.set_title(name)
    ax.axis("off")

plt.tight_layout()
plt.show()

for name, image in images.items():
    print(
        f"{name:12s} | "
        f"min={image.min():7.2f} | "
        f"max={image.max():7.2f} | "
        f"mean={image.mean():7.2f}"
    )

# Prueba
map_1 = gp_erosion(random_image)
map_2 = gp_dilation(random_image)

generated = gp_add(map_1, map_2)

fig, axes = plt.subplots(1, 4, figsize=(10, 3))

axes[0].imshow(random_image, cmap="gray", vmin=0, vmax=255)
axes[0].set_title("Input")
axes[0].axis("off")

axes[1].imshow(map_1, cmap="gray", vmin=0, vmax=255)
axes[1].set_title("Erosion")
axes[1].axis("off")

axes[2].imshow(map_2, cmap="gray", vmin=0, vmax=255)
axes[2].set_title("Dilation")
axes[2].axis("off")

axes[3].imshow(generated, cmap="gray", vmin=0, vmax=255)
axes[3].set_title("Combination")
axes[3].axis("off")

plt.tight_layout()
# plt.show()