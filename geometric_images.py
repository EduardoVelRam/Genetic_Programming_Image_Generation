import numpy as np
import random
import random
from deap import base, creator, gp, tools, algorithms
import numpy as np
import matplotlib.pyplot as plt
from skimage.morphology import (
    erosion,
    dilation,
    opening,
    closing,
    disk
)
from scipy.ndimage import convolve
from skimage.draw import line
from skimage.morphology import dilation, disk

print("geometric_images")
IMAGE = np.ndarray
FLOAT = float

def clip_image(image):

    return np.clip(
        image,
        0.0,
        255.0
    ).astype(np.float32)

# Construcción de las imágenes geométricas
# Punto
def create_point(size=28):

    image = np.zeros(
        (size, size),
        dtype=np.float32
    )

    center = size // 2

    image[center, center] = 255.0

    return image

# Línea horizontal
def create_horizontal_line(
    size=28,
    thickness=2
):

    image = np.zeros(
        (size, size),
        dtype=np.float32
    )

    center = size // 2
    half = thickness // 2

    image[
        center - half:center + half + 1,
        3:size - 3
    ] = 255.0

    return image

# Línea vertical
def create_vertical_line(
    size=28,
    thickness=2
):

    image = np.zeros(
        (size, size),
        dtype=np.float32
    )

    center = size // 2
    half = thickness // 2

    image[
        3:size - 3,
        center - half:center + half + 1
    ] = 255.0

    return image

# Líneas diagonales
def create_diagonal_line(
    size=28,
    direction=1,
    thickness=2
):

    image = np.zeros(
        (size, size),
        dtype=np.float32
    )

    if direction == 1:

        r0, c0 = 3, 3
        r1, c1 = size - 4, size - 4

    else:

        r0, c0 = 3, size - 4
        r1, c1 = size - 4, 3

    rr, cc = line(
        r0,
        c0,
        r1,
        c1
    )

    image[rr, cc] = 255.0

    if thickness > 1:

        image = dilation(
            image,
            footprint=disk(
                max(1, thickness // 2)
            )
        )

    return image.astype(np.float32)

# Círculo
def create_circle(
    size=28,
    radius=7,
    x=None,
    y=None
):

    image = np.zeros(
        (size, size),
        dtype=np.float32
    )

    if x is None:
        x = size // 2

    if y is None:
        y = size // 2

    yy, xx = np.ogrid[
        :size,
        :size
    ]

    mask = (
        (xx - x) ** 2 +
        (yy - y) ** 2
        <= radius ** 2
    )

    image[mask] = 255.0

    return image

# Cuadrado
def create_square(
    size=28,
    side=12
):

    image = np.zeros(
        (size, size),
        dtype=np.float32
    )

    center = size // 2
    half = side // 2

    y0 = max(
        0,
        center - half
    )

    y1 = min(
        size,
        center + half
    )

    x0 = max(
        0,
        center - half
    )

    x1 = min(
        size,
        center + half
    )

    image[
        y0:y1,
        x0:x1
    ] = 255.0

    return image

# Rectáangulo
def create_rectangle(
    size=28,
    width=16,
    height=10
):

    image = np.zeros(
        (size, size),
        dtype=np.float32
    )

    center = size // 2

    x0 = max(
        0,
        center - width // 2
    )

    x1 = min(
        size,
        center + width // 2
    )

    y0 = max(
        0,
        center - height // 2
    )

    y1 = min(
        size,
        center + height // 2
    )

    image[
        y0:y1,
        x0:x1
    ] = 255.0

    return image