import numpy as np
import random
import operator
import random
# from primitives_operations import gp_add, gp_closing, gp_convolution, gp_dilation, gp_erosion, gp_multiply, gp_opening, gp_subtract
from primitives_operations import target_image, target_class, target_images, mse, random_image
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

IMAGE = np.ndarray
FLOAT = float

# PrimitiveSet a PrimitiveSetTyped, porque ahora el árbol manejará dos tipos de valores:
# IMAGE: matrices 28×28.
# FLOAT: parámetros evolutivos como radio, coordenadas, longitud, etc.

pset = gp.PrimitiveSetTyped(
    "MAIN",
    [IMAGE],
    IMAGE
)

creator.create(
    "FitnessMin",
    base.Fitness,
    weights=(-1.0,)
)

creator.create(
    "Individual",
    gp.PrimitiveTree,
    fitness=creator.FitnessMin
)

pset.renameArguments(ARG0="I")

# Construcción de imágenes geométricas
def create_point(size=28):
    image = np.zeros((size, size), dtype=np.float32)

    center = size // 2
    image[center, center] = 255.0

    return image

def create_horizontal_line(size=28, thickness=2):
    image = np.zeros((size, size), dtype=np.float32)

    center = size // 2

    half = thickness // 2

    image[
        center - half:center + half + 1,
        3:size - 3
    ] = 255.0

    return image

def create_vertical_line(size=28, thickness=2):
    image = np.zeros((size, size), dtype=np.float32)

    center = size // 2

    half = thickness // 2

    image[
        3:size - 3,
        center - half:center + half + 1
    ] = 255.0

    return image



def create_diagonal_line(size=28, direction=1, thickness=2):

    image = np.zeros((size, size), dtype=np.float32)

    if direction == 1:
        r0, c0 = 3, 3
        r1, c1 = size - 4, size - 4
    else:
        r0, c0 = 3, size - 4
        r1, c1 = size - 4, 3

    rr, cc = line(r0, c0, r1, c1)

    image[rr, cc] = 255.0

    if thickness > 1:
        image = dilation(
            image,
            footprint=disk(thickness // 2)
        )

    return image.astype(np.float32)

def create_circle(
    size=28,
    radius=7,
    x=None,
    y=None
):

    image = np.zeros((size, size), dtype=np.float32)

    if x is None:
        x = size // 2

    if y is None:
        y = size // 2

    yy, xx = np.ogrid[:size, :size]

    mask = (
        (xx - x) ** 2 +
        (yy - y) ** 2
        <= radius ** 2
    )

    image[mask] = 255.0

    return image

def create_square(size=28, side=12):

    image = np.zeros((size, size), dtype=np.float32)

    center = size // 2

    half = side // 2

    y0 = max(0, center - half)
    y1 = min(size, center + half)

    x0 = max(0, center - half)
    x1 = min(size, center + half)

    image[y0:y1, x0:x1] = 255.0

    return image

def create_rectangle(
    size=28,
    width=16,
    height=10
):

    image = np.zeros((size, size), dtype=np.float32)

    center = size // 2

    x0 = max(0, center - width // 2)
    x1 = min(size, center + width // 2)

    y0 = max(0, center - height // 2)
    y1 = min(size, center + height // 2)

    image[y0:y1, x0:x1] = 255.0

    return image

# Construcción de las terminales

POINT = create_point()

LINE_H = create_horizontal_line()

LINE_V = create_vertical_line()

LINE_D1 = create_diagonal_line(
    direction=1
)

LINE_D2 = create_diagonal_line(
    direction=-1
)

CIRCLE = create_circle()

SQUARE = create_square()

RECTANGLE = create_rectangle(
    width=16,
    height=10
)

# Registrar las imágenes como terminales
pset.addTerminal(
    POINT,
    IMAGE,
    name="POINT"
)

pset.addTerminal(
    LINE_H,
    IMAGE,
    name="LINE_H"
)

pset.addTerminal(
    LINE_V,
    IMAGE,
    name="LINE_V"
)

pset.addTerminal(
    LINE_D1,
    IMAGE,
    name="LINE_D1"
)

pset.addTerminal(
    LINE_D2,
    IMAGE,
    name="LINE_D2"
)

pset.addTerminal(
    CIRCLE,
    IMAGE,
    name="CIRCLE"
)

pset.addTerminal(
    SQUARE,
    IMAGE,
    name="SQUARE"
)

pset.addTerminal(
    RECTANGLE,
    IMAGE,
    name="RECTANGLE"
)


# Primitivas geométricas evolutivas
# Circulo
def gp_circle(radius, x, y):

    radius = float(np.clip(radius, 1.0, 14.0))
    x = float(np.clip(x, 0.0, 27.0))
    y = float(np.clip(y, 0.0, 27.0))

    return create_circle(
        size=28,
        radius=radius,
        x=x,
        y=y
    )

pset.addPrimitive(
    gp_circle,
    [FLOAT, FLOAT, FLOAT],
    IMAGE,
    name="CIRCLE_PARAM"
)

# Línea
def gp_line(angle, length, thickness):

    angle = float(angle)

    length = float(
        np.clip(length, 2.0, 28.0)
    )

    thickness = float(
        np.clip(thickness, 1.0, 6.0)
    )

    image = np.zeros(
        (28, 28),
        dtype=np.float32
    )

    center = 13.5

    half_length = length / 2.0

    dx = np.cos(angle) * half_length
    dy = np.sin(angle) * half_length

    x0 = center - dx
    y0 = center - dy

    x1 = center + dx
    y1 = center + dy

    rr, cc = line(
        int(round(y0)),
        int(round(x0)),
        int(round(y1)),
        int(round(x1))
    )

    image[rr, cc] = 255.0

    if thickness > 1:

        image = dilation(
            image,
            footprint=disk(
                max(1, int(thickness // 2))
            )
        )

    return image.astype(np.float32)

pset.addPrimitive(
    gp_line,
    [FLOAT, FLOAT, FLOAT],
    IMAGE,
    name="LINE_PARAM"
)

# RECTÁNGULO
def gp_rectangle(
    width,
    height,
    x,
    y
):

    width = float(
        np.clip(width, 2.0, 28.0)
    )

    height = float(
        np.clip(height, 2.0, 28.0)
    )

    x = float(
        np.clip(x, 0.0, 27.0)
    )

    y = float(
        np.clip(y, 0.0, 27.0)
    )

    image = np.zeros(
        (28, 28),
        dtype=np.float32
    )

    x0 = int(round(x - width / 2))
    x1 = int(round(x + width / 2))

    y0 = int(round(y - height / 2))
    y1 = int(round(y + height / 2))

    x0 = max(0, x0)
    x1 = min(28, x1)

    y0 = max(0, y0)
    y1 = min(28, y1)

    image[y0:y1, x0:x1] = 255.0

    return image

pset.addPrimitive(
    gp_rectangle,
    [FLOAT, FLOAT, FLOAT, FLOAT],
    IMAGE,
    name="RECTANGLE_PARAM"
)

# Parámetros evolutivos
pset.addEphemeralConstant(
    "ANGLE",
    lambda: random.uniform(-np.pi, np.pi),
    FLOAT
)

pset.addEphemeralConstant(
    "LENGTH",
    lambda: random.uniform(2.0, 28.0),
    FLOAT
)

pset.addEphemeralConstant(
    "THICKNESS",
    lambda: random.uniform(1.0, 6.0),
    FLOAT
)

pset.addEphemeralConstant(
    "RADIUS",
    lambda: random.uniform(1.0, 14.0),
    FLOAT
)

pset.addEphemeralConstant(
    "X",
    lambda: random.uniform(0.0, 27.0),
    FLOAT
)

pset.addEphemeralConstant(
    "Y",
    lambda: random.uniform(0.0, 27.0),
    FLOAT
)

pset.addEphemeralConstant(
    "WIDTH",
    lambda: random.uniform(2.0, 28.0),
    FLOAT
)

pset.addEphemeralConstant(
    "HEIGHT",
    lambda: random.uniform(2.0, 28.0),
    FLOAT
)

# OPERADORES MORFOLÓGICOS
selem = disk(1)

average_kernel = (
    np.ones((3, 3), dtype=np.float32) / 9.0
)

def clip_image(image):
    return np.clip(image, 0.0, 255.0).astype(np.float32)

def gp_erosion(image):
    return clip_image(
        erosion(image, footprint=selem
        )
    )

def gp_dilation(image):
    return clip_image(
        dilation(image, footprint=selem
        )
    )

def gp_opening(image):
    return clip_image(
        opening(image, footprint=selem
        )
    )

def gp_closing(image):
    return clip_image(
        closing(image, footprint=selem)
    )

def gp_convolution(image):
    return clip_image(
        convolve(image, average_kernel, mode="reflect"
        )
    )

# Registro
pset.addPrimitive(
    gp_erosion,
    [IMAGE],
    IMAGE
)

pset.addPrimitive(
    gp_dilation,
    [IMAGE],
    IMAGE
)

pset.addPrimitive(
    gp_opening,
    [IMAGE],
    IMAGE
)

pset.addPrimitive(
    gp_closing,
    [IMAGE],
    IMAGE
)

pset.addPrimitive(
    gp_convolution,
    [IMAGE],
    IMAGE
)

# Operadores binarios
def gp_add(image_a, image_b):

    return clip_image(
        image_a + image_b
    )


def gp_subtract(image_a, image_b):

    return clip_image(
        image_a - image_b
    )


def gp_multiply(image_a, image_b):

    return clip_image(
        (image_a * image_b) / 255.0
    )

pset.addPrimitive(
    gp_add,
    [IMAGE, IMAGE],
    IMAGE
)

pset.addPrimitive(
    gp_subtract,
    [IMAGE, IMAGE],
    IMAGE
)

pset.addPrimitive(
    gp_multiply,
    [IMAGE, IMAGE],
    IMAGE
)

# Se genera el individuo
toolbox = base.Toolbox()

toolbox.register(
    "expr",
    gp.genHalfAndHalf,
    pset=pset,
    min_=1,
    max_=3
)

toolbox.register(
    "individual",
    tools.initIterate,
    creator.Individual,
    toolbox.expr
)

toolbox.register(
    "population",
    tools.initRepeat,
    list,
    toolbox.individual
)

# Compilar un árbol
toolbox.register(
    "compile",
    gp.compile,
    pset=pset
)

individual = toolbox.individual()

print(individual)

func = toolbox.compile(expr=individual)

result = func(random_image)

# FITNESS
# def evaluate_individual(individual):

#     func = toolbox.compile(
#         expr=individual
#     )

#     generated_image = func(random_image)

#     probability = classifier_probability(
#         generated_image,
#         target_class=4
#     )

#     fitness = 1.0 - probability

#     return (fitness,)


# # Ciclo
# current_image = random_image

# for cycle in range(100):

#     # Ejecutar GP
#     best_image = run_gp(
#         current_image
#     )

#     # Clasificar
#     probability = classifier_probability(
#         best_image,
#         target_class=4
#     )

#     print(
#         f"Ciclo {cycle}: "
#         f"P(4) = {probability:.4f}"
#     )

#     # Criterio de parada
#     if probability >= 0.9:
#         break

#     # La salida se convierte
#     # en la entrada del siguiente ciclo
#     current_image = best_image