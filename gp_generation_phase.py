from geometric_images import (
    clip_image, create_point, create_horizontal_line, create_vertical_line, create_circle, 
    create_diagonal_line, create_rectangle, create_square, IMAGE, FLOAT)
from import_data import load_mnist, mnist_path
import operator
import random
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import convolve
from skimage.draw import line
from skimage.morphology import (
    erosion,
    dilation,
    opening,
    closing,
    disk
)
from deap import base, creator, gp, tools, algorithms

X_train, y_train, X_test, y_test = load_mnist(mnist_path)

print("gp_generation_phase")

# Terminaales geométricas
POINT = create_point()
LINE_H = create_horizontal_line()
LINE_V = create_vertical_line()
LINE_D1 = create_diagonal_line(direction=1)
LINE_D2 = create_diagonal_line(direction=-1)
CIRCLE = create_circle()
SQUARE = create_square()
RECTANGLE = create_rectangle(width=16,height=10)

# Verificación de que son matrices
# print(type(CIRCLE))
# print(CIRCLE.shape)
# print(CIRCLE.dtype)

# Visualizar las terminales
geometric_terminals = {
    "POINT": POINT,
    "LINE_H": LINE_H,
    "LINE_V": LINE_V,
    "LINE_D1": LINE_D1,
    "LINE_D2": LINE_D2,
    "CIRCLE": CIRCLE,
    "SQUARE": SQUARE,
    "RECTANGLE": RECTANGLE
}

fig, axes = plt.subplots( 2, 4, figsize=(10, 5))

for ax, (name, image) in zip(axes.ravel(), geometric_terminals.items()):

    ax.imshow(image, cmap="gray", vmin=0, vmax=255)
    ax.set_title(name)
    ax.axis("off")

plt.tight_layout()
plt.show()

# Primitivas geométricas paramétricas
# Círculo
def gp_circle(radius, x, y):
    radius = float(np.clip(radius, 1.0, 14.0))
    x = float(np.clip(x,0.0,27.0))
    y = float(np.clip(y,0.0,27.0))
    return create_circle( size=28,radius=radius,x=x, y=y)

# Línea
def gp_line(angle,length,thickness):
    angle = float(angle)
    length = float(np.clip( length, 2.0, 28.0))
    thickness = float( np.clip( thickness, 1.0, 6.0) )
    image = np.zeros((28, 28), dtype=np.float32)
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
        image = dilation( image,footprint=disk(max(1, int(thickness // 2))))

    return image.astype(np.float32)

# REctángulo
def gp_rectangle(width, height, x, y):
    width = float(np.clip(width, 2.0, 28.0))
    height = float(np.clip(height,2.0,28.0))
    x = float(np.clip(x,0.0,27.0))
    y = float(np.clip(y, 0.0,27.0))

    image = np.zeros((28, 28),dtype=np.float32)
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

# OPERADORES MORFOLÓGICOS
selem = disk(1)

average_kernel = (
    np.ones(
        (3, 3),
        dtype=np.float32
    ) / 9.0
)

def gp_erosion(image):
    return clip_image(erosion(image, footprint=selem))

def gp_dilation(image):
    return clip_image(dilation(image,footprint=selem))

def debug_image(name, image):
    print(f"\n[{name}]")
    print("type:", type(image))
    if image is None:
        print("ERROR: image is None")
        return None
    print("shape:", image.shape)
    print("dtype:", image.dtype)
    print("min:", image.min())
    print("max:", image.max())
    print("mean:", image.mean())
    return image

def gp_dilation(image):
    debug_image("gp_dilation - entrada", image)
    selem = np.ones((3, 3), dtype=np.uint8)
    result = dilation(image, footprint=selem)
    return clip_image(result)

def gp_opening(image):
    return clip_image(opening(image,footprint=selem))

def gp_closing(image):
    return clip_image(closing(image,footprint=selem))

def gp_convolution(image):
    return clip_image(convolve(image,average_kernel, mode="reflect"))

# OPERADORES BINARIOS
def gp_add(image_a, image_b):
    return clip_image(image_a + image_b)

def gp_subtract(image_a, image_b):
    return clip_image(image_a - image_b)

def gp_multiply(image_a, image_b):
    return clip_image((image_a * image_b) / 255.0)

pset = gp.PrimitiveSetTyped(
    "MAIN",
    [IMAGE],
    IMAGE
)

pset.renameArguments(ARG0="I")

# Registrar las terminales geométricas
pset.addTerminal(POINT, IMAGE, name="POINT")
pset.addTerminal(LINE_H,IMAGE, name="LINE_H")
pset.addTerminal(LINE_V, IMAGE, name="LINE_V")
pset.addTerminal(LINE_D1, IMAGE, name="LINE_D1")
pset.addTerminal(LINE_D2, IMAGE, name="LINE_D2")
pset.addTerminal(CIRCLE, IMAGE, name="CIRCLE")
pset.addTerminal(SQUARE, IMAGE, name="SQUARE")
pset.addTerminal(RECTANGLE,IMAGE, name="RECTANGLE")

# Prmitivas geométricas paramétricas
pset.addPrimitive(gp_circle, [FLOAT, FLOAT, FLOAT], IMAGE, name="CIRCLE_PARAM")
pset.addPrimitive(gp_line, [FLOAT, FLOAT, FLOAT], IMAGE, name="LINE_PARAM")
pset.addPrimitive(gp_rectangle, [FLOAT, FLOAT, FLOAT, FLOAT], IMAGE, name="RECTANGLE_PARAM")

# PARÁMETROS EVOLUTIVOS
pset.addEphemeralConstant("ANGLE", lambda: random.uniform(-np.pi, np.pi), FLOAT)
pset.addEphemeralConstant("LENGTH",lambda: random.uniform(2.0, 28.0),FLOAT)
pset.addEphemeralConstant("THICKNESS",lambda: random.uniform(1.0,6.0), FLOAT)
pset.addEphemeralConstant("RADIUS",lambda: random.uniform(1.0,14.0),FLOAT)
pset.addEphemeralConstant("X",lambda: random.uniform(0.0,27.0),FLOAT)
pset.addEphemeralConstant("Y", lambda: random.uniform(0.0,27.0),FLOAT)
pset.addEphemeralConstant("WIDTH",lambda: random.uniform(2.0,28.0),FLOAT)
pset.addEphemeralConstant("HEIGHT",lambda: random.uniform(2.0,28.0),FLOAT)

# Operadores de la imagen
pset.addPrimitive(gp_erosion, [IMAGE], IMAGE)
pset.addPrimitive(gp_dilation,[IMAGE], IMAGE)
pset.addPrimitive(gp_opening, [IMAGE], IMAGE)
pset.addPrimitive(gp_closing, [IMAGE], IMAGE)
pset.addPrimitive(gp_convolution,[IMAGE], IMAGE)
pset.addPrimitive(gp_add,[IMAGE, IMAGE], IMAGE)
pset.addPrimitive(gp_subtract, [IMAGE, IMAGE], IMAGE)
pset.addPrimitive(gp_multiply, [IMAGE, IMAGE], IMAGE)

# Comprobar conjuntos de primitivas
print("Aquí viene lo bueno")
print(pset)

# Creator
if not hasattr(creator, "FitnessMin"):
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))

if not hasattr(creator, "Individual"):
    creator.create("Individual", gp.PrimitiveTree,fitness=creator.FitnessMin)

gp_state = {
    "input_image": None,
    "target_image": None,
    "target_class": 4
}
