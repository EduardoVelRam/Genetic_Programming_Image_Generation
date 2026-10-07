import numpy as np
import random
import operator
import random
from geometric_images import create_circle
# from primitives_operations import gp_add, gp_closing, gp_convolution, gp_dilation, gp_erosion, gp_multiply, gp_opening, gp_subtract
# from primitives_operations import target_image, target_class, target_images, mse, random_image
from deap import base, creator, gp, tools, algorithms
import numpy as np
import matplotlib.pyplot as plt
from skimage.draw import line
from skimage.morphology import (
    erosion,
    dilation,
    opening,
    closing,
    disk
)
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

