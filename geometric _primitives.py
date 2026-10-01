import numpy as np
import random
import operator
import random
# from primitives_operations import gp_add, gp_closing, gp_convolution, gp_dilation, gp_erosion, gp_multiply, gp_opening, gp_subtract
# from primitives_operations import target_image, target_class, target_images, mse, random_image
from deap import base, creator, gp, tools, algorithms
import numpy as np
import matplotlib.pyplot as plt

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

