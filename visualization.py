from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from geometric_images import (
    clip_image, create_point, create_horizontal_line, create_vertical_line, create_circle, 
    create_diagonal_line, create_rectangle, create_square, IMAGE, FLOAT)

# Terminaales geométricas
POINT = create_point()
LINE_H = create_horizontal_line()
LINE_V = create_vertical_line()
LINE_D1 = create_diagonal_line(direction=1)
LINE_D2 = create_diagonal_line(direction=-1)
CIRCLE = create_circle()
SQUARE = create_square()
RECTANGLE = create_rectangle(width=16,height=10)

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
# fig, axes = plt.subplots( 2, 4, figsize=(10, 5))
# for ax, (name, image) in zip(axes.ravel(), geometric_terminals.items()):
#     ax.imshow(image, cmap="gray", vmin=0, vmax=255)
#     ax.set_title(name)
#     ax.axis("off")
# plt.tight_layout()
# plt.show()