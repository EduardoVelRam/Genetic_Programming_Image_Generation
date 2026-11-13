# -*- coding: utf-8 -*-
"""
manual_generation.py
--------------------
Reconstruye manualmente un individuo obtenido con gp_generation_phase.py.

1. Copia la expresion impresa como "Mejor individuo".
2. Sustituye el contenido de INDIVIDUO (abajo) por esa expresion.
3. Ejecuta: python manual_generation.py

La salida se muestra y se guarda en ./img_salida/imagen_generada.png.
No ejecuta evolucion, no requiere DEAP, MNIST ni clasificadores entrenados.

Dependencias: pip install numpy scipy scikit-image matplotlib pillow

Notas de fidelidad:
- Las operaciones se reproducen segun los archivos compartidos.
- Las terminales geometricas se construyen con geometric_images.py
  (no se adjunto visualization.py, donde se definen originalmente).
- Line_Param conserva el comportamiento del archivo original, cuyo trazo
  esta comentado (por ello retorna una imagen negra).
"""

import ast
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy.ndimage import convolve
from skimage import draw
from skimage.morphology import closing, dilation, disk, erosion, opening


# =====================================================================
# 1. CONFIGURACION: PEGA AQUI EL MEJOR INDIVIDUO DE GP
# =====================================================================
INDIVIDUO = """
convolution_mean(gp_subtract(gp_subtract(Field_Add_Constant(gp_subtract(gp_dilation(convolution_mean(Circle)), Ellipse_Param(14.0, 14.0, 3.0, 12.0, 2.629371249127434)), 14.0), Z_val), gp_subtract(gp_dilation(gp_dilation(convolution_mean(convolution_mean(Line_D1)))), Ellipse_Param(10.0, 14.0, 20.0, 3.0, 1.0))))
""".strip()

TAMANO = 28
CARPETA_SALIDA = Path(__file__).resolve().parent / "img_salida"
NOMBRE_IMAGEN = "imagen_generada.png"
MOSTRAR_IMAGEN = True  # False si se ejecuta sin interfaz grafica.


# =====================================================================
# 2. GEOMETRIA: mismas implementaciones de geometric_images.py
# =====================================================================
def clip_image(image):
    image = np.asarray(image, dtype=np.float32)
    image = np.nan_to_num(image, nan=0.0, posinf=255.0, neginf=0.0)
    return np.clip(image, 0.0, 255.0)


def create_point(size=28):
    image = np.zeros((size, size), dtype=np.float32)
    center = size // 2
    image[center, center] = 255.0
    return image


def create_horizontal_line(size=28, thickness=2):
    image = np.zeros((size, size), dtype=np.float32)
    center = size // 2
    half = thickness // 2
    image[center - half:center + half + 1, 3:size - 3] = 255.0
    return image.astype(np.float32)


def create_vertical_line(size=28, thickness=2):
    image = np.zeros((size, size), dtype=np.float32)
    center = size // 2
    half = thickness // 2
    image[3:size - 3, center - half:center + half + 1] = 255.0
    return image.astype(np.float32)


def create_diagonal_line(size=28, direction=1, thickness=2):
    image = np.zeros((size, size), dtype=np.float32)
    if direction == 1:
        r0, c0, r1, c1 = 3, 3, size - 4, size - 4
    else:
        r0, c0, r1, c1 = 3, size - 4, size - 4, 3
    rr, cc = draw.line(r0, c0, r1, c1)
    image[rr, cc] = 255.0
    if thickness > 1:
        image = dilation(image, footprint=disk(max(1, thickness // 2)))
    return image.astype(np.float32)


def create_circle(size=28, radius=7, x=None, y=None):
    image = np.zeros((size, size), dtype=np.float32)
    if x is None:
        x = size // 2
    if y is None:
        y = size // 2
    yy, xx = np.ogrid[:size, :size]
    image[(xx - x)**2 + (yy - y)**2 <= radius**2] = 255.0
    return image.astype(np.float32)


def create_square(size=28, side=12):
    image = np.zeros((size, size), dtype=np.float32)
    center = size // 2
    half = side // 2
    image[max(0, center-half):min(size, center+half),
          max(0, center-half):min(size, center+half)] = 255.0
    return image.astype(np.float32)


def create_rectangle(size=28, width=16, height=10):
    image = np.zeros((size, size), dtype=np.float32)
    center = size // 2
    x0, x1 = max(0, center-width//2), min(size, center+width//2)
    y0, y1 = max(0, center-height//2), min(size, center+height//2)
    image[y0:y1, x0:x1] = 255.0
    return image.astype(np.float32)


# =====================================================================
# 3. GEOMETRIA PARAMETRICA: geometric_primitives_gp.py
# =====================================================================
def gp_circle(radius, x, y):
    radius = float(np.clip(radius, 1.0, 14.0))
    x = float(np.clip(x, 0.0, 27.0))
    y = float(np.clip(y, 0.0, 27.0))
    return create_circle(size=28, radius=radius, x=x, y=y)


def gp_line(angle, length, thickness):
    # Replica literal del codigo original: el trazo sobre image esta
    # comentado en geometric_primitives_gp.py. Por fidelidad se conserva.
    angle = float(angle)
    length = float(np.clip(length, 2.0, 28.0))
    thickness = float(np.clip(thickness, 1.0, 6.0))
    image = np.zeros((28, 28), dtype=np.float32)
    center = 13.5
    half_length = length / 2.0
    dx = np.cos(angle) * half_length
    dy = np.sin(angle) * half_length
    x0, y0 = center - dx, center - dy
    x1, y1 = center + dx, center + dy
    rr, cc = draw.line(int(round(y0)), int(round(x0)),
                       int(round(y1)), int(round(x1)))
    # image[rr, cc] = 255.0  # asi aparece en el codigo de origen
    if thickness > 1:
        image = dilation(image, footprint=disk(max(1, int(thickness // 2))))
    return image.astype(np.float32)


def gp_rectangle(width, height, x, y):
    width = float(np.clip(width, 2.0, 28.0))
    height = float(np.clip(height, 2.0, 28.0))
    x = float(np.clip(x, 0.0, 27.0))
    y = float(np.clip(y, 0.0, 27.0))
    image = np.zeros((28, 28), dtype=np.float32)
    x0, x1 = int(round(x-width/2)), int(round(x+width/2))
    y0, y1 = int(round(y-height/2)), int(round(y+height/2))
    x0, x1 = max(0, x0), min(28, x1)
    y0, y1 = max(0, y0), min(28, y1)
    image[y0:y1, x0:x1] = 255.0
    return image


def gp_point(x, y, radius):
    x = float(np.clip(x, 0.0, 27.0))
    y = float(np.clip(y, 0.0, 27.0))
    radius = float(np.clip(radius, 1.0, 5.0))
    image = np.zeros((28, 28), dtype=np.float32)
    rr, cc = draw.disk((y, x), radius, shape=image.shape)
    image[rr, cc] = 255.0
    return image


def gp_ellipse(x, y, width, height, angle):
    x = float(np.clip(x, 0.0, 27.0))
    y = float(np.clip(y, 0.0, 27.0))
    width = float(np.clip(width, 2.0, 28.0))
    height = float(np.clip(height, 2.0, 28.0))
    angle = float(angle)
    image = np.zeros((28, 28), dtype=np.float32)
    rr, cc = draw.ellipse(y, x, height/2.0, width/2.0,
                          rotation=angle, shape=image.shape)
    image[rr, cc] = 255.0
    return image


def gp_triangle(x, y, size, angle):
    x = float(np.clip(x, 0.0, 27.0))
    y = float(np.clip(y, 0.0, 27.0))
    size = float(np.clip(size, 3.0, 28.0))
    angle = float(angle)
    image = np.zeros((28, 28), dtype=np.float32)
    angles = np.array([0.0, 2.0*np.pi/3.0, 4.0*np.pi/3.0]) + angle
    radius = size / np.sqrt(3.0)
    vertices_x = x + radius*np.cos(angles)
    vertices_y = y + radius*np.sin(angles)
    rr, cc = draw.polygon(vertices_y, vertices_x, shape=image.shape)
    image[rr, cc] = 255.0
    return image


# =====================================================================
# 4. FUNCIONES DE CAMPO: trigonometric_expressions.py
# =====================================================================
def create_coordinate_grid(size=28):
    x = np.linspace(-np.pi, np.pi, size, dtype=np.float32)
    y = np.linspace(-np.pi, np.pi, size, dtype=np.float32)
    X, Y = np.meshgrid(x, y)
    Z = np.sqrt(X**2 + Y**2)
    return X.astype(np.float32), Y.astype(np.float32), Z.astype(np.float32)


def field_add(a, b):
    return np.asarray(a, dtype=np.float32) + np.asarray(b, dtype=np.float32)


def field_subtract(a, b):
    return np.asarray(a, dtype=np.float32) - np.asarray(b, dtype=np.float32)


def field_multiply(a, b):
    return np.asarray(a, dtype=np.float32) * np.asarray(b, dtype=np.float32)


def field_divide(a, b):
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    return np.divide(a, b, out=np.zeros_like(a), where=np.abs(b) > 1e-6)


def field_scale(a, c):
    return np.asarray(a, dtype=np.float32) * float(c)


def field_add_constant(a, c):
    return np.asarray(a, dtype=np.float32) + float(c)


def field_sin(a):
    result = np.sin(np.asarray(a, dtype=np.float32))
    result = np.nan_to_num(result, nan=0.0, posinf=10.0, neginf=-10.0)
    return np.clip(result, -10.0, 10.0).astype(np.float32)


def field_cos(a):
    result = np.cos(np.asarray(a, dtype=np.float32))
    result = np.nan_to_num(result, nan=0.0, posinf=10.0, neginf=-10.0)
    return np.clip(result, -10.0, 10.0).astype(np.float32)


def field_tan(a):
    result = np.tan(np.asarray(a, dtype=np.float32))
    result = np.nan_to_num(result, nan=0.0, posinf=10.0, neginf=-10.0)
    return np.clip(result, -10.0, 10.0).astype(np.float32)


def field_to_image(field):
    field = np.asarray(field, dtype=np.float32)
    field = np.nan_to_num(field, nan=0.0, posinf=0.0, neginf=0.0)
    minimum = np.min(field)
    maximum = np.max(field)
    if maximum - minimum < 1e-8:
        return np.zeros_like(field, dtype=np.float32)
    return ((field - minimum) / (maximum - minimum) * 255.0).astype(np.float32)


def field_ab_surface(a=3.0, b=60.0):
    x_mesh, y_mesh = np.meshgrid(
        np.arange(1, 29, dtype=np.float32),
        np.arange(1, 29, dtype=np.float32)
    )
    a = max(abs(float(a)), 1e-3)
    b = max(abs(float(b)), 1e-3)
    result = (np.cos(x_mesh/a)*np.sin(y_mesh/a)
              + np.cos(x_mesh/b)*np.sin(y_mesh/b))
    return result.astype(np.float32)


# =====================================================================
# 5. MORFOLOGIA Y OPERACIONES SOBRE IMAGEN: gp_generation_phase.py
# =====================================================================
DISCO = disk(1)
KERNEL_PROMEDIO = np.ones((3, 3), dtype=np.float32) / 9.0


def gp_erosion(image):
    return clip_image(erosion(image, footprint=DISCO))


def gp_dilation(image):
    return clip_image(dilation(image, footprint=np.ones((3, 3), dtype=np.uint8)))


def gp_opening(image):
    return clip_image(opening(image, footprint=DISCO))


def gp_closing(image):
    return clip_image(closing(image, footprint=DISCO))


def gp_convolution(image):
    return clip_image(convolve(image, KERNEL_PROMEDIO, mode="reflect"))


def gp_add(a, b):
    return clip_image(a + b)


def gp_subtract(a, b):
    return clip_image(a - b)


def gp_multiply(a, b):
    return clip_image(np.multiply(a, b))


# =====================================================================
# 6. NOMBRES DE TERMINALES Y PRIMITIVAS (IGUALES A LOS DE GP)
# =====================================================================
def construir_entorno():
    X, Y, Z = create_coordinate_grid(TAMANO)

    terminales = {
        "Point": create_point(TAMANO),
        "Line_H": create_horizontal_line(TAMANO),
        "Line_V": create_vertical_line(TAMANO),
        "Line_D1": create_diagonal_line(TAMANO, direction=1),
        "Line_D2": create_diagonal_line(TAMANO, direction=2),
        "Circle": create_circle(TAMANO),
        "Square": create_square(TAMANO),
        "Rectangle": create_rectangle(TAMANO),
        "X_val": X,
        "Y_val": Y,
        "Z_val": Z,
        "A_val": 3.0,
        "B_val": 60.0,
    }

    funciones = {
        "Circle_Param": gp_circle,
        "Line_Param": gp_line,
        "Rectangle_Param": gp_rectangle,
        "Point_Param": gp_point,
        "Ellipse_Param": gp_ellipse,
        "Triangle_Param": gp_triangle,
        "Sin": field_sin,
        "Cos": field_cos,
        "Tan": field_tan,
        "Coseno": np.cos,
        "Seno": np.sin,
        "Add": field_add,
        "Field_Subtract": field_subtract,
        "Field_Multiply": field_multiply,
        "Field_Divide": field_divide,
        "Field_To_Image": field_to_image,
        "Field_Scale": field_scale,
        "Field_Add_Constant": field_add_constant,
        "AB_SURFACE": field_ab_surface,
        "gp_erosion": gp_erosion,
        "gp_dilation": gp_dilation,
        "gp_opening": gp_opening,
        "gp_closing": gp_closing,
        "convolution_mean": gp_convolution,
        "gp_add": gp_add,
        "gp_subtract": gp_subtract,
        "gp_multiply": gp_multiply,
    }
    return terminales, funciones


# =====================================================================
# 7. INTERPRETACION SEGURA DEL ARBOL, SIN EJECUTAR CODIGO ARBITRARIO
# =====================================================================
def evaluar_expresion(expresion, terminales_adicionales=None):
    terminales, funciones = construir_entorno()

    if terminales_adicionales:
        terminales.update(terminales_adicionales)

    arbol = ast.parse(expresion.strip(), mode="eval")

    def evaluar(nodo):
        if isinstance(nodo, ast.Constant):
            if type(nodo.value) in (int, float):
                return nodo.value
            raise ValueError("Solo se permiten constantes numericas.")
        if isinstance(nodo, ast.UnaryOp) and isinstance(nodo.op, (ast.UAdd, ast.USub)):
            valor = evaluar(nodo.operand)
            if type(valor) not in (int, float):
                raise ValueError("El signo +/- solo se permite en constantes numericas.")
            return -valor if isinstance(nodo.op, ast.USub) else valor
        if isinstance(nodo, ast.Name):
            if nodo.id not in terminales:
                raise ValueError(f"Terminal desconocida: {nodo.id}")
            return terminales[nodo.id]
        if isinstance(nodo, ast.Call):
            if not isinstance(nodo.func, ast.Name) or nodo.keywords:
                raise ValueError("Solo se permiten llamadas directas sin argumentos keyword.")
            nombre = nodo.func.id
            if nombre not in funciones:
                raise ValueError(f"Primitiva desconocida: {nombre}")
            return funciones[nombre](*(evaluar(argumento) for argumento in nodo.args))
        raise ValueError(f"Elemento no permitido en el individuo: {type(nodo).__name__}")

    imagen = np.asarray(evaluar(arbol.body), dtype=np.float32)
    if imagen.shape != (TAMANO, TAMANO):
        raise ValueError(f"Se esperaba una imagen {(TAMANO, TAMANO)}, se obtuvo {imagen.shape}")
    if not np.all(np.isfinite(imagen)):
        raise ValueError("El individuo genero valores NaN o infinitos.")
    return imagen

# =====================================================================
# 8. TRANSFORMACION MANUAL DE IMAGENES
# =====================================================================

# CONFIGURACION
CARPETA_ENTRADA = Path(__file__).resolve().parent / "img_entrada"

EXPRESION_TRANSFORMACION = """
gp_subtract(imagen, Ellipse_Param(14.0, 14.0, 3.0, 12.0, 2.629371249127434))
""".strip()


def transformar_imagen():

    # ---------------------------------------------------------
    # 1. Buscar imagen de entrada
    # ---------------------------------------------------------
    if not CARPETA_ENTRADA.is_dir():
        raise FileNotFoundError(
            f"No existe la carpeta: {CARPETA_ENTRADA}"
        )

    extensiones = {
        ".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"
    }

    archivos = [
        archivo
        for archivo in CARPETA_ENTRADA.iterdir()
        if archivo.is_file() and archivo.suffix.lower() in extensiones
    ]

    if len(archivos) != 1:
        raise ValueError(
            f"Se esperaba exactamente una imagen en img_entrada. "
            f"Se encontraron {len(archivos)}."
        )

    archivo_entrada = archivos[0]

    # ---------------------------------------------------------
    # 2. Cargar y preparar imagen
    # ---------------------------------------------------------
    with Image.open(archivo_entrada) as img:
        img = img.convert("L")

        if img.size != (TAMANO, TAMANO):
            print(f"Redimensionando imagen a {TAMANO}x{TAMANO}")
            img = img.resize(
                (TAMANO, TAMANO),
                Image.Resampling.BILINEAR
            )

        imagen_entrada = np.asarray(img, dtype=np.float32)

    # ---------------------------------------------------------
    # 3. Aplicar expresion
    # ---------------------------------------------------------
    imagen_salida = evaluar_expresion(
        EXPRESION_TRANSFORMACION,
        terminales_adicionales={
            "imagen": imagen_entrada
        }
    )

    # ---------------------------------------------------------
    # 4. Guardar resultado
    # ---------------------------------------------------------
    CARPETA_SALIDA.mkdir(parents=True, exist_ok=True)

    destino = CARPETA_SALIDA / "transformed_image.png"

    imagen_png = np.rint(
        np.clip(imagen_salida, 0, 255)
    ).astype(np.uint8)

    Image.fromarray(imagen_png).save(destino)

    print(f"Imagen de entrada: {archivo_entrada.name}")
    print(f"Imagen guardada en: {destino}")

    # ---------------------------------------------------------
    # 5. Visualizar resultados
    # ---------------------------------------------------------
    if MOSTRAR_IMAGEN:

        fig, axes = plt.subplots(1, 2, figsize=(8, 4))

        axes[0].imshow(
            imagen_entrada,
            cmap="gray",
            vmin=0,
            vmax=255,
            interpolation="nearest"
        )
        axes[0].set_title("Imagen de entrada")

        axes[1].imshow(
            imagen_salida,
            cmap="gray",
            vmin=0,
            vmax=255,
            interpolation="nearest"
        )
        axes[1].set_title("Imagen transformada")

        for ax in axes:
            ax.axis("off")

        plt.tight_layout()
        plt.show()

    return imagen_salida

def main():
    imagen = evaluar_expresion(INDIVIDUO)
    CARPETA_SALIDA.mkdir(parents=True, exist_ok=True)
    destino = CARPETA_SALIDA / NOMBRE_IMAGEN

    # Se convierte a 8 bits unicamente para guardar PNG en escala de grises.
    # No se realiza una normalizacion extra: usa las mismas intensidades
    # [0, 255] que la visualizacion de gp_generation_phase.py.
    imagen_png = np.rint(np.clip(imagen, 0, 255)).astype(np.uint8)
    Image.fromarray(imagen_png).save(destino)

    print("Individuo reconstruido correctamente.")
    print(f"Tamano: {imagen.shape}; dtype: {imagen.dtype}")
    print(f"Min: {imagen.min():.4f}; Max: {imagen.max():.4f}; Media: {imagen.mean():.4f}")
    print(f"Imagen guardada en: {destino}")

    if MOSTRAR_IMAGEN:
        plt.figure(figsize=(5, 5))
        plt.imshow(imagen, cmap="gray", vmin=0, vmax=255, interpolation="nearest")
        plt.title("Individuo reconstruido (28 × 28)")
        plt.axis("off")
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":

    # MODO 1: Generación manual con individuo GP
    # main()

    # MODO 2: Transformación de imagen de entrada
    transformar_imagen()