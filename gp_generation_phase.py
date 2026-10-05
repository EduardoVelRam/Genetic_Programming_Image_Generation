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
targetn = 0  

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

# def gp_dilation(image):
#     return clip_image(dilation(image,footprint=selem))

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
pset.addEphemeralConstant("LENGTH",lambda: random.uniform(2.0, 28.0),FLOAT) # antes 2 - 28 -> 2 - 14
pset.addEphemeralConstant("THICKNESS",lambda: random.uniform(1.0,4.0), FLOAT) # antes 1 - 4 -> 1 - 4
pset.addEphemeralConstant("RADIUS",lambda: random.uniform(1.0,14.0),FLOAT) # antes 1 - 14 -> 1 - 7
pset.addEphemeralConstant("X",lambda: random.uniform(0.0,27.0),FLOAT) # antes 0 - 27 -> 0 - 15
pset.addEphemeralConstant("Y", lambda: random.uniform(0.0,27.0),FLOAT) # antes 0 - 27 -> 0 - 15
pset.addEphemeralConstant("WIDTH",lambda: random.uniform(2.0,28.0),FLOAT) # antes 2 - 28 -> 2 - 14
pset.addEphemeralConstant("HEIGHT",lambda: random.uniform(2.0,28.0),FLOAT) # antes 2 - 28 -> 2 - 14

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

# Creator
if not hasattr(creator, "FitnessMin"):
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))

if not hasattr(creator, "Individual"):
    creator.create("Individual", gp.PrimitiveTree,fitness=creator.FitnessMin)


gp_state = {
    "input_image": None,
    "target_image": None,
    "target_class": targetn
}

# MSE
def mse(image_a, image_b):
    image_a = np.asarray(image_a,dtype=np.float32)
    image_b = np.asarray(image_b,dtype=np.float32)
    return float(np.mean((image_a - image_b) ** 2))

# Clasificador
def classifier_probability(image,classifier,target_class=targetn,preprocess=None):
    """
    Obtiene P(clase objetivo | imagen)
    utilizando un clasificador compatible
    con predict_proba().
    """

    image = np.asarray(image, dtype=np.float32)

    if preprocess is not None:
        X = preprocess(image)
    else:
        X = image.reshape( 1, -1)

    probabilities = classifier.predict_proba(X)
    classes = classifier.classes_
    class_index = np.where(
        classes == target_class
    )[0]

    if len(class_index) == 0:

        raise ValueError(
            f"La clase {target_class} "
            "no existe en classifier.classes_."
        )

    return float(
        probabilities[
            0,
            class_index[0]
        ]
    )


# Terminales faltantes:
# Imagen de entrada
# pset.addTerminal(gp_state["input_image"], np.ndarray)                                # La imagen como input está dando problemas

# Constantes float para parámetros geométricos
float_constants = [0.0, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 12.0, 14.0, 16.0, 20.0, 24.0, 27.0]

for value in float_constants:
    pset.addTerminal(value, float)

# print("Primitivas:")
# for type_, primitives in pset.primitives.items():
#     print(type_, len(primitives))

# print("\nTerminales:")
# for type_, terminals in pset.terminals.items():
#     print(type_, len(terminals))


# Para solucionar el tema de que las terminales de tipo FLOAT no se generen correctamente, se implementa una función personalizada para generar individuos con tipos específicos.
def generate_typed(pset, min_, max_, type_=None):

    if type_ is None:
        type_ = pset.ret

    height = random.randint(min_, max_)
    expr = []
    stack = [(0, type_)]
    while stack:
        depth, current_type = stack.pop()
        # --------------------------------------------------
        # FLOAT: siempre debe ser un terminal
        # --------------------------------------------------
        if current_type is FLOAT:
            terminal = random.choice(
                pset.terminals[FLOAT]
            )
            if isinstance(terminal, gp.MetaEphemeral):
                terminal = terminal()
            expr.append(terminal)
            continue
        # --------------------------------------------------
        # Profundidad máxima: utilizar terminal
        # --------------------------------------------------
        if depth == height:
            terminal = random.choice(
                pset.terminals[current_type]
            )
            if isinstance(terminal, gp.MetaEphemeral):
                terminal = terminal()
            expr.append(terminal)
            continue
        # --------------------------------------------------
        # Antes de min_depth: utilizar primitiva
        # --------------------------------------------------
        if depth < min_:
            primitive = random.choice(
                pset.primitives[current_type]
            )
            expr.append(primitive)
            for arg_type in reversed(primitive.args):
                stack.append(
                    (depth + 1, arg_type)
                )
            continue
        # --------------------------------------------------
        # Entre min_depth y max_depth:
        # terminal o primitiva
        # --------------------------------------------------
        if random.random() < 0.5:
            terminal = random.choice(
                pset.terminals[current_type]
            )
            if isinstance(terminal, gp.MetaEphemeral):
                terminal = terminal()
            expr.append(terminal)
        else:
            primitive = random.choice(
                pset.primitives[current_type]
            )
            expr.append(primitive)
            for arg_type in reversed(primitive.args):
                stack.append(
                    (depth + 1, arg_type)
                )
    return expr

# Evaluación de individuos
gp_state["input_image"] 

def evaluate_individual(individual):
    input_image = gp_state["input_image"]
    target_image = gp_state["target_image"]
    func = toolbox.compile(expr=individual)
    generated_image = func(input_image)
    fitness = mse(generated_image, target_image)
    return (fitness,)


# TOOLBOX
toolbox = base.Toolbox()
#toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=1, max_=3) # cambiado por:
toolbox.register("expr", generate_typed, pset=pset, min_=1, max_=3)
toolbox.register("individual", tools.initIterate,creator.Individual, toolbox.expr)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("compile", gp.compile, pset=pset)
toolbox.register("evaluate", evaluate_individual)
toolbox.register("select", tools.selTournament, tournsize=3)
toolbox.register("mate", gp.cxOnePoint)
toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=8))
#toolbox.register("expr_mut", gp.genFull, pset=pset, min_=0, max_=2)
toolbox.register("expr_mut", generate_typed, pset=pset, min_=0, max_=2)
toolbox.register("mutate",gp.mutUniform, expr=toolbox.expr_mut, pset=pset)
toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=8))

stats = tools.Statistics(
    lambda individual:
        individual.fitness.values[0]
)

stats.register("min", np.min)
stats.register("avg", np.mean)
stats.register("max", np.max)

hall_of_fame = tools.HallOfFame(1)

target_class = targetn

target_images = X_train[y_train == target_class]

target_image = (target_images[0].astype(np.float32))

# Imagen aletoria
random_image = np.random.uniform( 0, 255, size=(28, 28)).astype(np.float32)

# Se configura el estado
gp_state["input_image"] = random_image
gp_state["target_image"] = target_image
gp_state["target_class"] = target_class

# Primeraa prueba
individual = toolbox.individual()

# print(individual)
# print("Altura:", individual.height)
# print("Nodos:", len(individual))

# fitness = toolbox.evaluate(individual)

# print("Fitness:", fitness)

# print("\nTipo de cada nodo:")

# for node in individual:
#     print(
#         node,
#         "->",
#         getattr(node, "ret", type(node))
#     )

# print("Ultimaa prueba")
for i in range(10):
    individual = toolbox.individual()
    # print(f"\nIndividuo {i+1}:")
    # print(individual)
    # print("Altura:", individual.height)
    # print("Nodos:", len(individual))
    fitness = toolbox.evaluate(individual)
    # print("Fitness:", fitness)


# print("\n=== TERMINALES IMAGE ===")

# for terminal in pset.terminals[IMAGE]:

#     print(
#         terminal.name,
#         type(terminal.value)
#     )

# Ejecutar el GP
population = toolbox.population(
    n=50
)

result_population, logbook = algorithms.eaSimple(
    population,
    toolbox,
    cxpb=0.5,
    mutpb=0.2,
    ngen=30,
    stats=stats,
    halloffame=hall_of_fame,
    verbose=True
)

best_individual = hall_of_fame[0]

print("\nMejor individuo:")
print(best_individual)
print("\nFitness:", best_individual.fitness.values[0])
best_function = toolbox.compile(expr=best_individual)
best_image = best_function(gp_state["input_image"])

print("\nImagen generada:")
print( "min =", best_image.min())
print( "max =", best_image.max())
print("mean =", best_image.mean())
print("MSE =", mse(best_image, gp_state["target_image"]))

# Visualizar resultado
fig, axes = plt.subplots( 1, 3, figsize=(10, 4))

axes[0].imshow(gp_state["input_image"],cmap="gray",vmin=0, vmax=255)
axes[0].set_title("Entrada")
axes[1].imshow(best_image, cmap="gray", vmin=0, vmax=255)
axes[1].set_title("Generada")
axes[2].imshow(gp_state["target_image"], cmap="gray", vmin=0, vmax=255)
axes[2].set_title("Objetivo")

for ax in axes:
    ax.axis("off")

plt.tight_layout()
plt.show()