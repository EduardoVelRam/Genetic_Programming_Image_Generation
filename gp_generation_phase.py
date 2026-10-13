from geometric_images import (
    clip_image, create_point, create_horizontal_line, create_vertical_line, create_circle, 
    create_diagonal_line, create_rectangle, create_square, IMAGE, FLOAT)
from import_data import load_mnist, mnist_path
from geometric_primitives_gp import gp_circle, gp_line, gp_rectangle
from aditional_functions import preprocess_for_knn, classifier_probability
from trigonometric_expressions import (
    field_to_image, gp_sin, gp_cos, gp_tan, X_GRID, Y_GRID, Z_GRID, FIELD, field_sin, 
    field_cos, field_tan, field_add, field_subtract, field_multiply, 
    field_scale, field_add_constant, field_divide)
import operator
import random
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import convolve
from skimage.morphology import (
    erosion,
    dilation,
    opening,
    closing,
    disk
)
from deap import base, creator, gp, tools, algorithms
from pathlib import Path
import joblib 

class FieldType:
    pass
class ImageType:
    pass

FIELD = FieldType
IMAGE = ImageType

DEBUG_DIR = Path("debug_fitness_zero")
DEBUG_DIR.mkdir(parents=True, exist_ok=True)

# Importar el modelo KNN entrenado
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "knn_mnist.joblib"

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"No se encontró el modelo KNN: {MODEL_PATH}. "
        "Ejecutar primero train_knn.py."
    )

knn1 = joblib.load(MODEL_PATH)

print("Clasificador KNN cargado.")

X_train, y_train, X_test, y_test = load_mnist(mnist_path)

print("gp_generation_phase")
target_class = 9    
generations = 170
target_images = X_train[y_train == target_class]

target_image = (target_images[4].astype(np.float32))


def inspect_individual(individual, image, probability, target_class):
    image = np.asarray(image, dtype=np.float32)

    X = preprocess_for_knn(image)
    predicted_class = knn1.predict(X)[0]

    print("\n=== INDIVIDUO CON FITNESS CERO ===")
    #print("Árbol:", individual)
    #print("Clase objetivo:", target_class)
    #print("Clase predicha:", predicted_class)
    #print("Probabilidad objetivo:", probability)
    #print("Mínimo:", image.min())
    print("Máximo:", image.max())
    print("Media:", image.mean())
    #print("Desviación estándar:", image.std())

    probabilities = knn1.predict_proba(X)[0]
    print("Probabilidades por clase:", probabilities)

    plt.figure(figsize=(4, 4))
    plt.imshow(image, cmap="gray", vmin=0, vmax=255)
    plt.title(
        f"Objetivo: {target_class} | "
        f"Predicción: {predicted_class} | "
        f"P: {probability:.3f}"
    )
    plt.axis("off")

    filename = DEBUG_DIR / (
        f"fitness_zero_{abs(hash(str(individual)))}.png"
    )

    plt.savefig(filename, bbox_inches="tight", dpi=150)
    plt.close()

    print("Imagen guardada en:", filename)

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
# geometric_terminals = {
#     "POINT": POINT,
#     "LINE_H": LINE_H,
#     "LINE_V": LINE_V,
#     "LINE_D1": LINE_D1,
#     "LINE_D2": LINE_D2,
#     "CIRCLE": CIRCLE,
#     "SQUARE": SQUARE,
#     "RECTANGLE": RECTANGLE
# }
# fig, axes = plt.subplots( 2, 4, figsize=(10, 5))
# for ax, (name, image) in zip(axes.ravel(), geometric_terminals.items()):
#     ax.imshow(image, cmap="gray", vmin=0, vmax=255)
#     ax.set_title(name)
#     ax.axis("off")
# plt.tight_layout()
# plt.show()

# OPERADORES MORFOLÓGICOS
selem = disk(1)
average_kernel = (np.ones((3, 3), dtype=np.float32) / 9.0)

def gp_erosion(image):
    return clip_image(erosion(image, footprint=selem))

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
    # debug_image("gp_dilation - entrada", image)
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
    result = np.multiply(image_a, image_b)
    return clip_image(result)

# pset = gp.PrimitiveSetTyped(
#     "MAIN",
#     [IMAGE],
#     IMAGE
# )

# Cambiado para que la imagen de entrada sea un argumento en lugar de una terminal 
pset = gp.PrimitiveSetTyped(
    "MAIN",
    [],
    IMAGE
)

# pset.renameArguments(ARG0="I")

# Registrar las terminales geométricas
pset.addTerminal(POINT, IMAGE, name="Point")
pset.addTerminal(LINE_H,IMAGE, name="Line_H")
pset.addTerminal(LINE_V, IMAGE, name="Line_V")
pset.addTerminal(LINE_D1, IMAGE, name="Line_D1")
pset.addTerminal(LINE_D2, IMAGE, name="Line_D2")
pset.addTerminal(CIRCLE, IMAGE, name="Circle")
pset.addTerminal(SQUARE, IMAGE, name="Square")
pset.addTerminal(RECTANGLE,IMAGE, name="Rectangle")
pset.addTerminal(X_GRID, FIELD, name="X_val")
pset.addTerminal(Y_GRID, FIELD, name="Y_val")
pset.addTerminal(Z_GRID, FIELD, name="Z_val")

# Prmitivas geométricas paramétricas
pset.addPrimitive(gp_circle, [FLOAT, FLOAT, FLOAT], IMAGE, name="Circle_Param")
pset.addPrimitive(gp_line, [FLOAT, FLOAT, FLOAT], IMAGE, name="Line_Param")
pset.addPrimitive(gp_rectangle, [FLOAT, FLOAT, FLOAT, FLOAT], IMAGE, name="Rectangle_Param")

# Primitivas trigonométricas y de campo
pset.addPrimitive(field_sin, [FIELD], FIELD, name="Sin")
pset.addPrimitive(field_cos, [FIELD], FIELD, name="Cos")
pset.addPrimitive(field_tan, [FIELD], FIELD, name="Tan")
pset.addPrimitive(np.cos, [FIELD], FIELD, name="Coseno")
pset.addPrimitive(np.sin, [FIELD], FIELD, name="Seno")
pset.addPrimitive(field_add, [FIELD, FIELD], FIELD, name="Add")
pset.addPrimitive(field_subtract, [FIELD, FIELD], FIELD, name="Subtract")
pset.addPrimitive(field_multiply, [FIELD, FIELD], FIELD, name="Multiply")
pset.addPrimitive(field_divide, [FIELD, FIELD], FIELD, name="F_Divide")
pset.addPrimitive(field_to_image, [FIELD], IMAGE, name="F_TO_IMG")
pset.addPrimitive(field_scale, [FIELD, FLOAT], FIELD, name="Scale")
pset.addPrimitive(field_add_constant, [FIELD, FLOAT], FIELD, name="OFFSET")

# PARÁMETROS EVOLUTIVOS
pset.addEphemeralConstant("ANGLE", lambda: random.uniform(-np.pi, np.pi), FLOAT)
pset.addEphemeralConstant("LENGTH",lambda: random.uniform(2.0, 28.0),FLOAT) # antes 2 - 28 -> 2 - 14
pset.addEphemeralConstant("THICKNESS",lambda: random.uniform(1.0,4.0), FLOAT) # antes 1 - 4 -> 1 - 4
pset.addEphemeralConstant("RADIUS",lambda: random.uniform(1.0,14.0),FLOAT) # antes 1 - 14 -> 1 - 7
pset.addEphemeralConstant("X_coord",lambda: random.uniform(0.0,27.0),FLOAT) # antes 0 - 27 -> 0 - 15
pset.addEphemeralConstant("Y_coord", lambda: random.uniform(0.0,27.0),FLOAT) # antes 0 - 27 -> 0 - 15
pset.addEphemeralConstant("WIDTH",lambda: random.uniform(2.0,28.0),FLOAT) # antes 2 - 28 -> 2 - 14
pset.addEphemeralConstant("HEIGHT",lambda: random.uniform(2.0,28.0),FLOAT) # antes 2 - 28 -> 2 - 14
pset.addEphemeralConstant("CONST", lambda: random.uniform(-5.0, 5.0), FLOAT)

# Operadores de la imagen
pset.addPrimitive(gp_erosion, [IMAGE], IMAGE, name="gp_erosion")
pset.addPrimitive(gp_dilation,[IMAGE], IMAGE, name="gp_dilation")
pset.addPrimitive(gp_opening, [IMAGE], IMAGE, name="gp_opening")
pset.addPrimitive(gp_closing, [IMAGE], IMAGE, name="gp_closing")
pset.addPrimitive(gp_convolution,[IMAGE], IMAGE, name="gp_convolution")
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
    #"input_image": None,
    "target_image": None,
    "target_class": target_class
}

# MSE
def mse(image_a, image_b):
    image_a = np.asarray(image_a,dtype=np.float32)
    image_b = np.asarray(image_b,dtype=np.float32)
    return float(np.mean((image_a - image_b) ** 2))


# Terminales faltantes:
# Imagen de entrada
# pset.addTerminal(gp_state["input_image"], np.ndarray)                                # La imagen como input está dando problemas

# # Constantes float para parámetros geométricos
float_constants = [-4.0, -3.0, -2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 7.0, 10.0, 12.0, 14.0, 16.0, 20.0, 24.0, 27.0, np.pi, 2*np.pi]

for value in float_constants:
    pset.addTerminal(value, FLOAT)


# Para solucionar el tema de que las terminales de tipo FLOAT no se generen correctamente, se implementa una función personalizada para generar individuos con tipos específicos.
def generate_typed(pset, min_, max_, type_=None):
    if type_ is None:
        type_ = pset.ret
    height = random.randint(min_, max_)
    expr = []
    stack = [(0, type_)]
    while stack:
        depth, current_type = stack.pop()
        if current_type is FLOAT:
            terminal = random.choice(pset.terminals[FLOAT])
            if isinstance(terminal, gp.MetaEphemeral):
                terminal = terminal()
            expr.append(terminal)
            continue
        if depth == height:
            terminal = random.choice(pset.terminals[current_type])
            if isinstance(terminal, gp.MetaEphemeral):
                terminal = terminal()
            expr.append(terminal)
            continue
        if depth < min_:
            primitive = random.choice(pset.primitives[current_type])
            expr.append(primitive)
            for arg_type in reversed(primitive.args):
                stack.append((depth + 1, arg_type))
            continue
        if random.random() < 0.5:
            terminal = random.choice(pset.terminals[current_type])
            if isinstance(terminal, gp.MetaEphemeral):
                terminal = terminal()
            expr.append(terminal)
        else:
            primitive = random.choice(pset.primitives[current_type])
            expr.append(primitive)
            for arg_type in reversed(primitive.args):
                stack.append((depth + 1, arg_type))
    return expr

# Evaluación de individuos
# gp_state["input_image"] 

# Antes con mse
def evaluate_individual(individual):
     target_image = gp_state["target_image"]
     # func = toolbox.compile(expr=expr)
     func = toolbox.compile(expr=individual)
     generated_image = func
     fitness = mse(generated_image, target_image) # puede agregarse un criterio de paro en caso de que el fitness no mejore después de un número determinado de generaciones
     return (fitness,)

# def evaluate_individual(individual):
#     target_class = gp_state["target_class"]
#     generated_image = toolbox.compile(expr=individual)
#     probability = classifier_probability(generated_image, classifier=knn1, target_class=target_class) 
#     fitness = 1.0 - probability
#     return (fitness,)

def evaluate_individual2(individual):
    target_class = gp_state["target_class"]
    generated_image = toolbox.compile(expr=individual)
    probability = classifier_probability(generated_image, classifier=knn1, target_class=target_class)
    fitness = 1.0 - probability
    if np.isclose(fitness, 0.0):
        inspect_individual(individual=individual, image=generated_image, probability=probability, target_class=target_class)
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


ELITE_SIZE = 2

stats = tools.Statistics(
    lambda individual:
        individual.fitness.values[0]
)

stats.register("min", np.min)
stats.register("avg", np.mean)
stats.register("max", np.max)

hall_of_fame = tools.HallOfFame(1)


# Imagen aletoria
# np.random.seed(RANDOM_SEED)
# random_image = np.random.uniform( 0, 255, size=(28, 28)).astype(np.float32)

# Se configura el estado
# gp_state["input_image"] = random_image
gp_state["target_image"] = target_image
gp_state["target_class"] = target_class



# Ejecutar el GP
population = toolbox.population(
    n=50
)


logbook = tools.Logbook()
logbook.header = [
    "gen",
    "nevals",
    "min",
    "avg",
    "max"
]

# Evaluación inicial
for individual in population:
    individual.fitness.values = toolbox.evaluate(individual)

hall_of_fame.update(population)

record = stats.compile(population)

logbook.record(
    gen=0,
    nevals=len(population),
    **record
)

print(logbook.stream)

cxpb = 0.5
mutpb = 0.6
# EVOLUCIÓN
for generation in range(1, generations+1):
    # 1. Selección
    offspring = toolbox.select(population,len(population) - ELITE_SIZE)
    offspring = algorithms.varAnd(offspring, toolbox, cxpb, mutpb)
    offspring = list(map(toolbox.clone, offspring))

    # 2. Crossover
    for child1, child2 in zip(offspring[::2], offspring[1::2]):
        if random.random() < 0.5:
            toolbox.mate(child1, child2)
            del child1.fitness.values
            del child2.fitness.values

    # 3. Mutación
    for mutant in offspring:
        if random.random() < 0.2:
            toolbox.mutate(mutant)
            del mutant.fitness.values

    # 4. Evaluación
    invalid_individuals = [
        individual
        for individual in offspring
        if not individual.fitness.valid
    ]

    for individual in invalid_individuals:
        individual.fitness.values = toolbox.evaluate(individual)

    # 5. Elitismo
    elites = tools.selBest(population, ELITE_SIZE)
    elites = list(map(toolbox.clone, elites))

    # 6. Nueva generación
    population = offspring + elites

    # 7. Hall of Fame
    hall_of_fame.update(population)

    # 8. Estadísticas
    record = stats.compile(population)

    logbook.record(
        gen=generation,
        nevals=len(invalid_individuals),
        **record
    )

    print(logbook.stream)


# result_population, logbook = algorithms.eaSimple(
#     population,
#     toolbox,
#     cxpb=0.5,
#     mutpb=0.6,
#     ngen=80,
#     stats=stats,
#     halloffame=hall_of_fame,
#     verbose=True
# )

best_individual = hall_of_fame[0]

print("\nMejor individuo:")
print(best_individual)
print("\nFitness:", best_individual.fitness.values[0])
best_function = toolbox.compile(expr=best_individual)
best_image = best_function #(gp_state["input_image"])


# print("\nImagen generada:")
# print( "min =", best_image.min())
# print( "max =", best_image.max())
# print("mean =", best_image.mean())
print("MSE =", mse(best_image, gp_state["target_image"]))

#black_image = np.zeros((28, 28), dtype=np.float32)
#X_black = preprocess_for_knn(black_image)
#print("Predicción:", knn1.predict(X_black)[0])
#print("Probabilidades:", knn1.predict_proba(X_black)[0])
#print("Clases:", knn1.classes_)

# Visualizar resultado
fig, axes = plt.subplots( 1, 3, figsize=(10, 4))

axes[0].imshow(best_image,cmap="gray",vmin=0, vmax=255)
axes[0].set_title("Generada")

image = np.where(best_image > best_image.mean(), 255, 0).astype(np.float32)
axes[1].imshow(image, cmap="gray", vmin=0, vmax=255)
axes[1].set_title("Normalizada")

imaget = np.where(gp_state["target_image"] > gp_state["target_image"].mean(), 255, 0).astype(np.float32)
axes[2].imshow(imaget, cmap="gray", vmin=0, vmax=255)
# axes[2].imshow(gp_state["target_image"], cmap="gray", vmin=0, vmax=255)
axes[2].set_title("Objetivo")

for ax in axes:
    ax.axis("off")

plt.tight_layout()
plt.show()