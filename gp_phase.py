import operator
import random
from primitives_operations import gp_add, gp_closing, gp_convolution, gp_dilation, gp_erosion, gp_multiply, gp_opening, gp_subtract
from primitives_operations import target_image, target_class, target_images, mse, random_image
from deap import base, creator, gp, tools, algorithms
import numpy as np
import matplotlib.pyplot as plt

print("Fase de GP")
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

# El 1 significa que se tendrá un argumento de entrada
pset = gp.PrimitiveSet("MAIN", 1)

# Operadores
pset.addPrimitive(gp_erosion, 1)
pset.addPrimitive(gp_dilation, 1)
pset.addPrimitive(gp_opening, 1)
pset.addPrimitive(gp_closing, 1)
pset.addPrimitive(gp_convolution, 1)
pset.addPrimitive(gp_add, 2)
pset.addPrimitive(gp_subtract, 2)
pset.addPrimitive(gp_multiply, 2)

# Paaara identifiar que el argumento es una imagen (I)
pset.renameArguments(ARG0="I")

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

# print(individual)

func = toolbox.compile(expr=individual)

result = func(random_image)

# print(result.shape)
# print(result.min())
# print(result.max())

def evaluate_individual(individual):
    func = toolbox.compile(expr=individual)

    generated_image = func(random_image)

    fitness = mse(
        generated_image,
        target_image
    )

    return (fitness,)

# Registrar la evalauación
toolbox.register(
    "evaluate",
    evaluate_individual
)

# Probar un individuo
individual = toolbox.individual()

# print("Árbol:")
# print(individual)

fitness = toolbox.evaluate(individual)

# print("Fitness:", fitness)


# Definición de Operadores Evolutivos
# Selección
toolbox.register(
    "select",
    tools.selTournament,
    tournsize=3
)

# Crossover
toolbox.register(
    "mate",
    gp.cxOnePoint
)

# Mutación
toolbox.register(
    "expr_mut",
    gp.genFull,
    min_=0, max_=2
)

toolbox.register(
    "mutate",
    gp.mutUniform,
    expr=toolbox.expr_mut, pset=pset
)

# Límites de profundidad
toolbox.decorate(
    "mate",
    gp.staticLimit(
        key=operator.attrgetter("height"),
        max_value=8
    )
)

toolbox.decorate(
    "mutate",
    gp.staticLimit(
        key=operator.attrgetter("height"),
        max_value=8
    )
)

# Primera evolución
population = toolbox.population(n=50)

n_gen = 30   # generaciones
CXPB = 0.8  # crossover
MUTPB = 0.2 # mutación
result_population, logbook = algorithms.eaSimple(
    population,
    toolbox,
    cxpb=CXPB,
    mutpb=MUTPB,
    ngen=n_gen,
    verbose=True
)

# Obtener el mejor individuo
best_individual = tools.selBest(
    result_population,
    k=1
)[0]

print("Mejor individuo:")
print(best_individual)

print("Fitness:")
print(best_individual.fitness.values)

# Generar imagen del mejor individuo
best_function = toolbox.compile(
    expr=best_individual
)

generated_image = best_function(
    random_image
)

# Comparar
fig, axes = plt.subplots(
    1, 3,
    figsize=(8, 3)
)

axes[0].imshow(
    random_image,
    cmap="gray",
    vmin=0,
    vmax=255
)
axes[0].set_title("Entrada")
axes[0].axis("off")

axes[1].imshow(
    target_image,
    cmap="gray",
    vmin=0,
    vmax=255
)
axes[1].set_title("Objetivo")
axes[1].axis("off")

axes[2].imshow(
    generated_image,
    cmap="gray",
    vmin=0,
    vmax=255
)
axes[2].set_title("Generada")
axes[2].axis("off")

plt.tight_layout()
plt.show()

# Regsitrar evolución del fitness
stats = tools.Statistics(
    lambda ind: ind.fitness.values[0]
)

stats.register("min", np.min)
stats.register("avg", np.mean)
stats.register("max", np.max)

hall_of_fame = tools.HallOfFame(1)


population = toolbox.population(n=50)

result_population, logbook = algorithms.eaSimple(
    population,
    toolbox,
    cxpb=CXPB,
    mutpb=MUTPB,
    ngen=n_gen,
    stats=stats,
    halloffame=hall_of_fame,
    verbose=True
)

best_individual = hall_of_fame[0]

print("Mejor individuo:")
print(best_individual)

print("Fitness:")
print(best_individual.fitness.values)

# Graficar fitness
generations = logbook.select("gen")
min_fitness = logbook.select("min")
avg_fitness = logbook.select("avg")

plt.figure(figsize=(8, 4))

plt.plot(
    generations,
    min_fitness,
    label="Mejor fitness"
)

plt.plot(
    generations,
    avg_fitness,
    label="Fitness promedio"
)

plt.xlabel("Generación")
plt.ylabel("MSE")
plt.title("Evolución del fitness")
plt.legend()
plt.grid()

plt.show()