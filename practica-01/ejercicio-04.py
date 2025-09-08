import random
from deap import base, creator, tools, algorithms
import math

# Definición de la clase para minimizar la distancia total
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin)

# Coordenadas de las ciudades (ejemplo: 5 ciudades)
ciudades = [(0, 0), (1, 2), (2, 1), (3, 3), (4, 0)]

# Función de evaluación: calcula la distancia total de la ruta
def eval_tsp(individual):
    distancia_total = 0
    for i in range(len(individual)):
        ciudad1 = ciudades[individual[i]]
        ciudad2 = ciudades[individual[(i + 1) % len(individual)]]
        distancia_total += math.sqrt((ciudad2[0] - ciudad1[0]) ** 2 + (ciudad2[1] - ciudad1[1]) ** 2)
    return distancia_total,

# Configuración del toolbox
def configurar_toolbox(num_ciudades):
    toolbox = base.Toolbox()
    toolbox.register("indices", random.sample, range(num_ciudades), num_ciudades)
    toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.indices)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("evaluate", eval_tsp)
    toolbox.register("mate", tools.cxPartialyMatched)
    toolbox.register("mutate", tools.mutShuffleIndexes, indpb=0.05)
    toolbox.register("select", tools.selTournament, tournsize=3)
    return toolbox

# Algoritmo Genético
def algoritmo_genetico_tsp(num_ciudades=5, tam_poblacion=50, generaciones=100, prob_cruza=0.7, prob_mut=0.2):
    random.seed(42)
    toolbox = configurar_toolbox(num_ciudades)
    
    poblacion = toolbox.population(n=tam_poblacion)
    hof = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", lambda x: sum(f[0] for f in x) / len(x))
    stats.register("min", lambda x: min(f[0] for f in x))

    poblacion, log = algorithms.eaSimple(poblacion, toolbox, cxpb=prob_cruza, mutpb=prob_mut,
                                        ngen=generaciones, stats=stats, halloffame=hof, verbose=True)
    
    mejor = hof[0]
    return mejor, log

if __name__ == "__main__":
    mejor_ruta, log = algoritmo_genetico_tsp()
    print(f"Mejor ruta encontrada: {mejor_ruta}, Distancia total: {mejor_ruta.fitness.values[0]}")