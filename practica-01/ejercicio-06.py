import random
from deap import base, creator, tools, algorithms
import math

# Definición de clases para minimizar distancia y tiempo
creator.create("FitnessMulti", base.Fitness, weights=(-1.0, -1.0))
creator.create("Individual", list, fitness=creator.FitnessMulti)

# Coordenadas de las ciudades (ejemplo: 5 ciudades)
ciudades = [(0, 0), (1, 2), (2, 1), (3, 3), (4, 0)]
# Tiempos asociados (ejemplo: matriz de tiempos por tráfico)
tiempos = [
    [0, 2, 1.5, 3, 2.5],
    [2, 0, 1, 2, 3],
    [1.5, 1, 0, 2.5, 2],
    [3, 2, 2.5, 0, 1.5],
    [2.5, 3, 2, 1.5, 0]
]

# Función de evaluación: distancia total y tiempo total
def eval_tsp_multi(individual):
    n = len(individual)
    distancia_total = 0
    tiempo_total = 0
    for i in range(n):
        ciudad1 = ciudades[individual[i]]
        ciudad2 = ciudades[individual[(i + 1) % n]]
        distancia_total += math.sqrt((ciudad2[0] - ciudad1[0]) ** 2 + (ciudad2[1] - ciudad1[1]) ** 2)
        tiempo_total += tiempos[individual[i]][individual[(i + 1) % n]]
    return distancia_total, tiempo_total

# Configuración del toolbox
def configurar_toolbox(num_ciudades):
    toolbox = base.Toolbox()
    toolbox.register("indices", random.sample, range(num_ciudades), num_ciudades)
    toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.indices)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("evaluate", eval_tsp_multi)
    toolbox.register("mate", tools.cxPartialyMatched)
    toolbox.register("mutate", tools.mutShuffleIndexes, indpb=0.05)
    toolbox.register("select", tools.selTournament, tournsize=3)
    return toolbox

# Algoritmo MOGA (NSGA-II)
def algoritmo_moga_tsp(num_ciudades=5, tam_poblacion=50, generaciones=100, prob_cruza=0.7, prob_mut=0.2):
    random.seed(42)
    toolbox = configurar_toolbox(num_ciudades)
    
    poblacion = toolbox.population(n=tam_poblacion)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", lambda x: [sum(f[i] for f in x) / len(x) for i in range(2)], names=("avg_dist", "avg_time"))
    stats.register("min", lambda x: min(f[0] for f in x), names="min_dist")

    poblacion, logbook = algorithms.eaMuPlusLambda(poblacion, toolbox, mu=tam_poblacion, lambda_=tam_poblacion,
                                                 cxpb=prob_cruza, mutpb=prob_mut, ngen=generaciones,
                                                 stats=stats, verbose=True)
    
    return tools.ParetoFront(), logbook

if __name__ == "__main__":
    frente_pareto, log = algoritmo_moga_tsp()
    print("Frente de Pareto:")
    for ind in frente_pareto:
        print(f"Ruta: {ind}, Distancia: {ind.fitness.values[0]}, Tiempo: {ind.fitness.values[1]}")