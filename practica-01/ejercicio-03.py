import random
from deap import base, creator, tools, algorithms
import math

# Definición de la clase para minimizar la distancia euclidiana
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin)

# Función de evaluación: calcula la distancia euclidiana RGB
def eval_rgb_distance(individual, target_rgb):
    r, g, b = individual
    tr, tg, tb = target_rgb
    distance = math.sqrt((r - tr) ** 2 + (g - tg) ** 2 + (b - tb) ** 2)
    return distance,

# Configuración del toolbox
def configurar_toolbox(target_rgb):
    toolbox = base.Toolbox()
    toolbox.register("attr_int", random.randint, 0, 255)
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_int, n=3)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("evaluate", eval_rgb_distance, target_rgb=target_rgb)
    toolbox.register("mate", tools.cxBlend, alpha=0.5)
    toolbox.register("mutate", tools.mutGaussian, mu=128, sigma=50, indpb=0.3)
    toolbox.register("select", tools.selTournament, tournsize=3)
    return toolbox

# Algoritmo Genético
def algoritmo_genetico_rgb(target_rgb=(255, 128, 0), tam_poblacion=50, generaciones=100, prob_cruza=0.7, prob_mut=0.2):
    random.seed(42)
    toolbox = configurar_toolbox(target_rgb)
    
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
    target = (255, 128, 0)  # Color objetivo: naranja
    mejor_color, log = algoritmo_genetico_rgb(target)
    print(f"Mejor color encontrado: RGB{mejor_color}, Distancia: {mejor_color.fitness.values[0]}")