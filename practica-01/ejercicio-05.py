import random
from deap import base, creator, tools, algorithms

# Definición de la clase para minimizar conflictos
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin)

# Función de evaluación: cuenta los conflictos entre reinas
def eval_n_queens(individual):
    n = len(individual)
    conflictos = 0
    for i in range(n):
        for j in range(i + 1, n):
            # Verificar diagonales y filas
            if abs(i - j) == abs(individual[i] - individual[j]):
                conflictos += 1
    return conflictos,

# Configuración del toolbox
def configurar_toolbox(n):
    toolbox = base.Toolbox()
    toolbox.register("attr_int", random.randint, 0, n - 1)
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_int, n=n)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("evaluate", eval_n_queens)
    toolbox.register("mate", tools.cxPartialyMatched)
    toolbox.register("mutate", tools.mutUniformInt, low=0, up=n - 1, indpb=0.05)
    toolbox.register("select", tools.selTournament, tournsize=3)
    return toolbox

# Algoritmo Genético
def algoritmo_genetico_n_queens(n=8, tam_poblacion=50, generaciones=100, prob_cruza=0.7, prob_mut=0.2):
    random.seed(42)
    toolbox = configurar_toolbox(n)
    
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
    n = 8  # Tamaño del tablero (8x8 por defecto)
    mejor_solucion, log = algoritmo_genetico_n_queens(n)
    print(f"Mejor solución encontrada: {mejor_solucion}, Conflictos: {mejor_solucion.fitness.values[0]}")