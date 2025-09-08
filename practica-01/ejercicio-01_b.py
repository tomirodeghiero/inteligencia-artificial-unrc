import random
from deap import base, creator, tools, algorithms

# Definición del problema de optimización
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

# Toolbox
toolbox = base.Toolbox()
tam_individuo = 20

# Registro de funciones de generación
toolbox.register("attr_bool", random.randint, 0, 1)
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_bool, tam_individuo)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

# Función de aptitud
def evalOneMax(individuo):
    return sum(individuo),

toolbox.register("evaluate", evalOneMax)
toolbox.register("mate", tools.cxTwoPoint)  # Cambiado a two-point como en el PDF
toolbox.register("mutate", tools.mutFlipBit, indpb=0.01)
toolbox.register("select", tools.selTournament, tournsize=3)

# Algoritmo con DEAP mejorado
def ejecutar_deap(tam_poblacion=50, generaciones=100, prob_cruza=0.7, prob_mut=0.2):
    random.seed(42)  # Para reproducibilidad
    poblacion = toolbox.population(n=tam_poblacion)
    hof = tools.HallOfFame(1)  # Hall of Fame para preservar el mejor (elitismo)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", lambda x: sum(f[0] for f in x)/len(x))
    stats.register("max", lambda x: max(f[0] for f in x))

    poblacion, log = algorithms.eaSimple(poblacion, toolbox, cxpb=prob_cruza, mutpb=prob_mut,
                                         ngen=generaciones, stats=stats, halloffame=hof, verbose=True)
    return hof[0]

if __name__ == "__main__":
    mejor = ejecutar_deap()
    print("Mejor individuo DEAP:", mejor)
    print("Fitness:", sum(mejor))