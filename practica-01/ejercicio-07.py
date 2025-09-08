import random
import numpy as np
from deap import base, creator, tools, algorithms

# Definir el problema como un multiobjetivo (maximizar capacidad, minimizar costo)
creator.create("FitnessMulti", base.Fitness, weights=(1.0, -1.0))  # 1.0 para maximizar, -1.0 para minimizar
creator.create("Individual", list, fitness=creator.FitnessMulti)

# Parámetros del problema
n_nodos = 5  # Número de nodos
k_max = 4    # Máximo número de conexiones
# Matriz de capacidades (Mbps) y costos (ejemplo simulado)
capacidades = np.array([[0, 10, 15, 8, 12],
                        [10, 0, 20, 5, 9],
                        [15, 20, 0, 7, 11],
                        [8, 5, 7, 0, 13],
                        [12, 9, 11, 13, 0]])
costos = np.array([[0, 5, 8, 4, 6],
                   [5, 0, 10, 3, 7],
                   [8, 10, 0, 5, 9],
                   [4, 3, 5, 0, 8],
                   [6, 7, 9, 8, 0]])

# Función para evaluar un individuo (solución)
def eval_network(individual):
    selected_edges = [(i, j) for i in range(n_nodos) for j in range(i + 1, n_nodos) if individual[i][j]]
    num_edges = len(selected_edges)
    if num_edges > k_max or not is_connected(selected_edges):
        return 0, float('inf')  # Penalización si excede K o no está conectada
    capacidad_total = sum(capacidades[i][j] for i, j in selected_edges)
    costo_total = sum(costos[i][j] for i, j in selected_edges)
    return capacidad_total, costo_total

# Verificar si la red está conectada usando DFS
def is_connected(edges):
    if not edges:
        return False
    visited = [False] * n_nodos
    def dfs(node):
        visited[node] = True
        for i, j in edges:
            if i == node and not visited[j]:
                dfs(j)
            elif j == node and not visited[i]:
                dfs(i)
    dfs(0)
    return all(visited)

# Inicialización de la población (matriz de adyacencia simétrica)
def generate_individual():
    individual = [[0] * n_nodos for _ in range(n_nodos)]
    for i in range(n_nodos):
        for j in range(i + 1, n_nodos):
            individual[i][j] = random.randint(0, 1)
            individual[j][i] = individual[i][j]  # Simetría
    return individual

# Configurar el algoritmo genético
toolbox = base.Toolbox()
toolbox.register("individual", tools.initIterate, creator.Individual, generate_individual)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

# Evaluación personalizada
toolbox.register("evaluate", eval_network)

# Mutación personalizada para matrices de booleanos
def mutar_matriz(individual, indpb=0.05):
    for i in range(n_nodos):
        for j in range(i + 1, n_nodos):
            if random.random() < indpb:
                individual[i][j] = 1 - individual[i][j]
                individual[j][i] = individual[i][j]  # Mantener simetría
    return individual,

toolbox.register("mate", tools.cxUniform, indpb=0.5)
toolbox.register("mutate", mutar_matriz, indpb=0.05)
toolbox.register("select", tools.selNSGA2)

# Parámetros del MOGA
pop = toolbox.population(n=100)
hof = tools.ParetoFront()  # Frente de Pareto
stats = tools.Statistics(lambda ind: ind.fitness.values)
stats.register("min", np.min, axis=0)
stats.register("max", np.max, axis=0)

# Ejecutar el algoritmo
pop, log = algorithms.eaMuPlusLambda(pop, toolbox, mu=100, lambda_=200, cxpb=0.7, mutpb=0.3, ngen=50, stats=stats, halloffame=hof, verbose=True)

# Retornar el frente de Pareto
pareto_solutions = [(ind, ind.fitness.values) for ind in hof]
for sol, fitness in pareto_solutions:
    print(f"Solución: {sol}, Capacidad: {fitness[0]}, Costo: {fitness[1]}")