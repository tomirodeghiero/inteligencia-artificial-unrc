import random

# --- Parte a: Implementación propia de Algoritmo Genético ---

# Función de fitness: minimiza la diferencia absoluta entre el valor binario y el target
# Retorna negativo para maximizar (ya que buscamos diferencia 0)
def fitness(individuo, target):
    valor = int(''.join(map(str, individuo)), 2)  # Convierte binario a decimal
    return -abs(valor - target)

# Genera un individuo aleatorio (cadena binaria)
def generar_individuo(longitud):
    return [random.randint(0, 1) for _ in range(longitud)]

# Genera una población inicial
def generar_poblacion(tam_poblacion, longitud):
    return [generar_individuo(longitud) for _ in range(tam_poblacion)]

# Selección por torneo: elige el mejor de k candidatos
def seleccion(poblacion, k=3, target=None):
    seleccionados = random.sample(poblacion, k)
    return max(seleccionados, key=lambda ind: fitness(ind, target))

# Cruce de un punto con probabilidad
def cruza(padre1, padre2, prob_cruza=0.8):
    if random.random() < prob_cruza:
        punto = random.randint(1, len(padre1)-1)
        hijo1 = padre1[:punto] + padre2[punto:]
        hijo2 = padre2[:punto] + padre1[punto:]
        return hijo1, hijo2
    return padre1[:], padre2[:]

# Mutación: flip de bits con probabilidad
def mutacion(individuo, prob):
    return [1 - gen if random.random() < prob else gen for gen in individuo]

# Algoritmo Genético propio
def algoritmo_genetico_propio(target=42, tam_individuo=8, tam_poblacion=50, generaciones=100, prob_mut=0.01, prob_cruza=0.8, elitismo=1):
    random.seed(42)  # Para reproducibilidad
    poblacion = generar_poblacion(tam_poblacion, tam_individuo)
    
    for gen in range(generaciones):
        # Ordenar por fitness (mejor primero)
        poblacion.sort(key=lambda ind: fitness(ind, target), reverse=True)
        mejor_actual = poblacion[0]
        avg_fitness = sum(fitness(ind, target) for ind in poblacion) / tam_poblacion
        print(f"Generación {gen}: Mejor fitness = {fitness(mejor_actual, target)}, Fitness promedio = {avg_fitness:.2f}")
        
        # Elitismo: copiar los mejores
        nueva_poblacion = poblacion[:elitismo]
        
        # Generar descendientes
        while len(nueva_poblacion) < tam_poblacion:
            padre1 = seleccion(poblacion, target=target)
            padre2 = seleccion(poblacion, target=target)
            hijo1, hijo2 = cruza(padre1, padre2, prob_cruza)
            hijo1 = mutacion(hijo1, prob_mut)
            hijo2 = mutacion(hijo2, prob_mut)
            nueva_poblacion.extend([hijo1, hijo2])
        
        # Reemplazar población
        poblacion = nueva_poblacion[:tam_poblacion]
    
    # Resultado final
    poblacion.sort(key=lambda ind: fitness(ind, target), reverse=True)
    mejor = poblacion[0]
    valor = int(''.join(map(str, mejor)), 2)
    print(f"\nMejor individuo encontrado (propio): {mejor}")
    print(f"Valor decimal: {valor}")
    print(f"Fitness: {fitness(mejor, target)} (0 es perfecto)")
    return mejor

# --- Parte b: Implementación con DEAP (requiere 'pip install deap') ---

try:
    from deap import base, creator, tools, algorithms
except ImportError:
    print("DEAP no está instalado. Instalalo con 'pip install deap' para usar esta parte.")
else:
    # Definición para maximizar (negativo de diferencia)
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)

    def eval_int_rep(individuo, target):
        valor = int(''.join(map(str, individuo)), 2)
        return -abs(valor - target),  # Tupla para DEAP

    def algoritmo_genetico_deap(target=42, tam_individuo=8, tam_poblacion=50, generaciones=100, prob_mut=0.01, prob_cruza=0.8):
        random.seed(42)
        toolbox = base.Toolbox()
        toolbox.register("attr_bool", random.randint, 0, 1)
        toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_bool, tam_individuo)
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)
        toolbox.register("evaluate", eval_int_rep, target=target)
        toolbox.register("mate", tools.cxTwoPoint)  # Cruce de dos puntos
        toolbox.register("mutate", tools.mutFlipBit, indpb=prob_mut)
        toolbox.register("select", tools.selTournament, tournsize=3)

        poblacion = toolbox.population(n=tam_poblacion)
        hof = tools.HallOfFame(1)  # Preserva el mejor
        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("avg", lambda x: sum(f[0] for f in x)/len(x))
        stats.register("max", lambda x: max(f[0] for f in x))

        algorithms.eaSimple(poblacion, toolbox, cxpb=prob_cruza, mutpb=prob_mut, ngen=generaciones, stats=stats, halloffame=hof, verbose=True)
        
        mejor = hof[0]
        valor = int(''.join(map(str, mejor)), 2)
        print(f"\nMejor individuo encontrado (DEAP): {mejor}")
        print(f"Valor decimal: {valor}")
        print(f"Fitness: {mejor.fitness.values[0]} (0 es perfecto)")
        return mejor

# --- Parte c: Ejecuciones con diferentes hiperparámetros ---

if __name__ == "__main__":
    print("\n--- Ejecución base (propio): Target=42, Ind=8, Pop=50, Gen=100, Mut=0.01, Cruza=0.8 ---")
    algoritmo_genetico_propio()

    print("\n--- Variación 1 (propio): Target=100, Ind=10, Pop=100, Gen=50, Mut=0.05, Cruza=0.7 ---")
    algoritmo_genetico_propio(target=100, tam_individuo=10, tam_poblacion=100, generaciones=50, prob_mut=0.05, prob_cruza=0.7)

    print("\n--- Variación 2 (propio): Target=255, Ind=8, Pop=20, Gen=200, Mut=0.1, Cruza=0.9 ---")
    algoritmo_genetico_propio(target=255, tam_individuo=8, tam_poblacion=20, generaciones=200, prob_mut=0.1, prob_cruza=0.9)

    # Para DEAP (ejecuta si está instalado)
    try:
        print("\n--- Ejecución base (DEAP): Target=42, Ind=8, Pop=50, Gen=100, Mut=0.01, Cruza=0.8 ---")
        algoritmo_genetico_deap()
        
        print("\n--- Variación 1 (DEAP): Target=100, Ind=10, Pop=100, Gen=50, Mut=0.05, Cruza=0.7 ---")
        algoritmo_genetico_deap(target=100, tam_individuo=10, tam_poblacion=100, generaciones=50, prob_mut=0.05, prob_cruza=0.7)
    except NameError:
        print("DEAP no disponible para ejecuciones.")