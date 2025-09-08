import random

# Función de fitness: cuenta los bits en 1
def fitness(individuo):
    return sum(individuo)

# Generar un individuo aleatorio
def generar_individuo(longitud):
    return [random.randint(0, 1) for _ in range(longitud)]

# Generar una población inicial
def generar_poblacion(tam_poblacion, longitud):
    return [generar_individuo(longitud) for _ in range(tam_poblacion)]

# Selección por torneo
def seleccion(poblacion, k=3):
    seleccionados = random.sample(poblacion, k)
    return max(seleccionados, key=fitness)

# Cruza de un punto con probabilidad
def cruza(padre1, padre2, prob_cruza=0.8):
    if random.random() < prob_cruza:
        punto = random.randint(1, len(padre1)-1)
        hijo1 = padre1[:punto] + padre2[punto:]
        hijo2 = padre2[:punto] + padre1[punto:]
        return hijo1, hijo2
    return padre1[:], padre2[:]

# Mutación por flip de bit
def mutacion(individuo, prob):
    return [1 - gen if random.random() < prob else gen for gen in individuo]

# Algoritmo Genético mejorado
def algoritmo_genetico(tam_individuo=20, tam_poblacion=50, generaciones=100, prob_mut=0.01, prob_cruza=0.8, elitismo=1):
    random.seed(42)  # Para reproducibilidad
    poblacion = generar_poblacion(tam_poblacion, tam_individuo)
    
    for gen in range(generaciones):
        # Evaluar y ordenar población
        poblacion.sort(key=fitness, reverse=True)
        mejor_actual = poblacion[0]
        avg_fitness = sum(fitness(ind) for ind in poblacion) / tam_poblacion
        print(f"Generación {gen}: Mejor fitness = {fitness(mejor_actual)}, Avg fitness = {avg_fitness:.2f}")
        
        # Elitismo: copiar los mejores directamente
        nueva_poblacion = poblacion[:elitismo]
        
        while len(nueva_poblacion) < tam_poblacion:
            # Selección de padres
            padre1 = seleccion(poblacion)
            padre2 = seleccion(poblacion)
            # Cruza
            hijo1, hijo2 = cruza(padre1, padre2, prob_cruza)
            # Mutación
            hijo1 = mutacion(hijo1, prob_mut)
            hijo2 = mutacion(hijo2, prob_mut)
            nueva_poblacion.extend([hijo1, hijo2])
        
        # Reemplazar población (truncar si extra)
        poblacion = nueva_poblacion[:tam_poblacion]
    
    poblacion.sort(key=fitness, reverse=True)
    return poblacion[0]

# Ejemplo de ejecución
if __name__ == "__main__":
    mejor = algoritmo_genetico()
    print("Mejor individuo encontrado:", mejor)
    print("Fitness:", fitness(mejor))