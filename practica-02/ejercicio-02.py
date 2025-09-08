import numpy as np
import matplotlib.pyplot as plt

# Configuración inicial
np.random.seed(42)  # Para reproducibilidad de los datos sintéticos
n_samples = 100  # Número de puntos de datos

# Generar datos sintéticos
x = np.linspace(0, 10, n_samples)  # Valores de x entre 0 y 10
e = np.random.normal(0, 1, n_samples)  # Ruido gaussiano N(0, 1)
y = 3 * x + 2 + e  # Ecuación y = 3x + 2 + e

# a) Aprender un modelo con la Ecuación Normal
# Añadir una columna de unos a X para el intercepto
X = np.vstack((np.ones(n_samples), x)).T  # Matriz X con forma (n_samples, 2)

# Calcular theta usando la Ecuación Normal: theta = (X^T * X)^(-1) * X^T * y
theta_normal = np.linalg.inv(X.T @ X) @ X.T @ y  # @ es multiplicación de matrices
theta_0_normal = theta_normal[0]  # Intercepto
theta_1_normal = theta_normal[1]  # Pendiente

print(f"Coeficientes con Ecuación Normal: Intercepto = {theta_0_normal:.2f}, Pendiente = {theta_1_normal:.2f}")

# b) Aprender un modelo con Gradiente Descendente
# Inicializar parámetros
theta_0_gd = 0.0  # Intercepto inicial
theta_1_gd = 0.0  # Pendiente inicial
learning_rate = 0.01  # Tasa de aprendizaje
n_iterations = 1000  # Número de iteraciones
m = n_samples  # Número de muestras

# Listas para almacenar el error y los parámetros durante las iteraciones
errors = []
theta_0_history = []
theta_1_history = []

# Función de costo (error cuadrático medio)
def compute_cost(theta_0, theta_1, x, y):
    predictions = theta_0 + theta_1 * x
    return np.mean((y - predictions) ** 2)

# Gradiente Descendente
for _ in range(n_iterations):
    # Calcular predicciones
    predictions = theta_0_gd + theta_1_gd * x
    
    # Calcular gradientes
    gradient_0 = -(2/m) * np.sum(y - predictions)  # Gradiente del intercepto
    gradient_1 = -(2/m) * np.sum((y - predictions) * x)  # Gradiente de la pendiente
    
    # Actualizar parámetros
    theta_0_gd = theta_0_gd - learning_rate * gradient_0
    theta_1_gd = theta_1_gd - learning_rate * gradient_1
    
    # Almacenar el error y los parámetros
    error = compute_cost(theta_0_gd, theta_1_gd, x, y)
    errors.append(error)
    theta_0_history.append(theta_0_gd)
    theta_1_history.append(theta_1_gd)

print(f"Coeficientes con Gradiente Descendente: Intercepto = {theta_0_gd:.2f}, Pendiente = {theta_1_gd:.2f}")

# c) Comparar resultados
print(f"Diferencia en intercepto: {abs(theta_0_normal - theta_0_gd):.2f}")
print(f"Diferencia en pendiente: {abs(theta_1_normal - theta_1_gd):.2f}")

# d) Graficar la convergencia del error
plt.figure(figsize=(10, 5))
plt.plot(range(n_iterations), errors, color="blue", label="Error (Costo)")
plt.xlabel("Iteración")
plt.ylabel("Error Cuadrático Medio")
plt.title("Convergencia del Error en Gradiente Descendente")
plt.legend()
plt.grid(True)
plt.savefig("convergencia_error.png", dpi=300, bbox_inches="tight")
plt.show()

# Graficar los datos y las rectas ajustadas
plt.figure(figsize=(10, 5))
plt.scatter(x, y, color="blue", label="Datos sintéticos")
plt.plot(x, theta_0_normal + theta_1_normal * x, color="red", label=f"Ecuación Normal: y={theta_1_normal:.2f}x+{theta_0_normal:.2f}")
plt.plot(x, theta_0_gd + theta_1_gd * x, color="green", label=f"Gradiente Descendente: y={theta_1_gd:.2f}x+{theta_0_gd:.2f}")
plt.xlabel("x")
plt.ylabel("y")
plt.title("Comparación de Modelos")
plt.legend()
plt.grid(True)
plt.savefig("comparacion_modelos.png", dpi=300, bbox_inches="tight")
plt.show()