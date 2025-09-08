import matplotlib.pyplot as plt
import numpy as np

# Datos originales
x = np.array([1, 2, 3, 4, 5])
y = np.array([2, 3, 5, 7, 8])

# Cálculo de coeficientes usando mínimos cuadrados (datos originales)
n = len(x)
w = (n * np.sum(x * y) - np.sum(x) * np.sum(y)) / (n * np.sum(x**2) - (np.sum(x))**2)
b = (np.sum(y) - w * np.sum(x)) / n

print(f"Coeficientes originales: w = {w:.2f}, b = {b:.2f}")

# Agregar más valores y valores atípicos
x_extended = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])  # Más valores
y_extended = np.array([2, 3, 5, 7, 8, 10, 12, 14, 100])  # Valor atípico (100)

# Recalcular coeficientes con datos extendidos
n_extended = len(x_extended)
w_extended = (n_extended * np.sum(x_extended * y_extended) - np.sum(x_extended) * np.sum(y_extended)) / (n_extended * np.sum(x_extended**2) - (np.sum(x_extended))**2)
b_extended = (np.sum(y_extended) - w_extended * np.sum(x_extended)) / n_extended

print(f"Coeficientes con atípicos: w = {w_extended:.2f}, b = {b_extended:.2f}")

# Gráfico
plt.scatter(x, y, color="blue", label="Datos originales")
plt.plot(x, w * x + b, color="red", label=f"Recta original: y={w:.2f}x+{b:.2f}")

plt.scatter(x_extended, y_extended, color="green", label="Datos con atípicos")
plt.plot(x_extended, w_extended * x_extended + b_extended, color="orange", label=f"Recta con atípicos: y={w_extended:.2f}x+{b_extended:.2f}")

# Ajustar escala del eje y (rango más pequeño, evitando el atípico extremo)
plt.ylim(0, 15)  # Ajuste manual del eje y (puedes modificarlo según necesites)

plt.xlabel("x")
plt.ylabel("y")
plt.title("Ajuste de recta con mínimos cuadrados")
plt.legend()
plt.grid(True)

# Guardar el gráfico en un archivo PNG
plt.savefig("grafico_recta.png", dpi=300, bbox_inches="tight")

# Mostrar el gráfico (opcional)
plt.show()