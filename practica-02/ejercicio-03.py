# ejercicio_3.py
# Este script implementa el ejercicio 3 sobre regresión lineal y polinomial
# Basado en la teoría de aprendizaje supervisado del documento IA-02 ML-Intro+Sup-Regresion.pdf
# Se genera un dataset sintético y se analizan modelos para underfitting y overfitting

# Importación de bibliotecas necesarias
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split

# Configuración para gráficos
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 12

print("Generando dataset sintético con 20 entradas basado en y = x² + ruido")

# Generación del dataset sintético para el ejercicio 3a y 3b
# X es un array 1D de 20 valores entre -3 y 3
np.random.seed(42)  # Para reproducibilidad
X = np.linspace(-3, 3, 20).reshape(-1, 1)
# y = x² + ruido gaussiano pequeño para simular datos reales
y = X**2 + np.random.normal(0, 0.5, size=(20, 1))

# 3a: Regresión lineal simple
print("\n3a: Regresión lineal simple")
lin_reg = LinearRegression()
lin_reg.fit(X, y)
y_pred_lin = lin_reg.predict(X)
mse_lin = mean_squared_error(y, y_pred_lin)
print(f"MSE de la regresión lineal: {mse_lin:.4f}")

# Gráfico de la regresión lineal
plt.figure()
plt.scatter(X, y, color='blue', label='Datos reales')
plt.plot(X, y_pred_lin, color='red', label='Recta ajustada (lineal)')
plt.xlabel('X')
plt.ylabel('y')
plt.title('Regresión Lineal sobre datos polinomiales (y ≈ x²)')
plt.legend()
plt.grid(True)
plt.show()

# 3b: Regresión polinomial de grado 2
print("\n3b: Regresión polinomial de grado 2")
poly = PolynomialFeatures(degree=2)
X_poly = poly.fit_transform(X)
lin_reg_poly = LinearRegression()
lin_reg_poly.fit(X_poly, y)
y_pred_poly = lin_reg_poly.predict(X_poly)
mse_poly = mean_squared_error(y, y_pred_poly)
print(f"MSE de la regresión polinomial (grado 2): {mse_poly:.4f}")

# Gráfico comparativo
plt.figure()
plt.scatter(X, y, color='blue', label='Datos reales')
plt.plot(X, y_pred_lin, color='red', label='Modelo lineal')
plt.plot(X, y_pred_poly, color='green', label='Modelo polinomial (grado 2)')
plt.xlabel('X')
plt.ylabel('y')
plt.title('Comparación: Lineal vs Polinomial (grado 2)')
plt.legend()
plt.grid(True)
plt.show()

# Imprimir dataset transformado para grados >2 (ejemplo con grado 3 y 4)
print("\nTransformación del dataset con PolynomialFeatures:")
for degree in [3, 4]:
    poly_high = PolynomialFeatures(degree=degree)
    X_poly_high = poly_high.fit_transform(X)
    print(f"\nPara grado {degree}:")
    print(f"Forma de X_poly: {X_poly_high.shape}")
    print("Primeras 5 filas del dataset transformado:")
    print(X_poly_high[:5])
    # Explicación: La primera columna es 1 (bias), luego X, X², X³, etc.

# 3c: Dataset más grande (100 puntos), split train/validation, exploración de grados
print("\n3c: Dataset grande con split train/validation y análisis de under/overfitting")
n_samples = 100
X_large = np.linspace(-3, 3, n_samples).reshape(-1, 1)
y_large = X_large**2 + np.random.normal(0, 0.5, size=(n_samples, 1))

# Split: 80% train, 20% test (validación)
X_train, X_test, y_train, y_test = train_test_split(X_large, y_large, test_size=0.2, random_state=42)

# Explorar grados de 1 a 10
degrees = range(1, 11)
mse_train_list = []
mse_test_list = []

for degree in degrees:
    poly = PolynomialFeatures(degree=degree)
    X_train_poly = poly.fit_transform(X_train)
    X_test_poly = poly.transform(X_test)
    
    lin_reg = LinearRegression()
    lin_reg.fit(X_train_poly, y_train)
    
    y_train_pred = lin_reg.predict(X_train_poly)
    y_test_pred = lin_reg.predict(X_test_poly)
    
    mse_train = mean_squared_error(y_train, y_train_pred)
    mse_test = mean_squared_error(y_test, y_test_pred)
    
    mse_train_list.append(mse_train)
    mse_test_list.append(mse_test)
    
    print(f"Grado {degree}: MSE Train = {mse_train:.4f}, MSE Test = {mse_test:.4f}")

# Gráfico de curvas de aprendizaje (bias-variance tradeoff)
plt.figure()
plt.plot(degrees, mse_train_list, label='MSE Train', marker='o')
plt.plot(degrees, mse_test_list, label='MSE Test', marker='s')
plt.xlabel('Grado del polinomio')
plt.ylabel('MSE')
plt.title('Análisis de Underfitting y Overfitting')
plt.legend()
plt.grid(True)
plt.show()

# Análisis:
# - Bajo grado (e.g., 1): Underfitting (alta MSE en train y test, modelo simple no captura la curvatura)
# - Grado óptimo (alrededor de 2): Buen balance, baja MSE en test
# - Alto grado (e.g., >4): Overfitting (baja MSE en train, pero alta en test, modelo memoriza ruido)

print("\nAnálisis basado en la teoría:")
print("- Underfitting: Ocurre con modelos simples (bajo grado) que no capturan la complejidad de los datos (e.g., relación polinomial). Alta bias.")
print("- Overfitting: Con modelos complejos (alto grado) que ajustan demasiado al ruido en entrenamiento, pero generalizan mal. Alta variance.")
print("- El grado 2 es ideal aquí, ya que los datos siguen y = x², como indica el Teorema No Free Lunch: no hay modelo universal, depende de los datos.")