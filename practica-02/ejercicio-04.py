# ejercicio_4.py
# Este script implementa el ejercicio 4 sobre regresión con un dataset real
# Usando California Housing para predecir precios de casas
# Basado en la teoría de importancia de los datos y modelos del documento IA-03 ML-Caso de Estudio - Python.pdf

# Importación de bibliotecas necesarias
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error

print("Seleccionando dataset: California Housing de sklearn.datasets")
print("Este dataset es multivariado para regresión: predecir precio de casas en California basado en features como ingresos medios, ubicación, etc.")
print("Realizando regresión lineal múltiple para capturar el conocimiento.")

# Cargar dataset
housing = fetch_california_housing()
X_housing = housing.data  # Features (8 columnas: MedInc, HouseAge, etc.)
y_housing = housing.target  # Target: precio medio (en 100k USD)

print(f"Forma del dataset: {X_housing.shape} features, {y_housing.shape} targets")
print("Nombres de features:", housing.feature_names)

# Split train/test
X_train_h, X_test_h, y_train_h, y_test_h = train_test_split(X_housing, y_housing, test_size=0.2, random_state=42)

# Regresión lineal múltiple
lin_reg_h = LinearRegression()
lin_reg_h.fit(X_train_h, y_train_h)
y_train_pred_h = lin_reg_h.predict(X_train_h)
y_test_pred_h = lin_reg_h.predict(X_test_h)

mse_train_h = mean_squared_error(y_train_h, y_train_pred_h)
mse_test_h = mean_squared_error(y_test_h, y_test_pred_h)

print(f"\nMSE Train: {mse_train_h:.4f}")
print(f"MSE Test: {mse_test_h:.4f}")
print(f"Coeficientes del modelo: {lin_reg_h.coef_}")
print(f"Intercepto: {lin_reg_h.intercept_:.4f}")

# Para mejorar (capturar mejor el conocimiento): Usar Ridge para regularización y evitar overfitting
# Explorar alpha en Ridge (regularización L2)
param_grid = {'alpha': [0.1, 1.0, 10.0, 100.0]}
ridge = Ridge()
grid_search = GridSearchCV(ridge, param_grid, cv=5, scoring='neg_mean_squared_error')
grid_search.fit(X_train_h, y_train_h)

best_ridge = grid_search.best_estimator_
y_test_pred_ridge = best_ridge.predict(X_test_h)
mse_test_ridge = mean_squared_error(y_test_h, y_test_pred_ridge)

print(f"\nMejor modelo Ridge - Alpha: {grid_search.best_params_['alpha']}")
print(f"MSE Test con Ridge: {mse_test_ridge:.4f} (mejor que lineal si hay multicolinealidad)")

# Análisis basado en la teoría:
print("\nAnálisis basado en la teoría del documento:")
print("- Importancia de los datos: Este dataset real destaca la necesidad de datos de calidad (evitar sesgos como en el ejemplo de Literary Digest).")
print("- Aprendizaje supervisado: Usamos etiquetas (precios) para entrenar y generalizar.")
print("- Modelos: LinearRegression asume linealidad; Ridge previene overfitting en datasets con features correlacionadas, alineado con NFL (no free lunch).")
print("- Para capturar mejor: Podríamos escalar features (StandardScaler) o usar más complejos como RandomForestRegressor, pero aquí nos enfocamos en lineal con regularización.")