## a) Ajuste de una recta \( y = wx + b \)

El método de **mínimos cuadrados** busca minimizar la suma de los errores al cuadrado entre los valores observados \( y \) y los valores predichos \( \hat{y} = wx + b \).

Para un conjunto de datos \( (x_i, y_i) \), los coeficientes \( w \) (pendiente) y \( b \) (intersección) se calculan con las siguientes fórmulas:

- **Pendiente:**
  \[
  w = \frac{n \sum (x_i y_i) - \sum x_i \sum y_i}{n \sum x_i^2 - (\sum x_i)^2}
  \]

- **Intersección:**
  \[
  b = \frac{\sum y_i - w \sum x_i}{n}
  \]

Donde \( n \) es el número de puntos de datos.

---

### Datos proporcionados

\[
(x, y) = \{(1, 2), (2, 3), (3, 5), (4, 7), (5, 8)\}
\]

- \( n = 5 \)
- \( \sum x_i = 1 + 2 + 3 + 4 + 5 = 15 \)
- \( \sum y_i = 2 + 3 + 5 + 7 + 8 = 25 \)
- \( \sum x_i y_i = (1 \cdot 2) + (2 \cdot 3) + (3 \cdot 5) + (4 \cdot 7) + (5 \cdot 8) = 91 \)
- \( \sum x_i^2 = 1^2 + 2^2 + 3^2 + 4^2 + 5^2 = 55 \)

---

### Cálculo de \( w \)

\[
w = \frac{5 \cdot 91 - 15 \cdot 25}{5 \cdot 55 - 15^2} = \frac{455 - 375}{275 - 225} = \frac{80}{50} = 1.6
\]

### Cálculo de \( b \)

\[
b = \frac{25 - 1.6 \cdot 15}{5} = \frac{25 - 24}{5} = 0.2
\]

---

### Resultado final

La recta ajustada a los datos es:

\[
\boxed{y = 1.6x + 0.2}
\]

El método de mínimos cuadrados permite encontrar la mejor recta que ajusta un conjunto de datos minimizando la suma de los errores al cuadrado. Este ejemplo muestra paso a paso cómo calcular \( w \) y \( b \) manualmente.
# inteligencia-artificial-unrc
