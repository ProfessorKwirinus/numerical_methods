import numpy as np
import matplotlib.pyplot as plt
import csv
import math

x_v5 = np.array([100, 200, 400, 800, 1600])
y_v5 = np.array([120, 110, 90, 65, 40])

def fps_model(n):
    return 252.8 - 28.85 * np.log(n)

def divided_differences(x, y):
    n = len(y)
    coef = np.zeros([n, n])
    coef[:, 0] = y
    for j in range(1, n):
        for i in range(n - j):
            coef[i][j] = (coef[i+1][j-1] - coef[i][j-1]) / (x[i+j] - x[i])
    return coef[0, :]

def newton_general(coef, x_nodes, x):
    n = len(coef)
    val = coef[0]
    for k in range(1, n):
        prod = 1.0
        for i in range(k):
            prod *= (x - x_nodes[i])
        val += coef[k] * prod
    return val

def finite_differences(y):
    n = len(y)
    diffs = np.zeros([n, n])
    diffs[:, 0] = y
    for j in range(1, n):
        for i in range(n - j):
            diffs[i][j] = diffs[i+1][j-1] - diffs[i][j-1]
    return diffs[0, :]

def newton_factorial(y_nodes, x_nodes, x):
    n = len(y_nodes)
    h = x_nodes[1] - x_nodes[0]
    diffs = finite_differences(y_nodes)
    t = (x - x_nodes[0]) / h
    val = diffs[0]
    t_prod = 1.0
    for k in range(1, n):
        t_prod *= (t - (k - 1))
        val += (diffs[k] * t_prod) / math.factorial(k)
    return val

node_counts = [5, 10, 15, 20]
x_plot = np.linspace(100, 1600, 1000)
styles = [
    {'c': 'red', 'lw': 4, 'ls': '-'},
    {'c': 'blue', 'lw': 3, 'ls': '--'},
    {'c': 'orange', 'lw': 2, 'ls': '-.'},
    {'c': 'green', 'lw': 1.5, 'ls': '-'}
]

plt.figure(figsize=(10, 6))
for i, n in enumerate(node_counts):
    xn = np.linspace(100, 1600, n)
    yn = fps_model(xn)
    if n == 5: xn, yn = x_v5, y_v5
    c = divided_differences(xn, yn)
    y_interp = [newton_general(c, xn, xi) for xi in x_plot]
    plt.plot(x_plot, y_interp, label=f'Вузлів: {n}', color=styles[i]['c'], lw=styles[i]['lw'], ls=styles[i]['ls'])
plt.scatter(x_v5, y_v5, color='black', label='Точки Варіанту 5', zorder=10)
plt.title('1. Загальна форма Ньютона ')
plt.xlabel('Кількість об\'єктів (n)'); plt.ylabel('FPS')
plt.legend(); plt.grid(True, alpha=0.3); plt.show()

plt.figure(figsize=(10, 6))
for i, n in enumerate(node_counts):
    xn = np.linspace(100, 1600, n)
    yn = fps_model(xn)
    y_interp = [newton_factorial(yn, xn, xi) for xi in x_plot]
    plt.plot(x_plot, y_interp, label=f'Вузлів: {n}', color=styles[i]['c'], lw=styles[i]['lw'], ls=styles[i]['ls'])
plt.title('2. Факторіальна форма ')
plt.xlabel('Кількість об\'єктів (n)'); plt.ylabel('FPS')
plt.legend(); plt.grid(True, alpha=0.3); plt.show()

plt.figure(figsize=(10, 6))
y_true = fps_model(x_plot)
for i, n in enumerate(node_counts):
    xn = np.linspace(100, 1600, n)
    yn = fps_model(xn)
    y_interp = np.array([newton_factorial(yn, xn, xi) for xi in x_plot])
    plt.plot(x_plot, np.abs(y_true - y_interp), label=f'n={n}', color=styles[i]['c'], lw=styles[i]['lw'])
plt.title('3. Абсолютна похибка факторіального методу')
plt.xlabel('n'); plt.ylabel('Похибка ε(x)')
plt.legend(); plt.grid(True, alpha=0.3); plt.show()

final_c = divided_differences(x_v5, y_v5)
print(f"Прогноз для 1000 об'єктів (n=5): {newton_general(final_c, x_v5, 1000):.2f} FPS")