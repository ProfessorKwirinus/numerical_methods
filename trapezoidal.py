import numpy as np

def f(x):
    return x**2  # Функція, яку інтегруємо

def trapezoidal_rule(f, a, b, n):
    x = np.linspace(a, b, n)
    y = f(x)
    h = (b - a) / (n - 1)
    # Формула: (h/2) * (перша + остання + 2 * сума внутрішніх)
    return (h / 2) * (y[0] + 2 * np.sum(y[1:-1]) + y[-1])

# Параметри з лекції
n = 20
a, b = 0, 1
result = trapezoidal_rule(f, a, b, n)

print(f"Наближене значення: {result:.6f}")
print(f"Точне значення: {1/3:.6f}")
