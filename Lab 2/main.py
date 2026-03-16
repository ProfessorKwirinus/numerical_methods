import numpy as np
import matplotlib.pyplot as plt
import csv



def prepare_input_data(filename="data.csv"):
    """Створює файл з даними згідно з Варіантом 1 [cite: 1115-1116]"""
    data = [
        ['n', 't'],  # Розмір даних та час виконання (мс)
        [1000, 3],
        [2000, 5],
        [4000, 11],
        [8000, 28],
        [16000, 85]
    ]
    with open(filename, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(data)
    return filename


def read_data(filename):
    """Зчитує дані для подальшої обробки [cite: 1094-1102]"""
    x, y = [], []
    with open(filename, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            x.append(float(row['n']))
            y.append(float(row['t']))
    return np.array(x), np.array(y)


def calculate_divided_differences(x, y):
    """Обчислює таблицю розділених різниць [cite: 928-931]"""
    n = len(y)
    coef = np.zeros([n, n])
    coef[:, 0] = y
    for j in range(1, n):
        for i in range(n - j):
            coef[i, j] = (coef[i + 1, j - 1] - coef[i, j - 1]) / (x[i + j] - x[i])
    return coef[0, :]  # Повертаємо коефіцієнти для формули Ньютона


def newton_poly(x_nodes, coefs, x):
    """Обчислює значення многочлена Ньютона N_n(x) [cite: 976]"""
    n = len(x_nodes) - 1
    p = coefs[n]
    for k in range(1, n + 1):
        p = coefs[n - k] + (x - x_nodes[n - k]) * p
    return p


def factorial_poly(x_nodes, y_nodes, x):
    """Інтерполяція факторіальними многочленами (для рівномірної сітки) [cite: 1071]"""
    h = x_nodes[1] - x_nodes[0]
    t = (x - x_nodes[0]) / h
    n = len(x_nodes)

    # Скінченні різниці Delta [cite: 1005-1006]
    diffs = np.zeros((n, n))
    diffs[:, 0] = y_nodes
    for j in range(1, n):
        for i in range(n - j):
            diffs[i, j] = diffs[i + 1, j - 1] - diffs[i, j - 1]

    res = diffs[0, 0]
    t_prod = 1.0
    fact = 1.0
    for k in range(1, n):
        t_prod *= (t - (k - 1))
        fact *= k
        res += (diffs[0, k] / fact) * t_prod
    return res


def omega_n(x_nodes, x):
    """Допоміжний многочлен w_n(x) [cite: 955]"""
    res = 1.0
    for xi in x_nodes:
        res *= (x - xi)
    return res



def runge_test():
    """Аналіз ефекту Рунге (коливання на краях) """
    f = lambda x: 1 / (1 + 25 * x ** 2)
    x_plot = np.linspace(-1, 1, 500)

    plt.figure(figsize=(10, 5))
    plt.plot(x_plot, f(x_plot), 'k--', label='Оригінальна функція (Рунге)')

    for n in [5, 11]:
        xn = np.linspace(-1, 1, n)
        yn = f(xn)
        c = calculate_divided_differences(xn, yn)
        plt.plot(x_plot, [newton_poly(xn, c, xi) for xi in x_plot], label=f'n={n} вузлів')

    plt.title("Дослідження ефекту Рунге")
    plt.legend()
    plt.grid(True)
    plt.show()



def run_lab():
    print("=" * 50)
    print("ЛАБОРАТОРНА РОБОТА №2: МЕТОД НЬЮТОНА")
    print("=" * 50)

    # Підготовка даних [cite: 1109]
    filename = prepare_input_data()
    x_nodes, y_nodes = read_data(filename)
    coefs = calculate_divided_differences(x_nodes, y_nodes)

    # 1. Прогноз для n=6000
    target = 6000
    res_n = newton_poly(x_nodes, coefs, target)
    res_f = factorial_poly(x_nodes, y_nodes, target)

    print(f"\n[РЕЗУЛЬТАТИ ВАРІАНТА 1]")
    print(f"Вхідні вузли n: {x_nodes}")
    print(f"Вхідні значення t: {y_nodes}")
    print(f"---")
    print(f"Прогноз для n={target} (Ньютон): {res_n:.2f} мс")
    print(f"Прогноз для n={target} (Факторіальний): {res_f:.2f} мс")

    # 2. Побудова графіків [cite: 1081-1082]
    x_range = np.linspace(min(x_nodes), max(x_nodes), 200)
    y_interp = [newton_poly(x_nodes, coefs, xi) for xi in x_range]
    errors = [abs(newton_poly(x_nodes, coefs, xi) - factorial_poly(x_nodes, y_nodes, xi)) for xi in x_range]
    omegas = [omega_n(x_nodes, xi) for xi in x_range]

    # Графік 1: Основна інтерполяція
    plt.figure(figsize=(10, 6))
    plt.scatter(x_nodes, y_nodes, color='red', zorder=5, label='Експериментальні точки')
    plt.plot(x_range, y_interp, 'b-', label='Поліном Ньютона')
    plt.axvline(x=target, color='green', linestyle=':', label=f'Точка прогнозу {target}')
    plt.title("Прогнозування часу виконання (Варіант 1)")
    plt.xlabel("Розмір даних n")
    plt.ylabel("Час t (мс)")
    plt.legend()
    plt.grid(True)
    plt.show()

    # Графік 2: Похибка та функція вузлів
    fig, ax1 = plt.subplots(figsize=(10, 5))
    ax1.plot(x_range, errors, 'r-', label='Похибка epsilon(x)')
    ax1.set_ylabel('Epsilon', color='r')

    ax2 = ax1.twinx()
    ax2.plot(x_range, omegas, 'g--', label='omega_n(x)')
    ax2.set_ylabel('Omega', color='g')

    plt.title("Аналіз похибки та функції omega_n(x)")
    plt.show()

    # 3. Дослідницька частина
    print("\n[ДОСЛІДЖЕННЯ ЕФЕКТУ РУНГЕ]")
    runge_test()

    print("\n" + "=" * 50)
    print("РОБОТА ЗАВЕРШЕНА УСПІШНО")
    print("=" * 50)


if __name__ == "__main__":
    run_lab()