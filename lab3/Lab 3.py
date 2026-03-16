import numpy as np
import matplotlib.pyplot as plt

x_nodes = np.arange(1, 25)
y_nodes = np.array([-2, 0, 5, 10, 15, 20, 23, 22, 17, 10, 5, 0,
                    -10, 3, 7, 13, 19, 20, 22, 21, 18, 15, 10, 3])

def gauss_solve(A, b):
    n = len(b)
    Ab = np.column_stack((A, b)).astype(float)
    for k in range(n):
        max_row = np.argmax(np.abs(Ab[k:, k])) + k
        Ab[[k, max_row]] = Ab[[max_row, k]]
        for i in range(k + 1, n):
            factor = Ab[i, k] / Ab[k, k]
            Ab[i, k:] -= factor * Ab[k, k:]
    x_sol = np.zeros(n)
    for i in range(n - 1, -1, -1):
        x_sol[i] = (Ab[i, -1] - np.dot(Ab[i, i+1:n], x_sol[i+1:n])) / Ab[i, i]
    return x_sol

def lsm_weights(x, y, m, w=None):
    if w is None: w = np.ones(len(x))
    B = np.zeros((m + 1, m + 1))
    C = np.zeros(m + 1)
    for k in range(m + 1):
        for l in range(m + 1):
            B[k, l] = np.sum(w * (x**(k + l)))
        C[k] = np.sum(w * y * (x**k))
    return gauss_solve(B, C)

def poly_val(coef, x):
    if np.isscalar(x): return sum(c * (x**i) for i, c in enumerate(coef))
    return np.array([sum(c * (xi**i) for i, c in enumerate(coef)) for xi in x])

def lagrange_val(xn, yn, xv):
    if np.isscalar(xv):
        return sum(yn[i] * np.prod([(xv - xn[j])/(xn[i] - xn[j]) for j in range(len(xn)) if i != j]) for i in range(len(xn)))
    return np.array([lagrange_val(xn, yn, xi) for xi in xv])

def newton_val(xn, yn, xv):
    n = len(yn)
    coef = yn.astype(float)
    for j in range(1, n):
        coef[j:n] = (coef[j:n] - coef[j-1:n-1]) / (xn[j:n] - xn[0:n-j])
    def _val(v):
        p = coef[-1]
        for k in range(2, n + 1): p = coef[-k] + (v - xn[-k]) * p
        return p
    return _val(xv) if np.isscalar(xv) else np.array([_val(vi) for vi in xv])

variances = []
degrees = range(1, 11)
for m in degrees:
    c = lsm_weights(x_nodes, y_nodes, m)
    v = np.sqrt(np.mean((poly_val(c, x_nodes) - y_nodes)**2))
    variances.append(v)

opt_m = degrees[np.argmin(variances)]
opt_c = lsm_weights(x_nodes, y_nodes, opt_m)

x_h1 = np.linspace(1, 24, 20 * len(x_nodes))
x_ext = np.linspace(24, 27, 100)
x_future = np.array([25, 26, 27])

plt.figure(figsize=(12, 8))
plt.plot(x_nodes, y_nodes, 'ko', label='Вузли спостережень', markersize=6)
plt.plot(x_h1, poly_val(opt_c, x_h1), 'r-', linewidth=2, label=f'МНК апроксимація (m={opt_m})')
plt.title('Апроксимація температурних даних', fontsize=14)
plt.xlabel('Місяць', fontsize=12); plt.ylabel('Температура (°C)', fontsize=12)
plt.legend(); plt.grid(True, linestyle='--', alpha=0.7)
plt.show()

plt.figure(figsize=(12, 8))
plt.plot(x_nodes[-6:], y_nodes[-6:], 'ko', label='Останні дані')
plt.plot(x_ext, poly_val(opt_c, x_ext), 'r--', label='Лінія прогнозу')
plt.plot(x_future, poly_val(opt_c, x_future), 'b*', markersize=12, label='Прогноз на 3 місяці')
plt.title('Екстраполяція та прогноз температури', fontsize=14)
plt.xlabel('Місяць', fontsize=12); plt.ylabel('Температура (°C)', fontsize=12)
plt.legend(); plt.grid(True, linestyle='--', alpha=0.7)
plt.show()

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(degrees, variances, 'bo-', linewidth=2)
plt.axvline(opt_m, color='r', linestyle='--', label=f'Опт. m={opt_m}')
plt.title('Залежність дисперсії від степеня m')
plt.xlabel('Ступінь m'); plt.ylabel('Дисперсія δ'); plt.legend(); plt.grid(True)

plt.subplot(1, 2, 2)
for m_test in [1, 2, 4, opt_m]:
    c_t = lsm_weights(x_nodes, y_nodes, m_test)
    err = np.abs(poly_val(c_t, x_h1) - lagrange_val(x_nodes, y_nodes, x_h1))
    plt.plot(x_h1, err, label=f'm={m_test}')
plt.title('Графік похибки ε(x) на сітці h1')
plt.xlabel('x'); plt.ylabel('Похибка'); plt.legend(); plt.grid(True)
plt.tight_layout()
plt.show()

print(f"Оптимальний ступінь полінома: {opt_m}")
print(f"Прогноз на 25, 26, 27 місяці: {np.round(poly_val(opt_c, x_future), 2)}")