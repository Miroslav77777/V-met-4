import numpy as np

def trapezoidal_rule(f, a, x, n=1000):
    """Вычисляет ∫ₐˣ f(t) dt методом трапеций."""
    if np.isclose(x, a):
        return 0.0
    t = np.linspace(a, x, n)
    y = f(t)
    h = (x - a) / (n - 1)
    return h * (0.5 * y[0] + 0.5 * y[-1] + np.sum(y[1:-1]))

def solve_integral_trapezoid(f, a, b_target, x_left, x_right, tol=1e-6, max_iter=100):
    """
    Решает ∫ₐˣ f(t) dt = b методом трапеций + бинарного поиска.
    
    Параметры:
        f: функция f(t)
        a: нижний предел
        b_target: целевое значение интеграла
        x_left, x_right: границы интервала для x
        tol: точность
        max_iter: максимальное число итераций
    """
    for _ in range(max_iter):
        x_mid = (x_left + x_right) / 2
        F_mid = trapezoidal_rule(f, a, x_mid)
        
        if np.abs(F_mid - b_target) < tol:
            return x_mid
        
        if F_mid < b_target:
            x_left = x_mid
        else:
            x_right = x_mid
    
    raise ValueError("Решение не найдено. Проверьте интервал.")

# Пример 1: ∫₀ˣ sin(t) dt = 1 → x ≈ π/2 ≈ 1.5708
f = lambda t: np.sin(t)
a = 0
b = 1.0
x_sol = solve_integral_trapezoid(f, a, b, x_left=0, x_right=2)
print(f"Решение: x ≈ {x_sol:.5f} (ожидается ~1.5708)")

# Пример 2: ∫₀ˣ e^{-t²} dt = 0.5 → x ≈ 0.5510
f = lambda t: np.exp(-t**2)
a = 0
b = 0.5
x_sol = solve_integral_trapezoid(f, a, b, x_left=0, x_right=1)
print(f"Решение: x ≈ {x_sol:.5f} (ожидается ~0.5510)")
