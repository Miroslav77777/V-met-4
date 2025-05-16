import numpy as np

def trapezoidal_integral(f, a, x, n=1000):
    """
    Вычисляет ∫ₐˣ f(t) dt методом трапеций.
    
    Параметры:
        f: функция f(t)
        a: нижний предел
        x: верхний предел
        n: число разбиений
    
    Возвращает:
        Значение интеграла
    """
    if x == a:
        return 0.0
    t = np.linspace(a, x, n)
    y = f(t)
    h = (x - a) / (n - 1)
    integral = h * (0.5 * y[0] + 0.5 * y[-1] + np.sum(y[1:-1]))
    return integral

def solve_integral_eq_trapezoid(f, a, b_target, x_init=None, tol=1e-6, max_iter=100, n_points=1000):
    """
    Решает уравнение ∫ₐˣ f(t) dt = b методом трапеций + Ньютона.
    
    Параметры:
        f: функция f(t)
        a: нижний предел
        b_target: целевое значение интеграла
        x_init: начальное приближение (по умолчанию a + 1.0)
        tol: точность
        max_iter: максимальное число итераций
        n_points: число точек для интегрирования
    
    Возвращает:
        x: решение уравнения
    """
    if x_init is None:
        x_prev = a + 1.0
    else:
        x_prev = x_init

    for _ in range(max_iter):
        # Вычисляем интеграл методом трапеций
        integral = trapezoidal_integral(f, a, x_prev, n_points)

        # Проверяем условие останова
        if abs(integral - b_target) < tol:
            return x_prev

        # Защита от деления на ноль
        if abs(f(x_prev)) < 1e-12:
            x_prev += 0.1
            continue

        # Итерация Ньютона
        x_new = x_prev - (integral - b_target) / f(x_prev)

        # Обновляем x
        x_prev = x_new

    raise ValueError(f"Метод не сошелся за {max_iter} итераций")

# Пример 1: ∫₀ˣ sin(t) dt = 1 → x ≈ 1.5708 (π/2)
f = lambda t: np.sin(t)
a = 0
b_target = 1.0
x_sol = solve_integral_eq_trapezoid(f, a, b_target)
print(f"Решение: x ≈ {x_sol:.5f} (ожидается ~1.5708)")

# Пример 2: ∫₀ˣ e^{-t²} dt = 0.5 → x ≈ 0.5510
f = lambda t: np.exp(-t**2)
a = 0
b_target = 0.5
x_sol = solve_integral_eq_trapezoid(f, a, b_target)
print(f"Решение: x ≈ {x_sol:.5f} (ожидается ~0.5510)")
