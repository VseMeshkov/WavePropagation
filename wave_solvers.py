import numpy as np
from scipy.optimize import minimize


def solve_wave_fd(data, c_est, n_substeps=8):
    """
    Конечно-разностный солвер для 3D волнового уравнения:
        u_tt = c^2 * Laplacian(u)

    Ключевые особенности:
      * n_substeps ФИКСИРОВАНО (не зависит от c_est), чтобы factor
        монотонно зависел от c_est и функция потерь была гладкой.
      * Первый шаг по времени инициализируется через формулу Тейлора
        (u_t(x, 0) = 0), НЕ берётся из данных.
      * Проверка условия Куранта + защита от нефизического роста.

    Параметры
    ----------
    data : dict
        Словарь с ключами "x", "y", "z", "t", "u".
        u имеет форму (nt, nx, ny, nz).
    c_est : float
        Оцениваемая скорость волны.
    n_substeps : int
        Число суб-шагов на каждый выходной шаг по времени. По умолчанию 4.

    Возвращает
    ----------
    u : np.ndarray, форма (nt, nx, ny, nz)
        Численное решение. Если схема неустойчива или решение взрывается,
        возвращается массив, заполненный np.nan.
    """
    # --- Распаковка данных ---
    x, y, z, t = data["x"], data["y"], data["z"], data["t"]
    u0 = data["u"][0]

    nx, ny, nz = len(x), len(y), len(z)
    dx = x[1] - x[0]
    dy = y[1] - y[0]
    dz = z[1] - z[0]
    dt_out = t[1] - t[0]
    nt = len(t)

    # --- Шаг по времени и параметр схемы ---
    dt = dt_out / n_substeps
    factor = (c_est * dt) ** 2

    # --- Проверка условия Куранта (для 3D) ---
    # c * dt * sqrt(1/dx^2 + 1/dy^2 + 1/dz^2) <= 1
    inv_dx2 = 1.0 / dx**2 + 1.0 / dy**2 + 1.0 / dz**2
    courant = c_est * dt * np.sqrt(inv_dx2)
    if courant > 1.0 or not np.isfinite(courant):
        return np.full((nt, nx, ny, nz), np.nan)

    # --- Вспомогательная функция: лапласиан ---
    def laplacian_of(w):
        """7-точечный лапласиан, границы остаются нулевыми."""
        lap = np.zeros_like(w)
        lap[1:-1, 1:-1, 1:-1] = (
            (w[2:, 1:-1, 1:-1] - 2.0 * w[1:-1, 1:-1, 1:-1] + w[:-2, 1:-1, 1:-1]) / dx**2
            + (w[1:-1, 2:, 1:-1] - 2.0 * w[1:-1, 1:-1, 1:-1] + w[1:-1, :-2, 1:-1]) / dy**2
            + (w[1:-1, 1:-1, 2:] - 2.0 * w[1:-1, 1:-1, 1:-1] + w[1:-1, 1:-1, :-2]) / dz**2
        )
        return lap

    # --- Инициализация ---
    u = np.zeros((nt, nx, ny, nz), dtype=np.float64)
    u[0] = u0

    # Инициализация первого шага через Тейлор:
    #   u_1 = u_0 + dt * u_t(0) + 0.5 * dt^2 * u_tt(0)
    #   u_t(0) = 0  (покой)
    #   u_tt(0) = c^2 * Laplacian(u_0)
    #   => u_1 = u_0 + 0.5 * factor * Laplacian(u_0)
    u_prev = u0.copy()
    lap0 = laplacian_of(u0)
    u_curr = u0 + 0.5 * factor * lap0

    # Остальные суб-шаги до t[1]
    for _ in range(n_substeps - 1):
        lap = laplacian_of(u_curr)
        u_next = 2.0 * u_curr - u_prev + factor * lap
        u_prev = u_curr
        u_curr = u_next

    if not np.all(np.isfinite(u_curr)):
        return np.full((nt, nx, ny, nz), np.nan)
    u[1] = u_curr

    # --- Основной цикл по выходным шагам ---
    u0_norm = np.linalg.norm(u0) + 1e-12  # защита от деления на 0

    for i in range(2, nt):
        for _ in range(n_substeps):
            lap = laplacian_of(u_curr)
            u_next = 2.0 * u_curr - u_prev + factor * lap

            # Защита от взрыва
            if not np.all(np.isfinite(u_next)):
                return np.full((nt, nx, ny, nz), np.nan)
            if np.linalg.norm(u_next) > 10.0 * u0_norm:
                return np.full((nt, nx, ny, nz), np.nan)

            u_prev = u_curr
            u_curr = u_next

        u[i] = u_curr

    return u


def loss_function(c_est, data, n_frames=5):
    """
    MSE по нескольким кадрам (а не только по последнему).
    Это устраняет 'ложные минимумы', возникающие из-за случайных
    совпадений в отдельном кадре.
    """
    u_sim = solve_wave_fd(data, c_est)
    if np.any(np.isnan(u_sim)):
        return 1e10

    u_true = data["u"]
    nt = u_true.shape[0]
    # Берём n_frames равномерно распределённых кадров из второй половины
    idx = np.linspace(nt // 4, nt - 1, n_frames, dtype=int)

    mse = 0.0
    for i in idx:
        mse += np.mean((u_sim[i] - u_true[i]) ** 2)
    return float(mse / n_frames)


def optimize_c_robust(data, c_min=0.05, c_max=2.0, n_grid=20, n_refine=3):
    """
    Надёжная стратегия оптимизации c:
      1. Грубое сканирование по сетке.
      2. Локальная оптимизация из лучших стартовых точек (L-BFGS-B).
      3. Возврат лучшего результата.

    Параметры
    ----------
    data : dict
        Данные, с которыми сравниваем симуляцию.
    c_min, c_max : float
        Границы поиска c.
    n_grid : int
        Число точек на грубой сетке.
    n_refine : int
        Сколько лучших точек с сетки использовать как стартовые для L-BFGS-B.

    Возвращает
    ----------
    best_c : float
        Найденное оптимальное c.
    """
    # --- Шаг 1: грубое сканирование ---
    c_grid = np.linspace(c_min, c_max, n_grid)
    losses = np.array([loss_function(c, data) for c in c_grid])

    # --- Шаг 2: топ-N лучших стартовых точек ---
    idx_sorted = np.argsort(losses)[:n_refine]
    top_starts = [(c_grid[i], losses[i]) for i in idx_sorted]
    print(f"Топ-{n_refine} стартовых точек по сетке: "
          f"{[(round(c, 3), f'{l:.2e}') for c, l in top_starts]}")

    # --- Шаг 3: локальная оптимизация ---
    best_c, best_loss = None, np.inf
    for c0, _ in top_starts:
        # Локальные границы вокруг старта
        c_lo = max(c_min, c0 - 0.15)
        c_hi = min(c_max, c0 + 0.15)
        res = minimize(
            loss_function,
            c0,
            args=(data,),
            method="L-BFGS-B",
            bounds=[(c_lo, c_hi)],
            options={"ftol": 1e-14, "gtol": 1e-8},
        )
        c_opt, loss_opt = res.x[0], res.fun
        print(f"  Старт {c0:.3f} → c_opt = {c_opt:.5f}, Loss = {loss_opt:.6e}")

        if loss_opt < best_loss:
            best_loss = loss_opt
            best_c = c_opt

    print(f"✅ Итог: c = {best_c:.5f}, Loss = {best_loss:.6e}")
    return best_c


if __name__ == "__main__":
    # Пример использования
    from wave_data_loader import get_wave_data

    data = get_wave_data("homogeneous", nx=40, ny=40, nz=40, nt=300, c=1.0, T=2.0)

    print("Истинное c в данных:", data["c"])

    # Сканирование по c для отладки
    print("\nСканирование по c:")
    for c_test in [0.5, 0.7, 0.9, 1.0, 1.1, 1.3, 1.5]:
        L = loss_function(c_test, data)
        print(f"  c = {c_test:.2f} → Loss = {L:.6e}")

    print("\nОптимизация:")
    c_opt = optimize_c_robust(data)
    print(f"\nОтносительная ошибка: {abs(c_opt - 1.0) / 1.0 * 100:.4f}%")