import numpy as np

def generate_homogeneous_wave(nx=50, ny=50, nz=50, nt=100, c=1.0, T=2.0):
    """
    Источник 1: Аналитическое решение для однородной среды.
    u(x,y,z,t) = sin(pi*x) * sin(pi*y) * sin(pi*z) * cos(sqrt(3)*pi*c*t)
    """
    x = np.linspace(0, 1, nx)
    y = np.linspace(0, 1, ny)
    z = np.linspace(0, 1, nz)
    t = np.linspace(0, T, nt)
    
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    
    u = np.zeros((nt, nx, ny, nz))
    for i, ti in enumerate(t):
        u[i] = np.sin(np.pi * X) * np.sin(np.pi * Y) * np.sin(np.pi * Z) * np.cos(np.sqrt(3) * np.pi * c * ti)
    
    return {"x": x, "y": y, "z": z, "t": t, "u": u, "c": c}

def generate_inhomogeneous_wave(nx=50, ny=50, nz=50, nt=100, c0=1.0, T=2.0):
    """
    Источник 2: Численное решение (FD) для среды с препятствием.
    Скорость c(x,y,z) = c0 * (1 + 0.5 * exp(-r²/0.1))
    
    Использует суб-шаги для устойчивости (условие Куранта).
    """
    x = np.linspace(0, 1, nx)
    y = np.linspace(0, 1, ny)
    z = np.linspace(0, 1, nz)
    t = np.linspace(0, T, nt)
    dx = x[1] - x[0]
    dy = y[1] - y[0]
    dz = z[1] - z[0]
    dt_out = t[1] - t[0]
    
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    
    # Неоднородная скорость
    c_field = c0 * (1 + 0.5 * np.exp(-((X-0.5)**2 + (Y-0.5)**2 + (Z-0.5)**2) / 0.1))
    c_max = c_field.max()
    
    # ---- Суб-шаги для устойчивости ----
    inv_dx2 = 1.0/dx**2 + 1.0/dy**2 + 1.0/dz**2
    safety_factor = 0.5
    dt_max = safety_factor / (c_max * np.sqrt(inv_dx2))
    n_substeps = max(1, int(np.ceil(dt_out / dt_max)))
    dt = dt_out / n_substeps
    
    print(f"[inhomogeneous] c_max = {c_max:.3f}, n_substeps = {n_substeps}, "
          f"courant = {c_max * dt * np.sqrt(inv_dx2):.3f}")
    
    # ---- Инициализация ----
    u = np.zeros((nt, nx, ny, nz))
    u_curr = np.sin(np.pi * X) * np.sin(np.pi * Y) * np.sin(np.pi * Z)
    u_prev = u_curr.copy()
    u[0] = u_curr.copy()
    
    def laplacian_of(w):
        lap = np.zeros_like(w)
        lap[1:-1,1:-1,1:-1] = (
            (w[2:,1:-1,1:-1] - 2*w[1:-1,1:-1,1:-1] + w[:-2,1:-1,1:-1]) / dx**2 +
            (w[1:-1,2:,1:-1] - 2*w[1:-1,1:-1,1:-1] + w[1:-1,:-2,1:-1]) / dy**2 +
            (w[1:-1,1:-1,2:] - 2*w[1:-1,1:-1,1:-1] + w[1:-1,1:-1,:-2]) / dz**2
        )
        return lap
    
    # Первый шаг через Тейлор (u_t(0) = 0)
    lap0 = laplacian_of(u_curr)
    u_next = u_curr + 0.5 * (c_field * dt)**2 * lap0
    u_prev = u_curr
    u_curr = u_next
    
    # Остальные суб-шаги до t[1]
    for _ in range(n_substeps - 1):
        lap = laplacian_of(u_curr)
        u_next = 2*u_curr - u_prev + (c_field * dt)**2 * lap
        u_prev = u_curr
        u_curr = u_next
    u[1] = u_curr
    
    # ---- Основной цикл ----
    for i in range(2, nt):
        for _ in range(n_substeps):
            lap = laplacian_of(u_curr)
            u_next = 2*u_curr - u_prev + (c_field * dt)**2 * lap
            u_prev = u_curr
            u_curr = u_next
        u[i] = u_curr
        
        # Защита от взрыва
        if not np.all(np.isfinite(u[i])):
            print(f"[inhomogeneous] Взрыв на шаге t[{i}] = {t[i]:.3f}")
            break
    
    return {
        "x": x, "y": y, "z": z, "t": t,
        "u": u, "c": c0, "c_field": c_field,
    }

def get_wave_data(source="homogeneous", **kwargs):
    if source == "homogeneous":
        return generate_homogeneous_wave(**kwargs)
    elif source == "inhomogeneous":
        return generate_inhomogeneous_wave(**kwargs)
    else:
        raise ValueError("Unknown source")
