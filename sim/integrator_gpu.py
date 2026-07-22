import math
import numpy as np
from numba import cuda
from .physics_gpu import acceleration_gpu, check_collision_gpu

@cuda.jit
def sweep_kernel(x_min, x_max, y_min, y_max, nx, ny,
                  body_x, body_y, body_mass, n_bodies,
                  collision_radius, dt, max_time, output):
    """
    Un thread CUDA por cada punto (i, j) de la grilla. Cada thread integra
    su propia trayectoria RK4 de forma completamente independiente.
    """
    i, j = cuda.grid(2)
    if i >= nx or j >= ny:
        return

    x = x_min + i * (x_max - x_min) / (nx - 1)
    y = y_min + j * (y_max - y_min) / (ny - 1)
    vx = 0.0
    vy = 0.0

    n_steps = int(max_time / dt)
    result = -1  # no converge por defecto

    for _ in range(n_steps):
        # RK4 manual (sin arrays, todo escalar -- requisito de cuda.jit)
        ax1, ay1 = acceleration_gpu(x, y, body_x, body_y, body_mass, n_bodies)

        x2 = x + 0.5*dt*vx
        y2 = y + 0.5*dt*vy
        vx2 = vx + 0.5*dt*ax1
        vy2 = vy + 0.5*dt*ay1
        ax2, ay2 = acceleration_gpu(x2, y2, body_x, body_y, body_mass, n_bodies)

        x3 = x + 0.5*dt*vx2
        y3 = y + 0.5*dt*vy2
        vx3 = vx + 0.5*dt*ax2
        vy3 = vy + 0.5*dt*ay2
        ax3, ay3 = acceleration_gpu(x3, y3, body_x, body_y, body_mass, n_bodies)

        x4 = x + dt*vx3
        y4 = y + dt*vy3
        vx4 = vx + dt*ax3
        vy4 = vy + dt*ay3
        ax4, ay4 = acceleration_gpu(x4, y4, body_x, body_y, body_mass, n_bodies)

        x  += (dt/6.0) * (vx + 2*vx2 + 2*vx3 + vx4)
        y  += (dt/6.0) * (vy + 2*vy2 + 2*vy3 + vy4)
        vx += (dt/6.0) * (ax1 + 2*ax2 + 2*ax3 + ax4)
        vy += (dt/6.0) * (ay1 + 2*ay2 + 2*ay3 + ay4)

        col = check_collision_gpu(x, y, body_x, body_y, collision_radius, n_bodies)
        if col != -2:
            result = col
            break

    output[i, j] = result