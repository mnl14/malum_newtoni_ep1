import math
import numpy as np
from numba import cuda
from .config import G

@cuda.jit(device=True)
def acceleration_gpu(x, y, body_x, body_y, body_mass, n_bodies):
    ax = 0.0
    ay = 0.0
    for k in range(n_bodies):
        dx = body_x[k] - x
        dy = body_y[k] - y
        r2 = dx*dx + dy*dy
        r = math.sqrt(r2)
        if r < 1e-12:
            continue
        r3 = r2 * r
        ax += G * body_mass[k] * dx / r3
        ay += G * body_mass[k] * dy / r3
    return ax, ay


@cuda.jit(device=True)
def check_collision_gpu(x, y, body_x, body_y, collision_radius, n_bodies):
    for k in range(n_bodies):
        dx = x - body_x[k]
        dy = y - body_y[k]
        dist = math.sqrt(dx*dx + dy*dy)
        if dist < collision_radius:
            return k
    return -2