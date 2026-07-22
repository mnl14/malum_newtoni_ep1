# integrator.py
import numpy as np
from numba import njit
from .config import DT, MAX_TIME, COLLISION_RADIUS
from .physics import acceleration, check_collision, _BODY_POS, _BODY_MASS, _BODY_COLOR_ID


# ---------- Versión compilada (usada por el sweep, sin trayectoria) ----------

@njit(cache=True)
def _derivatives_fast(state, body_pos, body_mass):
    pos = state[:2]
    vel = state[2:]
    acc = acceleration(pos, body_pos, body_mass)
    out = np.empty(4)
    out[0] = vel[0]
    out[1] = vel[1]
    out[2] = acc[0]
    out[3] = acc[1]
    return out


@njit(cache=True)
def _rk4_step_fast(state, dt, body_pos, body_mass):
    k1 = _derivatives_fast(state, body_pos, body_mass)
    k2 = _derivatives_fast(state + 0.5 * dt * k1, body_pos, body_mass)
    k3 = _derivatives_fast(state + 0.5 * dt * k2, body_pos, body_mass)
    k4 = _derivatives_fast(state + dt * k3, body_pos, body_mass)
    return state + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)


@njit(cache=True)
def simulate_fast(x0, y0, dt, max_time, body_pos, body_mass,
                   collision_radius, body_color_id):
    """
    Versión rápida para el sweep: sin trayectoria, sin overhead de Python.
    return: color_id (int) o -1 si no converge.
    """
    state = np.array([x0, y0, 0.0, 0.0])
    n_steps = int(max_time / dt)

    for _ in range(n_steps):
        state = _rk4_step_fast(state, dt, body_pos, body_mass)
        result = check_collision(state[:2], body_pos, collision_radius, body_color_id)
        if result != -2:
            return result

    return -1