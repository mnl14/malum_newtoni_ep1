# physics.py
import numpy as np
from numba import njit
from .config import G, BODIES, COLLISION_RADIUS

# Pre-extraer datos de BODIES a arrays planos (numba no soporta listas de dicts)
_BODY_POS = np.array([b["pos"] for b in BODIES], dtype=np.float64)      # shape (3, 2)
_BODY_MASS = np.array([b["mass"] for b in BODIES], dtype=np.float64)    # shape (3,)
_BODY_COLOR_ID = np.array([b["color_id"] for b in BODIES], dtype=np.int64)  # shape (3,)


@njit(cache=True)
def acceleration(pos, body_pos, body_mass):
    """
    Calcula la aceleración gravitatoria sobre la masa de prueba en 'pos'.

    pos: array (2,)
    body_pos: array (N, 2)
    body_mass: array (N,)
    return: array (2,)
    """
    acc = np.zeros(2)

    for k in range(body_pos.shape[0]):
        r_vec = body_pos[k] - pos
        r_mag = np.sqrt(r_vec[0]**2 + r_vec[1]**2)

        if r_mag < 1e-12:
            continue

        acc += G * body_mass[k] * r_vec / r_mag**3

    return acc


@njit(cache=True)
def check_collision(pos, body_pos, collision_radius, body_color_id):
    """
    Verifica colisión con algún cuerpo.
    return: color_id (int) del cuerpo colisionado, o -2 si no hay colisión
            (usamos -2 en vez de None porque numba exige tipo de retorno consistente;
             -1 ya está reservado para "no converge" en integrator.py)
    """
    for k in range(body_pos.shape[0]):
        r_vec = pos - body_pos[k]
        dist = np.sqrt(r_vec[0]**2 + r_vec[1]**2)
        if dist < collision_radius:
            return body_color_id[k]

    return -2