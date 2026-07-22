import numpy as np
from .config import BODIES

START_POS = np.array([b["pos"] for b in BODIES], dtype=np.float64)

# Posiciones finales -- AJUSTAR a los valores reales que quieras mostrar
END_POS = np.array([
    (-1.5, 0),
    ( 0, 0),
    ( 1,5, 0),
])

def body_positions_at(t):
    """t en [0,1]. return: array (3,2) interpolado linealmente."""
    return START_POS + t * (END_POS - START_POS)