# debug_plot.py
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import matplotlib.pyplot as plt
from sim.config import BODIES, GRID_X_RANGE, GRID_Y_RANGE
# ... resto igual
basin = np.load("data/basin_map.npy")  # shape (nx, ny), basin[i,j]: i~x, j~y

# imshow espera [fila, columna]; queremos fila~y, columna~x, con origin='lower'
# así que transponemos (sin flip manual, dejamos que origin='lower' lo maneje)
plt.imshow(
    basin.T,
    origin="lower",
    extent=[*GRID_X_RANGE, *GRID_Y_RANGE],
    cmap="tab10",
)

for body in BODIES:
    x, y = body["pos"]
    plt.scatter(x, y, color="white", edgecolor="black", s=80)

plt.savefig("debug_basin.png", dpi=150)
print("Guardado debug_basin.png")