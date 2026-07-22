# sweep.py
import numpy as np
from multiprocessing import Pool, cpu_count
from .config import GRID_X_RANGE, GRID_Y_RANGE, GRID_RESOLUTION
from .integrator import simulate

def _worker(args):
    i, j, x, y = args
    result = simulate((x, y))
    return i, j, result

def run_sweep(save_path="data/basin_map.npy", n_workers=None):
    """
    Barre la grilla de condiciones iniciales y calcula el resultado
    (color_id o -1) para cada punto. Guarda un array 2D en save_path.
    """
    xs = np.linspace(GRID_X_RANGE[0], GRID_X_RANGE[1], GRID_RESOLUTION)
    ys = np.linspace(GRID_Y_RANGE[0], GRID_Y_RANGE[1], GRID_RESOLUTION)

    tasks = [
        (i, j, x, y)
        for i, x in enumerate(xs)
        for j, y in enumerate(ys)
    ]

    basin_map = np.zeros((GRID_RESOLUTION, GRID_RESOLUTION), dtype=int)

    n_workers = n_workers or cpu_count()
    total = len(tasks)
    done = 0

    with Pool(n_workers) as pool:
        for i, j, result in pool.imap_unordered(_worker, tasks, chunksize=50):
            basin_map[i, j] = result
            done += 1
            if done % 1000 == 0:
                print(f"{done}/{total} ({100*done/total:.1f}%)")

    np.save(save_path, basin_map)
    print(f"Guardado en {save_path}")
    return basin_map

if __name__ == "__main__":
    run_sweep()