import numpy as np
from numba import cuda
from .config import BODIES, GRID_X_RANGE, GRID_Y_RANGE, GRID_RESOLUTION, DT, MAX_TIME, COLLISION_RADIUS
from .integrator_gpu import sweep_kernel

def run_sweep_gpu(save_path="data/basin_map.npy"):
    nx = ny = GRID_RESOLUTION

    body_x = np.array([b["pos"][0] for b in BODIES], dtype=np.float64)
    body_y = np.array([b["pos"][1] for b in BODIES], dtype=np.float64)
    body_mass = np.array([b["mass"] for b in BODIES], dtype=np.float64)
    n_bodies = len(BODIES)

    d_body_x = cuda.to_device(body_x)
    d_body_y = cuda.to_device(body_y)
    d_body_mass = cuda.to_device(body_mass)
    d_output = cuda.to_device(np.zeros((nx, ny), dtype=np.int64))

    threads_per_block = (16, 16)
    blocks_x = (nx + threads_per_block[0] - 1) // threads_per_block[0]
    blocks_y = (ny + threads_per_block[1] - 1) // threads_per_block[1]
    blocks_per_grid = (blocks_x, blocks_y)

    sweep_kernel[blocks_per_grid, threads_per_block](
        GRID_X_RANGE[0], GRID_X_RANGE[1], GRID_Y_RANGE[0], GRID_Y_RANGE[1],
        nx, ny,
        d_body_x, d_body_y, d_body_mass, n_bodies,
        COLLISION_RADIUS, DT, MAX_TIME,
        d_output
    )
    cuda.synchronize()

    basin_map = d_output.copy_to_host()
    np.save(save_path, basin_map)
    print(f"Guardado en {save_path}, shape={basin_map.shape}")
    return basin_map

if __name__ == "__main__":
    run_sweep_gpu()import numpy as np
from numba import cuda
from .config import BODIES, GRID_X_RANGE, GRID_Y_RANGE, GRID_RESOLUTION, DT, MAX_TIME, COLLISION_RADIUS
from .integrator_gpu import sweep_kernel

def run_sweep_gpu(save_path="data/basin_map.npy"):
    nx = ny = GRID_RESOLUTION

    body_x = np.array([b["pos"][0] for b in BODIES], dtype=np.float64)
    body_y = np.array([b["pos"][1] for b in BODIES], dtype=np.float64)
    body_mass = np.array([b["mass"] for b in BODIES], dtype=np.float64)
    n_bodies = len(BODIES)

    d_body_x = cuda.to_device(body_x)
    d_body_y = cuda.to_device(body_y)
    d_body_mass = cuda.to_device(body_mass)
    d_output = cuda.to_device(np.zeros((nx, ny), dtype=np.int64))

    threads_per_block = (16, 16)
    blocks_x = (nx + threads_per_block[0] - 1) // threads_per_block[0]
    blocks_y = (ny + threads_per_block[1] - 1) // threads_per_block[1]
    blocks_per_grid = (blocks_x, blocks_y)

    sweep_kernel[blocks_per_grid, threads_per_block](
        GRID_X_RANGE[0], GRID_X_RANGE[1], GRID_Y_RANGE[0], GRID_Y_RANGE[1],
        nx, ny,
        d_body_x, d_body_y, d_body_mass, n_bodies,
        COLLISION_RADIUS, DT, MAX_TIME,
        d_output
    )
    cuda.synchronize()

    basin_map = d_output.copy_to_host()
    np.save(save_path, basin_map)
    print(f"Guardado en {save_path}, shape={basin_map.shape}")
    return basin_map

if __name__ == "__main__":
    run_sweep_gpu()