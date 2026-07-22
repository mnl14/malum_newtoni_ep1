import numpy as np
from numba import cuda
from .config import GRID_X_RANGE, GRID_Y_RANGE, DT, MAX_TIME, COLLISION_RADIUS, BODIES
from .integrator_gpu import sweep_kernel
from .motion import body_positions_at

ANIM_RESOLUTION = 150
N_FRAMES = 72

def run_sweep_animation(save_path="data/basin_frames.npy"):
    nx = ny = ANIM_RESOLUTION
    n_bodies = len(BODIES)
    body_mass = np.array([b["mass"] for b in BODIES], dtype=np.float64)
    frames = np.zeros((N_FRAMES, nx, ny), dtype=np.int64)

    threads_per_block = (16, 16)
    blocks_x = (nx + threads_per_block[0] - 1) // threads_per_block[0]
    blocks_y = (ny + threads_per_block[1] - 1) // threads_per_block[1]
    blocks_per_grid = (blocks_x, blocks_y)

    for f in range(N_FRAMES):
        t = f / (N_FRAMES - 1)
        pos = body_positions_at(t)
        body_x = np.ascontiguousarray(pos[:, 0])
        body_y = np.ascontiguousarray(pos[:, 1])

        d_body_x = cuda.to_device(body_x)
        d_body_y = cuda.to_device(body_y)
        d_body_mass = cuda.to_device(body_mass)
        d_output = cuda.to_device(np.zeros((nx, ny), dtype=np.int64))

        sweep_kernel[blocks_per_grid, threads_per_block](
            GRID_X_RANGE[0], GRID_X_RANGE[1], GRID_Y_RANGE[0], GRID_Y_RANGE[1],
            nx, ny, d_body_x, d_body_y, d_body_mass, n_bodies,
            COLLISION_RADIUS, DT, MAX_TIME, d_output
        )
        cuda.synchronize()
        frames[f] = d_output.copy_to_host()
        print(f"frame {f+1}/{N_FRAMES}")

    np.save(save_path, frames)
    print(f"Guardado {save_path}, shape={frames.shape}")

if __name__ == "__main__":
    run_sweep_animation()