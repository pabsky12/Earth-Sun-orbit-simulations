from Constants import G
import numpy as np


def compute_accelerations(positions, masses):
    """
    Compute the gravitational acceleration on every body from a single,
    consistent snapshot of the system.

    positions : ndarray, shape (n_bodies, 2)   -- one row per body
    masses    : ndarray, shape (n_bodies,)

    Returns ndarray, shape (n_bodies, 2) -- acceleration of each body.

    Crucially, this only ever reads from `positions`; it never mutates
    anything. That means every body's acceleration is computed against the
    *same* instant in time, which is what makes it safe to use inside
    higher-order integrators (velocity verlet, leapfrog, RK4, ...) as well
    as simple ones -- there's no risk of body #2 "seeing" body #1's already
    -updated position.
    """
    n = len(positions)
    accelerations = np.zeros_like(positions, dtype=float)

    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            r_vec = positions[j] - positions[i]
            r = np.linalg.norm(r_vec)
            accelerations[i] += G * masses[j] * r_vec / r ** 3

    return accelerations
