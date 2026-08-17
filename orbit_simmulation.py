from constants import DT, relativistic
from physics import compute_accelerations
import numpy as np


def simulate(simulator, bodies, NUM_ITERATIONS):
    """
    Run the simulation and return the FULL time history for every body.

    Positions/Velocities are returned with shape (NUM_ITERATIONS + 1, n_bodies, 2):
    axis 0 = time step, axis 1 = body, axis 2 = x/y.

    Note: this function reads each body's initial position/velocity once at
    the start, but never mutates the Body objects themselves. That keeps
    `simulate` a pure function of its inputs, so calling it again with a
    different integrator always starts from the same initial conditions
    instead of picking up wherever the previous run left off.
    """
    n = len(bodies)
    masses = np.array([body.mass for body in bodies], dtype=float)

    positions = np.zeros((NUM_ITERATIONS + 1, n, 3))
    velocities = np.zeros((NUM_ITERATIONS + 1, n, 3))

    positions[0] = np.array([body.position for body in bodies])
    velocities[0] = np.array([body.velocity for body in bodies])

    for step in range(1, NUM_ITERATIONS + 1):
        # The whole system is advanced together from one consistent
        # snapshot (positions[step - 1]) -- no body ever "sees" another
        # body's already-updated position within the same step.
        positions[step], velocities[step] = simulator(
            positions[step - 1],
            velocities[step - 1],
            masses,
            DT,
            compute_accelerations,
            relativistic,
        )

    return {
        "Positions": positions,
        "Velocities": velocities,
        "Masses": masses,
    }
