from Constants import G
import numpy as np


def kinetic_energy(mass, velocity):
    return 0.5 * mass * np.dot(velocity, velocity)


def potential_energy(m1, m2, r_vec):
    return -G * m1 * m2 / np.linalg.norm(r_vec)


def total_energy(masses, positions, velocities):
    """
    positions, velocities : shape (n_bodies, 2) -- one snapshot of the system.
    Sums the kinetic energy of every body plus the potential energy of
    every unique pair (so each pair is only counted once).
    """
    ke = sum(
        kinetic_energy(m, v) for m, v in zip(masses, velocities)
    )

    n = len(masses)
    pe = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            pe += potential_energy(masses[i], masses[j], positions[i] - positions[j])

    return ke + pe


def angular_momentum(masses, positions, velocities):
    """Total angular momentum (z-component, since we're in 2D) of the system."""
    total_L = 0.0
    for m, pos, vel in zip(masses, positions, velocities):
        total_L += m * (pos[0] * vel[1] - pos[1] * vel[0])
    return total_L

def compute_barycenter(masses, positions):
    barycenter = (sum(masses[i] * positions[i] for i in range(len(masses))) / sum(masses))
    
    return np.linalg.norm(barycenter)

def analyse_results(results):
    positions = results["Positions"]   # shape (steps+1, n_bodies, 2)
    velocities = results["Velocities"]
    masses = results["Masses"]

    num_steps = positions.shape[0]

    expected_energy = total_energy(masses, positions[0], velocities[0])
    expected_momentum = angular_momentum(masses, positions[0], velocities[0])
    expected_barycenter = compute_barycenter(masses, positions[0])

    energy_errors = [0.0]
    momentum_errors = [0.0]
    barycenter_errors = [0.0]

    for i in range(1, num_steps):
        energy = total_energy(masses, positions[i], velocities[i])
        momentum = angular_momentum(masses, positions[i], velocities[i])
        barycenter = compute_barycenter(masses, positions[i])

        energy_errors.append((energy - expected_energy) / expected_energy)

        # Guard against a system with exactly zero net angular momentum,
        # where a relative error would be a division by zero.
        if expected_momentum != 0:
            momentum_errors.append((momentum - expected_momentum) / expected_momentum)
        else:
            momentum_errors.append(momentum)

        # Avoid division by zero errors if it uses the same body as both inputs
        if expected_barycenter == 0:
            barycenter_errors.append(barycenter)
        else:
            barycenter_errors.append((barycenter - expected_barycenter) / expected_barycenter)

    return np.array(energy_errors), np.array(momentum_errors), np.array(barycenter_errors)
