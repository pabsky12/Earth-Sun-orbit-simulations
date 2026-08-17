from constants import G
import numpy as np

def kinetic_energy(mass, velocity):
    return 0.5 * mass * np.dot(velocity, velocity)

def potential_energy(m1, m2, r_vec):
    return -G * m1 * m2 / np.linalg.norm(r_vec)

def total_energy(masses, positions, velocities):
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
    total_L = 0.0
    for m, pos, vel in zip(masses, positions, velocities):
        total_L += m * np.cross(pos, vel)
    return total_L

def compute_barycenter(masses, positions):
    barycenter = (masses[:, None] * positions).sum(0) / masses.sum()

    # old code for performance analysis later
    # barycenter = (sum(masses[i] * positions[i] for i in range(len(masses))) / sum(masses))
    
    return barycenter

def analyse_results(results):
    positions = results["Positions"]
    velocities = results["Velocities"]
    masses = results["Masses"]

    num_steps = positions.shape[0]

    expected_energy = total_energy(masses, positions[0], velocities[0])
    expected_angular_momentum = np.linalg.norm(angular_momentum(masses, positions[0], velocities[0]))

    energy_errors = [0.0]
    angular_momenta_errors = [0.0]
    barycenters = [compute_barycenter(masses, positions[0])]

    for i in range(1, num_steps):
        energy = total_energy(masses, positions[i], velocities[i])
        momentum = np.linalg.norm(angular_momentum(masses, positions[i], velocities[i]))
        barycenter = compute_barycenter(masses, positions[i])

        energy_errors.append((energy - expected_energy) / expected_energy)
        angular_momenta_errors.append((momentum - expected_angular_momentum) / expected_angular_momentum)
        barycenters.append(barycenter)


    return np.array(energy_errors), np.array(angular_momenta_errors), np.array(barycenters)