from Constants import G, SUN_MASS, EARTH_MASS, NUM_ITERATIONS
import numpy as np 

def kinetic_energy(mass, velocity):
    return 0.5 * mass * np.dot(velocity, velocity)

def potential_energy(m1, m2, r):
    return -G * m1 * m2 / np.linalg.norm(r)

def total_energy(m1, m2, position, velocity):
    return (
        kinetic_energy(m1, velocity)
        + potential_energy(m1, m2, position)
    )

def angular_momentum(mass, position, velocity):
    return mass * (position[0]*velocity[1] - position[1]*velocity[0])

def analyse_results(results):

    positions = results["Positions"]
    velocities = results["Velocities"]

    # The initial errors are zero because the errors are calculated based on the initial energies and momentums
    energy_errors = [0.0]
    momentum_errors = [0.0]
    expected_energy = total_energy(SUN_MASS, EARTH_MASS, positions[0], velocities[0])
    expencted_momentum = angular_momentum(EARTH_MASS, positions[0], velocities[0])

    for i in range(1, NUM_ITERATIONS):
        position = positions[i]
        velocity = velocities[i]

        energy_error = (
            expected_energy
            - total_energy(SUN_MASS, EARTH_MASS, position, velocity)
        ) / expected_energy

        momentum_error = (
            expencted_momentum
            - angular_momentum(EARTH_MASS, position, velocity)
        ) / expencted_momentum

        energy_errors.append(energy_error)
        momentum_errors.append(momentum_error)

    return np.array(energy_errors).flatten(), np.array(momentum_errors).flatten()