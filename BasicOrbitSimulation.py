from Constants import DT
from Physics import calculate_acceleration
from math import pi
import numpy as np

earth_position = [1.0, 0.0]
earth_velocity = [0.0, 2*pi]

def simulate(simulator, NUM_ITERATIONS):
    # Distance measured in AU so initial distance is 1.
    earth_position = [1.0, 0.0]
    # Velocity obtained through orbital velocity equation v = sqrt(GM/r) and taking M = r = 1 and G = 4*pi^2
    earth_velocity = [0.0, 2*pi]
    # Fixed sun
    sun_position = [0.0, 0.0]

    # Store all positions and velocities for later analysis
    positions = np.array([earth_position])
    velocities = np.array([earth_velocity])

    for step in range(NUM_ITERATIONS):

        acceleration = calculate_acceleration(sun_position, earth_position)
        # Update the earth's position and velocity
        earth_position, earth_velocity = simulator(
            earth_position,
            earth_velocity,
            acceleration,
            DT
        )

        positions = np.vstack([positions, earth_position])
        velocities = np.vstack([velocities, earth_velocity])

        times = np.arange(NUM_ITERATIONS) * DT
    results = {
        "Positions": positions,
        "Velocities": velocities,
        "Times": times
    }
    return results