from math import sqrt, pi
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# Represent location and velocity of sun/earth as 2D vectors

# Sun's position is [0,0] as it is the origin
sun_position = [0, 0]
sun_velocity = [0.0, 0.0]

# Earth's position starts [1.0, 0.0] as it is 1 AU away from the sun in the x-axis

earth_position = [1.0, 0.0]
# Earth's orbital velocity is calculated using the formula v_orbit = sqrt(G*M/r) where 
# r = separation between bodies
# M = mass of the sun
# G = Newton´s gravitational constant, approximately 4pi^2
# r and M are normalised to minimise numerical errors. The formula yields sqrt(4pi^2) == 2pi
# All initial velocity in y as it acts perpendicular to the sun
earth_velocity: float = [0, 2 * pi]

time_step = 0.001

def acceleration_calculation(sun_position, earth_position):

    # Calculate Euclidean distance between planets 
    delta_x = sun_position[0] - earth_position[0]
    delta_y = sun_position[1] - earth_position[1]
    distance_earth_sun = sqrt((delta_x)**2 + (delta_y)**2)

    # Calculate acceleration components using a = F/m and gravitational force formula, the approximations mentioned above for G, M, and r, and then separating it into components 
    # using sin(theta) = deltax/x and cos(theta) = deltay/y, yielding a_x = G * M * delta_x / r^3
    
    acceleration_x = (4 * (pi**2) * delta_x) / distance_earth_sun ** 3
    acceleration_y = (4 * (pi**2) * delta_y) / distance_earth_sun ** 3

    return [acceleration_x, acceleration_y]

def euler_cromer_method(old_position, old_velocity, accelerations):
    new_velocity = [old_velocity[0] + accelerations[0] * time_step, old_velocity[1] + accelerations[1] * time_step]
    new_position = [old_position[0] + new_velocity[0] * time_step, old_position[1] + new_velocity[1]* time_step]

    return new_position, new_velocity

def main():
    sun_position = [0, 0]
    earth_position = [1.0, 0.0]
    earth_velocity: float = [0, 2 * pi]
    num_iterations = 10000
    positions = np.array(earth_position)

    for _ in range(num_iterations):
        accelerations = acceleration_calculation(sun_position, earth_position)

        # Update earth's position and velocity
        earth_position, earth_velocity = euler_cromer_method(earth_position, earth_velocity, accelerations)
        positions = np.vstack([positions, earth_position])
    print("Computation completed")
    pos_x = positions[:, 0]
    pos_y = positions[:, 1]

    fig, ax = plt.subplots()
    blue_patch = mpatches.Patch(color="blue", label="Earth's orbit around the sun")
    ax.legend(handles=[blue_patch])
    plt.plot(pos_x, pos_y)
    plt.plot(0,ls="", marker="o")
    plt.text(0, 0, "Sun")
    plt.title(f"Simulation of Earth's orbit around the Sun using the Euler-Cromer method in {num_iterations} steps")
    plt.xlabel("x-axis (AU)")
    plt.ylabel("y-axis (AU)")
    plt.show()
main()