from math import sqrt
from Constants import G
def calculate_acceleration(sun_position, earth_position):

    # Calculate Euclidean distance between planets 
    delta_x = sun_position[0] - earth_position[0]
    delta_y = sun_position[1] - earth_position[1]
    distance_earth_sun = sqrt((delta_x)**2 + (delta_y)**2)

    # Calculate acceleration components using a = F/m and gravitational force formula. Normalise units with M = r = 1 and then separating it into components 
    # using sin(theta) = deltax/x and cos(theta) = deltay/y, yielding a_x = G * M * delta_x / r^3. 
    # Approximate G with 4*pi^2
    
    acceleration_x = (G * delta_x) / distance_earth_sun ** 3
    acceleration_y = (G * delta_y) / distance_earth_sun ** 3

    return [acceleration_x, acceleration_y]
