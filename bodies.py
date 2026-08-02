from math import pi
import numpy as np

class Body:
    def __init__(self, name, mass, position, velocity):
        self.name = name
        self.mass = mass 
        self.position = position
        self.velocity = velocity

    def relative_distance(self, second_body):
        return np.linalg.norm(self.position - second_body.position)
        

earth = Body("Earth", 3.003e-6, np.array([1.0, 0.0]), np.array([0.0, 2*pi]))
sun = Body("Sun", 1, np.array([0.0, 0.0]), np.array([0.0, 0.0]))

# Values for mercury calculated using data from https://lco.global/spacebook/solar-system/mercury/ and using orbital velocity equation 
mercury = Body("Mercury", 1.652e-7, np.array([0.387, 0.0]), np.array([0.0, 10.1]))

bodies = [
    earth,
    sun,
    mercury,
]