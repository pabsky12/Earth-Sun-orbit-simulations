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
        

earth = Body("Earth", 3.003e-6, np.array([1.0, 0.0, .03]), np.array([0.0, 2*pi, 0.0]))
sun = Body("Sun", 1, np.array([0.0, 0.0, 0.0]), np.array([0.0, 0.0, 0.0]))
mercury = Body("Mercury", 1.652e-7, np.array([0.387, 0.0, 0.2]), np.array([0.0, 10.1, 0.0]))
venus = Body("Venus", 2.447e-6, np.array([0.72, 0.0, -0.4]), np.array([0.0, 7.4, 0.0]))

bodies = [
    earth,
    sun,
    mercury,
    venus,
]