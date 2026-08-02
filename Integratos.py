import numpy as np

# Every integrator here has the same signature:
#
#   simulator(positions, velocities, masses, dt, acceleration_func)
#
# positions, velocities : ndarray, shape (n_bodies, 2)  -- the WHOLE system
#                          at the current step (not one body at a time!)
# masses                : ndarray, shape (n_bodies,)
# acceleration_func      : function(positions, masses) -> accelerations,
#                          e.g. Physics.compute_accelerations
#
# Operating on the whole system at once (instead of body-by-body) is what
# guarantees every body's update uses a single, consistent snapshot in
# time. It also happens to be exactly what velocity verlet / leapfrog need,
# since they must evaluate the acceleration a second time at the NEW
# positions of every body -- something you can't do if you only have one
# body's position in hand.


def normal_euler(positions, velocities, masses, dt, acceleration_func):
    accelerations = acceleration_func(positions, masses)
    new_positions = positions + velocities * dt
    new_velocities = velocities + accelerations * dt
    return new_positions, new_velocities


def euler_cromer(positions, velocities, masses, dt, acceleration_func):
    accelerations = acceleration_func(positions, masses)
    new_velocities = velocities + accelerations * dt
    new_positions = positions + new_velocities * dt
    return new_positions, new_velocities


def velocity_verlet(positions, velocities, masses, dt, acceleration_func):
    # "Kick-drift-kick" form of velocity verlet:
    #   1. Use the acceleration at the OLD positions to drift to new
    #      positions (a half-implicit position update).
    #   2. Re-evaluate the acceleration at the NEW positions.
    #   3. Average the old and new accelerations to update the velocity.
    #
    # This is a symplectic integrator -- it conserves energy far better
    # than simple Euler methods over long integrations, which is why it's
    # the standard choice for orbit simulations.
    a_old = acceleration_func(positions, masses)
    new_positions = positions + velocities * dt + 0.5 * a_old * dt ** 2
    a_new = acceleration_func(new_positions, masses)
    new_velocities = velocities + 0.5 * (a_old + a_new) * dt
    return new_positions, new_velocities


def leapfrog(positions, velocities, masses, dt, acceleration_func):
    # Leapfrog ("kick-drift-kick" variant) is mathematically equivalent to
    # velocity_verlet above -- same trajectory, same energy behaviour, just
    # derived/written slightly differently. It's included separately here
    # so you can see both forms side by side.
    accelerations = acceleration_func(positions, masses)
    velocities_half = velocities + 0.5 * accelerations * dt
    new_positions = positions + velocities_half * dt
    new_accelerations = acceleration_func(new_positions, masses)
    new_velocities = velocities_half + 0.5 * new_accelerations * dt
    return new_positions, new_velocities


simulators = [normal_euler, euler_cromer, velocity_verlet, leapfrog]
