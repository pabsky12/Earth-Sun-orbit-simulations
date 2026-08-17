from math import cbrt
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


def normal_euler(positions, velocities, masses, dt, acceleration_function, relativistic):
    accelerations = acceleration_function(positions, masses, velocities, relativistic)
    new_positions = positions + velocities * dt
    new_velocities = velocities + accelerations * dt
    return new_positions, new_velocities


def euler_cromer(positions, velocities, masses, dt, acceleration_function, relativistic):
    accelerations = acceleration_function(positions, masses, velocities, relativistic)
    new_velocities = velocities + accelerations * dt
    new_positions = positions + new_velocities * dt
    return new_positions, new_velocities


def velocity_verlet(positions, velocities, masses, dt, acceleration_function, relativistic):
    # "Kick-drift-kick" form of velocity verlet:
    #   1. Use the acceleration at the OLD positions to drift to new
    #      positions (a half-implicit position update).
    #   2. Re-evaluate the acceleration at the NEW positions.
    #   3. Average the old and new accelerations to update the velocity.
    #
    # This is a symplectic integrator -- it conserves energy far better
    # than simple Euler methods over long integrations, which is why it's
    # the standard choice for orbit simulations.
    a_old = acceleration_function(positions, masses, velocities, relativistic)
    new_positions = positions + velocities * dt + 0.5 * a_old * dt ** 2
    a_new = acceleration_function(new_positions, masses, velocities, relativistic)
    new_velocities = velocities + 0.5 * (a_old + a_new) * dt
    return new_positions, new_velocities


def leapfrog(positions, velocities, masses, dt, acceleration_function, relativistic):
    # Leapfrog ("kick-drift-kick" variant) is mathematically equivalent to
    # velocity_verlet above -- same trajectory, same energy behaviour, just
    # derived/written slightly differently. It's included separately here
    # so you can see both forms side by side.
    accelerations = acceleration_function(positions, masses, velocities, relativistic)
    velocities_half = velocities + 0.5 * accelerations * dt
    new_positions = positions + velocities_half * dt
    new_accelerations = acceleration_function(new_positions, masses, velocities_half, relativistic)
    new_velocities = velocities_half + 0.5 * new_accelerations * dt
    return new_positions, new_velocities


def RK4(positions, velocities, masses, dt, acceleration_function, relativistic):
    # K1 step
    k1_velocities = velocities
    k1_accelerations = acceleration_function(positions, masses, k1_velocities, relativistic)

    # K2 step
    half_step_positions = positions + .5 * dt * k1_velocities
    k2_velocities = velocities + .5 * dt * k1_accelerations
    k2_accelerations = acceleration_function(half_step_positions, masses, k2_velocities, relativistic)

    # K3 step
    new_half_step_positions = positions + .5 * dt * k2_velocities
    k3_velocities = velocities + .5 * dt * k2_accelerations
    k3_accelerations = acceleration_function(new_half_step_positions, masses, k3_velocities, relativistic)

    # K4 step
    full_step_positions = positions + dt * k3_velocities
    k4_velocities = velocities + dt * k3_accelerations
    k4_accelerations = acceleration_function(full_step_positions, masses, k4_velocities, relativistic)

    # Compute final positions and velocities using RK4 formula
    new_positions = positions + dt * (k1_velocities + 2 * k2_velocities + 2 * k3_velocities + k4_velocities) / 6
    new_velocities = velocities + dt * (k1_accelerations + 2 * k2_accelerations + 2 * k3_accelerations + k4_accelerations) / 6

    return new_positions, new_velocities


def Yoshida(positions, velocities, masses, dt, acceleration_function, relativistic):

    # Define the required constants for Yoshida
    w_0 = - cbrt(2) / (2 - cbrt(2))
    w_1 = 1 / (2 - cbrt(2))
    c_1 = w_1 / 2
    c_2 = (w_0 + w_1) / 2

    # First step
    xi_1 = positions + c_1 * velocities * dt
    vi_1 = velocities + w_1 * acceleration_function(xi_1, masses, velocities, relativistic) * dt

    # Second step
    xi_2 = xi_1 + c_2 * vi_1 * dt
    vi_2 = vi_1 + w_0 * acceleration_function(xi_2, masses, vi_1, relativistic) * dt

    # Third step
    xi_3 = xi_2 + c_2 * vi_2 * dt
    vi_3 = vi_2 + w_1 * acceleration_function(xi_3, masses, vi_2, relativistic) * dt

    # Final step
    x_next = xi_3 + c_1 * vi_3 * dt
    # v_next = vi_3

    return x_next, vi_3

simulators = [normal_euler, euler_cromer, velocity_verlet, leapfrog, RK4, Yoshida]