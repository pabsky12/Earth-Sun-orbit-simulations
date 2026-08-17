from constants import G, epsilon, C
import numpy as np

def compute_accelerations(positions, masses, velocities, relativistic=False):

    n = len(positions)

    # r_vec[i, j] = vector from body i to body j, shape (n, n, 2)
    r_vec = positions[np.newaxis, :, :] - positions[:, np.newaxis, :]

    # distance[i, j] = |r_vec[i, j]|, shape (n, n)
    r = np.linalg.norm(r_vec, axis=2)

    # Add softening paramemeter to prevent division by zero errors
    r_soft = np.sqrt(r**2 + epsilon**2)

    # avoid division by zero on the diagonal (i == j)
    np.fill_diagonal(r_soft, np.inf)

    # acceleration contribution on i from j, shape (n, n, 2)
    # broadcasting: masses[np.newaxis, :] has shape (1, n) -> (n, n, 1) after reshape
    contributions = G * masses[np.newaxis, :, np.newaxis] * r_vec / r_soft[:, :, np.newaxis] ** 3

    # sum over j (axis=1) to get total acceleration on each body i
    accelerations = np.sum(contributions, axis=1)

    if relativistic:
        accelerations += compute_relativistic_acceleration(positions, velocities, masses)
        
    return accelerations

def compute_relativistic_acceleration(positions, velocities, masses):
    
    r_vec = positions[np.newaxis, :, :] - positions[:, np.newaxis, :]   # r_vec[i,j] = pos[j] - pos[i]
    v_vec = velocities[np.newaxis, :, :] - velocities[:, np.newaxis, :] # v_vec[i,j] = vel[j] - vel[i]

    r = np.linalg.norm(r_vec, axis=2)
    np.fill_diagonal(r, np.inf)  # avoid self-interaction blow-up

    v_sq = np.sum(v_vec ** 2, axis=2)         # |v_vec[i,j]|^2, shape (n, n)
    r_dot_v = np.sum(r_vec * v_vec, axis=2)   # r_vec[i,j] . v_vec[i,j], shape (n, n)

    mass = masses[np.newaxis, :]              # masses[j], broadcasts to (n, n)

    prefactor = (G * mass) / (C**2 * r**3)    # shape (n, n)
    bracket = (((4 * G * mass / r) - v_sq)[:, :, np.newaxis] * r_vec
               + 4 * r_dot_v[:, :, np.newaxis] * v_vec)   # shape (n, n, 3)

    contributions = prefactor[:, :, np.newaxis] * bracket
    return np.sum(contributions, axis=1)      # shape (n, 3): total correction per body i