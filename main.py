import numpy as np
import matplotlib as plt
from orbit_simmulation import simulate
from integrators import simulators
from analysis import analyse_results
from constants import NUM_ITERATIONS, DT, stride, SIMULATE
from plotting import orbit_analysis_plot, energy_momentum_analysis
from time import time_ns
from bodies import bodies
from plotting_3D import plot_orbits, add_barycenter_trace

# +1 because `simulate` returns the initial state as well as every step
time = np.arange(NUM_ITERATIONS + 1) * DT
body_names = [body.name for body in bodies]

def main():

    all_energies = {}
    all_momentums = {}
    all_barycenters = {}
    computation_times = {}
    
    for simulator in simulators:
        start_time = time_ns()
        # `simulate` reads bodies' initial state but never mutates the
        # Body objects, so every integrator here starts from the exact
        # same initial conditions -- runs are directly comparable.
        results = simulate(simulator, bodies, NUM_ITERATIONS)
        end_time = time_ns()
        time_taken = (end_time - start_time) / 1_000_000_000

        positions = results["Positions"]  # shape (steps+1, n_bodies, 2)

        energies, angular_momenta, barycenters = analyse_results(results)

        fig = plot_orbits(positions, body_names, simulator.__name__, NUM_ITERATIONS, stride, SIMULATE)
        fig = add_barycenter_trace(fig, barycenters)
        fig.show()

        orbit_analysis_plot(positions, body_names, energies, angular_momenta, barycenters, time, NUM_ITERATIONS, simulator, time_taken)

        all_energies[simulator.__name__] = energies
        all_momentums[simulator.__name__] = angular_momenta
        all_barycenters[simulator.__name__] = barycenters
        computation_times[simulator.__name__] = time_taken

    energy_momentum_analysis(all_energies, all_momentums, all_barycenters, time, simulators, computation_times)

    plt.show()

if __name__ == "__main__":
    main()
