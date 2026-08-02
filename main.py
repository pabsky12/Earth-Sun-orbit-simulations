import numpy as np
from BasicOrbitSimulation import simulate
from Integratos import simulators
from analysis import analyse_results
from Constants import NUM_ITERATIONS, DT
from Plotting import orbit_analysis_plot, energy_momentum_analysis
from time import time_ns
from bodies import bodies

# +1 because `simulate` returns the initial state as well as every step
time = np.arange(NUM_ITERATIONS + 1) * DT
body_names = [body.name for body in bodies]

all_energies = {}
all_momentums = {}
computation_times = {}


def main():

    for simulator in simulators:
        start_time = time_ns()
        # `simulate` reads bodies' initial state but never mutates the
        # Body objects, so every integrator here starts from the exact
        # same initial conditions -- runs are directly comparable.
        results = simulate(simulator, bodies, NUM_ITERATIONS)
        end_time = time_ns()
        time_taken = (end_time - start_time) / 1_000_000_000

        positions = results["Positions"]  # shape (steps+1, n_bodies, 2)

        energies, momentums, barycenters = analyse_results(results)

        orbit_analysis_plot(positions, body_names, energies, momentums, barycenters, time, NUM_ITERATIONS, simulator, time_taken)

        all_energies[simulator.__name__] = energies
        all_momentums[simulator.__name__] = momentums
        computation_times[simulator.__name__] = time_taken

    energy_momentum_analysis(all_energies, all_momentums, time, simulators, computation_times)


main()
