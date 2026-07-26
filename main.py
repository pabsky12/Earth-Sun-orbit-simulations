from BasicOrbitSimulation import simulate
from Integratos import simulators
from analysis import analyse_results
from Constants import NUM_ITERATIONS
from Plotting import orbit_analysis_plot
import matplotlib.pyplot as plt

def main():
    for simulator in simulators:
        results = simulate(
            simulator,
            NUM_ITERATIONS
        )

        # Plot the orbit
        positions = results["Positions"]
        pos_x = positions[:, 0]
        pos_y = positions[:, 1]

        # Plot variation of energy and momemtum with time
        
        time = results["Times"]
        energies, momentums = analyse_results(results)
        print("Finished")

        orbit_analysis_plot(pos_x, pos_y, energies, momentums, time, NUM_ITERATIONS, simulator)

main()