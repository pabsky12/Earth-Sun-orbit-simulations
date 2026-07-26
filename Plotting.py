from Constants import DT
import matplotlib.pyplot as plt

def orbit_analysis_plot(pos_x, pos_y,energies, momentums, time, num_iterations, simulator):

    # Create the figure
    fig = plt.figure(figsize=(14, 8), constrained_layout=True)
    fig.canvas.manager.set_window_title("Earth Orbit Simulation")

    # Overall title
    fig.suptitle(
        f"Earth Orbit Simulation ({simulator.__name__}, {num_iterations} steps, {num_iterations * DT} years)",
        fontsize=16
    )

    # Layout
    orbit_ax = plt.subplot2grid((2, 2), (0, 0), rowspan=2)
    energy_ax = plt.subplot2grid((2, 2), (0, 1))
    momentum_ax = plt.subplot2grid((2, 2), (1, 1))

    # ==========================
    # Orbit
    # ==========================

    orbit_ax.plot(pos_x, pos_y, label="Earth")

    orbit_ax.plot(
        0,
        0,
        marker="o",
        markersize=10,
        color="orange",
        label="Sun"
    )

    orbit_ax.set_title("Orbit")
    orbit_ax.set_xlabel("x (AU)")
    orbit_ax.set_ylabel("y (AU)")
    orbit_ax.grid(True)
    orbit_ax.legend()

    # ==========================
    # Energy
    # ==========================

    energy_ax.plot(
        time,
        energies,
        label="Relative energy error"
    )

    energy_ax.axhline(
        0,
        color="red",
        linestyle="--",
        label="Expected value"
    )

    energy_ax.set_title("Energy Variation")
    energy_ax.set_xlabel("Time (years)")
    energy_ax.set_ylabel(r"$(E-E_0)/E_0$")
    energy_ax.grid(True)
    energy_ax.legend()

    # ==========================
    # Angular Momentum
    # ==========================

    momentum_ax.plot(
        time,
        momentums,
        label="Relative angular momentum error"
    )

    momentum_ax.axhline(
        0,
        color="blue",
        linestyle="--",
        label="Expected value"
    )

    momentum_ax.set_title("Angular Momentum Variation")
    momentum_ax.set_xlabel("Time (years)")
    momentum_ax.set_ylabel(r"$(L-L_0)/L_0$")
    momentum_ax.grid(True)
    momentum_ax.legend()

    plt.show()