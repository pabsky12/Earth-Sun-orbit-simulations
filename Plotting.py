from Constants import DT
import matplotlib.pyplot as plt
import numpy as np


def orbit_analysis_plot(positions, body_names, energies, momentums, barycenters, time, num_iterations, simulator, time_taken):
    """
    positions : shape (steps+1, n_bodies, 2)
    body_names : list of names, one per body (same order as positions' axis 1)
    """

    fig = plt.figure(figsize=(20, 8), constrained_layout=True)
    fig.canvas.manager.set_window_title(f"Orbit Simulation using {simulator.__name__} method")

    fig.suptitle(
        f"Orbit Simulation ({simulator.__name__}, with {num_iterations} steps and {DT} years as time step (total {num_iterations * DT} years) \n Time taken: {time_taken:.4f} seconds",
        fontsize=16
    )

    orbit_ax = plt.subplot2grid((2, 2), (0, 0), rowspan=2)
    energy_ax = plt.subplot2grid((2, 2), (0, 1))
    momentum_ax = plt.subplot2grid((2, 2), (1, 1))
    # barycenter_ax = plt.subplot2grid((2, 2), (1, 0))

    # ==========================
    # Orbit -- one line per body
    # ==========================

    for i, name in enumerate(body_names):
        orbit_ax.plot(positions[:, i, 0], positions[:, i, 1], label=name)
        # Mark each body's starting point
        orbit_ax.plot(positions[0, i, 0], positions[0, i, 1], marker="o", markersize=6)

    orbit_ax.set_title("Orbit")
    orbit_ax.set_xlabel("x (AU)")
    orbit_ax.set_ylabel("y (AU)")
    orbit_ax.set_aspect("equal", adjustable="box")
    orbit_ax.grid(True)
    orbit_ax.legend()

    # ==========================
    # Energy
    # ==========================

    energy_ax.plot(time, energies, label="Relative energy error.")
    energy_ax.axhline(0, color="red", linestyle="--", label="Expected value")
    energy_ax.set_title(f"Energy Variation, Max = {max(abs(energies)):.2}")
    energy_ax.set_xlabel("Time (years)")
    energy_ax.set_ylabel(r"$(E-E_0)/E_0$")
    energy_ax.grid(True)
    energy_ax.legend()

    # ==========================
    # Angular Momentum
    # ==========================

    momentum_ax.plot(time, momentums, label="Relative angular momentum error")
    momentum_ax.axhline(0, color="blue", linestyle="--", label="Expected value")
    momentum_ax.set_title(f"Angular Momentum Variation. Max = {max(abs(momentums)):.2}")
    momentum_ax.set_xlabel("Time (years)")
    momentum_ax.set_ylabel(r"$(L-L_0)/L_0$")
    momentum_ax.grid(True)
    momentum_ax.legend()

    # ==========================
    # Barycenter
    # ==========================    

    # barycenter_ax.plot(time, barycenters, label="Relative barycenter position error")
    # barycenter_ax.axhline(0, color="blue", linestyle="--", label="Expected value")
    # barycenter_ax.set_title(f"Barycenter Position Variation. Max = {max(abs(barycenters)):.2}")
    # barycenter_ax.set_xlabel("Time (years)")
    # barycenter_ax.set_ylabel("Relative barycenter error")
    # barycenter_ax.grid(True)
    # barycenter_ax.legend()

    plt.show()


def energy_momentum_analysis(energies, momentums, time, simulators, computation_times):

    fig = plt.figure(figsize=(15, 8), constrained_layout=True)
    fig.canvas.manager.set_window_title("Energy and Angular Momentum Analysis")
    fig.suptitle("Comparison of Energy and Angular Momentum Conservation", fontsize=16)

    # ==========================
    # Layout
    # (fixed: previously time_table_ax and momentum_ax both occupied
    #  cell (1,1) and silently overlapped each other. The unused time
    #  table has been dropped in favour of showing each integrator's
    #  computation time directly in the row labels below.)
    # ==========================

    energy_table_ax = plt.subplot2grid((2, 2), (0, 0))
    momentum_table_ax = plt.subplot2grid((2, 2), (1, 0))
    energy_ax = plt.subplot2grid((2, 2), (0, 1))
    momentum_ax = plt.subplot2grid((2, 2), (1, 1))

    energy_table_ax.axis("off")
    momentum_table_ax.axis("off")

    energy_ax.set_title("Relative Energy Error")
    energy_ax.set_xlabel("Time (years)")
    energy_ax.set_ylabel(r"$(E-E_0)/E_0$")
    energy_ax.grid(True)

    momentum_ax.set_title("Relative Angular Momentum Error")
    momentum_ax.set_xlabel("Time (years)")
    momentum_ax.set_ylabel(r"$(L-L_0)/L_0$")
    momentum_ax.grid(True)

    energy_ax.axhline(0, color="black", linestyle="--", linewidth=1.5, label="Expected")
    momentum_ax.axhline(0, color="black", linestyle="--", linewidth=1.5, label="Expected")

    energy_table_data = []
    momentum_table_data = []
    rows = []

    colors = plt.cm.tab10.colors

    for i, simulator in enumerate(simulators):
        simulator_name = simulator.__name__
        rows.append(f"{simulator_name}\n({computation_times[simulator_name]:.3f}s)")

        colour = colors[i % len(colors)]

        energy_error = np.abs(energies[simulator_name])
        momentum_error = np.abs(momentums[simulator_name])

        energy_ax.plot(time, energies[simulator_name], color=colour, linewidth=2, label=simulator_name)
        momentum_ax.plot(time, momentums[simulator_name], color=colour, linewidth=2, label=simulator_name)

        energy_table_data.append([
            f"{np.max(energy_error):.2e}",
            f"{np.mean(energy_error):.2e}",
            f"{np.sqrt(np.mean(energy_error**2)):.2e}"
        ])

        momentum_table_data.append([
            f"{np.max(momentum_error):.2e}",
            f"{np.mean(momentum_error):.2e}",
            f"{np.sqrt(np.mean(momentum_error**2)):.2e}"
        ])

    energy_ax.legend()
    momentum_ax.legend()

    energy_table = energy_table_ax.table(
        cellText=energy_table_data,
        rowLabels=rows,
        colLabels=["Maximum", "Mean", "RMS"],
        cellLoc="center",
        loc="center"
    )

    momentum_table = momentum_table_ax.table(
        cellText=momentum_table_data,
        rowLabels=rows,
        colLabels=["Maximum", "Mean", "RMS"],
        cellLoc="center",
        loc="center"
    )

    for table in (energy_table, momentum_table):
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.7)

    energy_table_ax.set_title("Energy Error Summary", fontsize=12, pad=15)
    momentum_table_ax.set_title("Angular Momentum Error Summary", fontsize=12, pad=15)

    plt.show()
