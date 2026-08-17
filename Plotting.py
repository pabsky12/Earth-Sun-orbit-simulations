from constants import DT
import matplotlib.pyplot as plt
import numpy as np

def orbit_analysis_plot(positions, body_names, energies, momentums, barycenters, time, num_iterations, simulator, time_taken):

    fig = plt.figure(figsize=(20, 8), constrained_layout=True)
    fig.canvas.manager.set_window_title(f"Orbit Simulation using {simulator.__name__} method")

    fig.suptitle(
        f"Orbit Simulation ({simulator.__name__}, with {num_iterations} steps and {DT} years as time step (total {num_iterations * DT} years) \n Time taken: {time_taken:.4f} seconds",
        fontsize=16
    )
    
    energy_ax = plt.subplot2grid((2, 3), (0, 1))
    momentum_ax = plt.subplot2grid((2, 3), (1, 1))
    barycenter_ax = plt.subplot2grid((2, 3), (0, 2), rowspan=2, projection="3d")
    table_ax = plt.subplot2grid((2, 3), (0, 0), rowspan=2)
    table_ax.axis("off")

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

    momentum_ax.plot(time, momentums,label="Relative angular momentum error")
    momentum_ax.axhline(0, color="blue", linestyle="--", label="Expected value")
    momentum_ax.set_title(f"Angular Momentum Variation. Max = {max(abs(momentums)):.2}")
    momentum_ax.set_xlabel("Time (years)")
    momentum_ax.set_ylabel(r"$(L-L_0)/L_0$")
    momentum_ax.grid(True)
    momentum_ax.legend()

    # ==========================
    # Barycenter
    # ==========================    

    barycenter_ax.plot(barycenters[:, 0], barycenters[:, 1], barycenters[:, 2], label="Position of the barycenter with time")
    barycenter_ax.axhline(0, color="blue", linestyle="--", label="Expected value")
    barycenter_ax.set_title(f"Barycenter Position Variation.")
    barycenter_ax.set_xlabel("X-axis")
    barycenter_ax.set_ylabel("Y-axis")
    barycenter_ax.grid(True)
    barycenter_ax.legend()

    # =========================
    # Numbers
    # =========================

    numbers_table_data = [
        [f"{np.max(np.abs(energies)):.2e}",
        f"{np.mean(energies):.2e}",
        f"{np.sqrt(np.mean(energies**2)):.2e}"],
        [f"{np.max(np.abs(momentums)):.2e}",
        f"{np.mean(momentums):.2e}",
        f"{np.sqrt(np.mean(momentums**2)):.2e}"],
        [f"{np.max(np.abs(barycenters)):.2e}",
        f"{np.mean(barycenters):.2e}",
        f"{np.sqrt(np.mean(barycenters**2)):.2e}"],
    ]

    numbers_table = table_ax.table(
        cellText=numbers_table_data,
        rowLabels=["Energy error", "Momentum Error", "Barycenter error"],
        colLabels=["Maximum", "Mean", "RMS"],
        cellLoc="center",
        loc="center",
    )

    return fig


def energy_momentum_analysis(energies, momentums, barycenters, time, simulators, computation_times):

    fig = plt.figure(figsize=(20, 10), constrained_layout=True)
    fig.canvas.manager.set_window_title("Energy and Angular Momentum Analysis")
    fig.suptitle("Comparison of Energy and Angular Momentum Conservation", fontsize=16)

    # ==========================
    # Layout
    # ==========================
    energy_table_ax = plt.subplot2grid((3, 3), (0, 0))
    momentum_table_ax = plt.subplot2grid((3, 3), (1, 0), projection="3d")
    barycenter_table_ax = plt.subplot2grid((3, 3), (2, 0))

    energy_ax = plt.subplot2grid((3, 3), (0, 1))
    momentum_ax = plt.subplot2grid((3, 3), (1, 1))
    barycenter_ax = plt.subplot2grid((3, 3), (0, 2), rowspan=3)

    energy_table_ax.axis("off")
    momentum_table_ax.axis("off")
    barycenter_table_ax.axis("off")

    energy_ax.set_title("Relative Energy Error")
    energy_ax.set_xlabel("Time (years)")
    energy_ax.set_ylabel(r"$(E-E_0)/E_0$")
    energy_ax.grid(True)

    momentum_ax.set_title("Relative Angular Momentum Error")
    momentum_ax.set_xlabel("Time (years)")
    momentum_ax.set_ylabel(r"$(L-L_0)/L_0$")
    momentum_ax.grid(True)

    barycenter_ax.set_title("Position of the barycenter with time")
    barycenter_ax.set_xlabel("X-axis")
    barycenter_ax.set_ylabel("Y-axis")
    barycenter_ax.grid(True)

    energy_ax.axhline(0, color="black", linestyle="--", linewidth=1.5, label="Expected")
    momentum_ax.axhline(0, color="black", linestyle="--", linewidth=1.5, label="Expected")
    barycenter_ax.axhline(0, color="black", linestyle="--", linewidth=1.5, label="Expected")

    energy_table_data = []
    momentum_table_data = []
    barycenter_table_data = []
    rows = []

    colors = plt.cm.tab10.colors

    for i, simulator in enumerate(simulators):
        simulator_name = simulator.__name__
        rows.append(f"{simulator_name}\n")

        colour = colors[i % len(colors)]

        energy_error = np.abs(energies[simulator_name])
        momentum_error = np.abs(momentums[simulator_name])
        bc = barycenters[simulator_name]
        barycenter_error = np.linalg.norm(bc - bc[0], axis=1)   # shape (num_steps,)

        energy_ax.plot(time, energies[simulator_name], color=colour, linewidth=2, label=simulator_name)
        momentum_ax.plot(time, momentums[simulator_name], color=colour, linewidth=2, label=simulator_name)
        barycenter_ax.plot(barycenters[simulator_name][:, 0], barycenters[simulator_name][:, 1], barycenters[simulator_name][:, 2], color=colour, linewidth=2, label=simulator_name)

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

        barycenter_table_data.append([
            f"{np.max(barycenter_error):.2e}",
            f"{np.mean(barycenter_error):.2e}",
            f"{np.sqrt(np.mean(barycenter_error**2)):.2e}"
        ])

    energy_ax.legend()
    momentum_ax.legend()
    barycenter_ax.legend()

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

    barycenter_table = barycenter_table_ax.table(
        cellText=barycenter_table_data,
        rowLabels=rows,
        colLabels=["Maximum", "Mean", "RMS"],
        cellLoc="center",
        loc="center"
    )

    for table in (energy_table, momentum_table, barycenter_table):
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.7)

    energy_table_ax.set_title("Energy Error Summary", fontsize=12, pad=15)
    momentum_table_ax.set_title("Angular Momentum Error Summary", fontsize=12, pad=15)
    barycenter_table_ax.set_title("Barycenter Error Summary", fontsize=12, pad=15)

    plt.show()
