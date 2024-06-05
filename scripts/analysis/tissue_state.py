import numpy as np


def write_tissue_state(path: str, radial_positions, history_density, history_time):
    num_positions = len(radial_positions)
    num_times = len(history_time)

    history_density_g = history_density[:, :num_positions]
    history_density_d = history_density[:, num_positions:]

    with open(path, "w") as output:
        output.write("# Tissue state at time t={0:.3e}\n".format(
            np.max(history_time)))
        output.write(
            "# Number of radial positions n_r={0:d}\n".format(num_positions))
        output.write(
            "# Number of entries in history n_t={0:d}\n".format(num_times))

        output.write("-1")
        # For growing population density
        for i in range(num_positions):
            output.write("\t{0:.6e}".format(radial_positions[i]))
        # For dividing population density
        for i in range(num_positions):
            output.write("\t{0:.6e}".format(radial_positions[i]))
        output.write("\n")

        for t in range(num_times):
            output.write("{0:.6e}".format(history_time[t]))
            # For growing population density
            for p in range(num_positions):
                output.write("\t{0:.6e}".format(history_density_g[t][p]))
            # For dividing population density
            for p in range(num_positions):
                output.write("\t{0:.6e}".format(history_density_d[t][p]))
            output.write("\n")


def read_tissue_state(path: str):
    # Read whole matrix as data:
    data = np.loadtxt(path, ndmin=2)

    num_positions = int((len(data[0])-1)/2)
    # num_times = len(data)-1

    history_times = data[1:, 0].copy()
    history_density = data[1:, 1:].copy()
    r_vals = data[0, 1:num_positions+1].copy()

    return (r_vals, history_density, history_times)



def write_tissue_state_chemical(path: str, radial_positions, history_density, history_time):
    num_positions = len(radial_positions)
    num_times = len(history_time)

    history_density_g = history_density[:, :num_positions]
    history_density_d = history_density[:, num_positions:2*num_positions]
    history_density_chem = history_density[:, 2*num_positions:]

    with open(path, "w") as output:
        output.write("# Tissue state at time t={0:.3e}\n".format(
            np.max(history_time)))
        output.write(
            "# Number of radial positions n_r={0:d}\n".format(num_positions))
        output.write(
            "# Number of entries in history n_t={0:d}\n".format(num_times))
        output.write(
            "# Contains third density of diffusing chemical\n")

        output.write("-1")
        # For growing population density
        for i in range(num_positions):
            output.write("\t{0:.6e}".format(radial_positions[i]))
        # For dividing population density
        for i in range(num_positions):
            output.write("\t{0:.6e}".format(radial_positions[i]))
        # For chemical density
        for i in range(num_positions):
            output.write("\t{0:.6e}".format(radial_positions[i]))
        output.write("\n")

        for t in range(num_times):
            output.write("{0:.6e}".format(history_time[t]))
            # For growing population density
            for p in range(num_positions):
                output.write("\t{0:.6e}".format(history_density_g[t][p]))
            # For dividing population density
            for p in range(num_positions):
                output.write("\t{0:.6e}".format(history_density_d[t][p]))
            # For chemical density
            for p in range(num_positions):
                output.write("\t{0:.6e}".format(history_density_chem[t][p]))
            output.write("\n")


def read_tissue_state_chemical(path: str):
    # Read whole matrix as data:
    data = np.loadtxt(path, ndmin=2)

    # With chemical, we have three times the data
    num_positions = int((len(data[0])-1)/3)
    # num_times = len(data)-1

    history_times = data[1:, 0].copy()
    history_density = data[1:, 1:].copy()
    r_vals = data[0, 1:num_positions+1].copy()

    return (r_vals, history_density, history_times)
