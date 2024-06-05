import argparse
import matplotlib.pyplot as plt
from matplotlib import rc
import argparse
import numpy as np

from scripts.simulator.tissue_state import read_tissue_state_chemical

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Script to plot chemical and tissue density distribution over time."
    )

    parser.add_argument(
        "-rs",
        "--report_step",
        default=0.2,
        type=float,
        help="Time delta between plots",
    )

    parser.add_argument(
        "-t",
        "--total_time",
        default=4,
        type=float,
        help="Maximum time to plot.",
    )

    parser.add_argument(
        "-i",
        "--input_dir",
        default="data/",
        help="Path of directory with simulation data.",
    )
    parser.add_argument(
        "-o",
        "--output_prefix",
        default="data/",
        help="Output prefix (e.g. path) for the resulting files.",
    )

    args = parser.parse_args()

    # Get the parameters

    import pathlib

    input_dir = args.input_dir
    output_prefix = args.output_prefix

    t_max = args.total_time
    rs = args.reporting_step

    r_vals, total_rho_history, total_t_history = read_tissue_state_chemical(
        input_dir + "full_trajectory.txt")

    time_filter = total_t_history < t_max
    total_t_history = total_t_history[time_filter]
    total_rho_history = total_rho_history[time_filter]

    curr_t = total_t_history[0]

    full_rho_plot = [total_rho_history[0]]
    full_t_plot = [total_t_history[0]]

    num_steps = (t_max-curr_t)/rs+1

    for i in range(len(total_t_history)):
        if curr_t+rs < total_t_history[i]:
            full_rho_plot.append(total_rho_history[i])
            full_t_plot.append(total_t_history[i])
            curr_t = total_t_history[i]

    full_t_plot = np.array(full_t_plot)
    full_rho_plot = np.array(full_rho_plot)

    positions = len(r_vals)

    


    plt.imshow(Z, origin='lower', interpolation='bilinear')
