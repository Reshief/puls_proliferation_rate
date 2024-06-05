import argparse
import matplotlib.pyplot as plt
from matplotlib import rc
import argparse
import numpy as np
from matplotlib import colormaps
from mpl_toolkits.axes_grid1 import make_axes_locatable

from tissue_state import read_tissue_state_chemical

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
        default=100,
        type=float,
        help="Maximum time to plot.",
    )

    parser.add_argument(
        "-r",
        "--r_max",
        default=10,
        type=float,
        help="Maximum radius to plot.",
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
    r_max = args.r_max
    rs = args.report_step

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

    position_filter = r_vals <= r_max

    rho_g = full_rho_plot[:, :positions]
    rho_d = full_rho_plot[:, positions:(2*positions)]
    rho_chem = full_rho_plot[:, (2*positions):(3*positions)]

    rho_cell = rho_g+rho_d

    plt.ion()
    report_fig, report_axs = plt.subplots(1, 2)

    im1 = report_axs[0].imshow(rho_cell[:,position_filter], origin='lower', interpolation='bilinear',
                         aspect='auto', cmap=colormaps["inferno"])
    report_axs[0].set_title("Cell density")
    
    divider = make_axes_locatable(report_axs[0])
    cax = divider.append_axes('right', size='5%', pad=0.15)
    report_fig.colorbar(im1, cax=cax, orientation='vertical')

    im2 = report_axs[1].imshow(rho_chem[:,position_filter], origin='lower', interpolation='bilinear',
                         aspect='auto', cmap=colormaps["viridis"])
    report_axs[1].set_title("Chem density")
    
    divider = make_axes_locatable(report_axs[1])
    cax = divider.append_axes('right', size='5%', pad=0.05)

    report_fig.colorbar(im2, cax=cax, orientation='vertical')

    report_fig.savefig(output_prefix+"_full_trajectory_evo.pdf", dpi=200)
