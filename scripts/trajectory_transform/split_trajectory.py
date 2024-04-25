#!/usr/bin/env python3

from audioop import reverse
from os import spawnlp, times_result
import matplotlib.pyplot as plt
from matplotlib import rc
import argparse
import numpy as np
import sys
from os.path import dirname, abspath
import scipy.integrate as sp_int
import scipy.interpolate as sp_interpol
from ddesolver import solve_dde
from theory.meanfield import rate
from theory.prolif_theory_fit import fit_meanfield_model_stable_known, Meanfield_Params
from scipy.optimize import curve_fit

sys.path.append(abspath(dirname(__file__)))

rc("font", **{"family": "sans-serif", "sans-serif": ["Helvetica"]})
# for Palatino and other serif fonts use:
# rc('font',**{'family':'serif','serif':['Palatino']})
rc("text", usetex=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Script to reduce a detailed trajectory to a trajectory with lower sampling rate"
    )
    parser.add_argument(
        "-i",
        "--input_path",
        type=str,
        help="The input path pointing to the relevant data .",
    )

    parser.add_argument(
        "-o",
        "--output_prefix",
        type=str,
        help="Output prefix for the resulting files.",
    )

    parser.add_argument(
        "-rs",
        "--sampling_rate",
        type=float,
        help="Time between frames to be sampled.",
    )

    parser.add_argument(
        "-g",
        "--graphical_mode",
        action='store_true',
        help="Flag to enable rendered outputs.",
    )

    args = parser.parse_args()

    # Get the parameters

    # Input path
    input_path = args.input_path

    # The path to put the resulting file in
    output_prefix = args.output_prefix

    # Time between frames to be output
    sampling_rate = args.sampling_rate

    # Flag if plotting should occur
    graphics_enabled = args.graphical_mode

    full_trajectory = np.loadtxt(input_path)

    if len(full_trajectory) == 0:
        sys.exit(0)

    n = int((len(full_trajectory[0])-1)/2)

    r_vals = full_trajectory[0, 1:1+n]

    traj = full_trajectory[1:]

    times = traj[:, 0]
    last_time = None
    max_g = np.max(traj[:, 1:n+1])
    max_s = np.max(traj[:, n+1:2*n+1])

    max_rho = max_g + max_s

    if graphics_enabled:
        plt.ion()

    for i in range(len(times)):

        t = times[i]

        if last_time is None or last_time+sampling_rate <= t:
            last_time = t

            rho_g = traj[i, 1:n+1]
            rho_s = traj[i, n+1:(2*n+1)]

            if graphics_enabled:
                plt.clf()
                plt.title("State at $t={0:.2f}$".format(t))
                plt.plot(r_vals, rho_s, label=r"$\rho_s$")
                plt.plot(r_vals, rho_g, label=r"$\rho_s$")
                plt.plot(r_vals, rho_g+rho_s, label=r"$\rho$")

                plt.legend()
                plt.ylim((0, max_rho))
                plt.xlim((0, np.max(r_vals)))
                plt.savefig(output_prefix+"_t{0:.3f}_density.pdf".format(t))

            with open(output_prefix+"_t{0:.3f}_density.dat".format(t), "w") as output:
                output.write(
                    "#Data at time t={0:.3f} of trajectory {1}\n".format(t, input_path))
                output.write(
                    "# Format: each line has the following format:\n")
                output.write(
                    "# <r>\\t<rho_g(r)>\\t<rho_s(r)>\n")

                for p in range(len(r_vals)):
                    output.write("{0:.5e}\t{1:.8e}\t{2:.8e}\n".format(
                        r_vals[p], rho_g[p], rho_s[p]))
