#!/usr/bin/env python3

from audioop import reverse
from calendar import c
import math
from os import times_result
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
from theory.prolif_theory_fit import fit_meanfield_model, Meanfield_Params
from scipy.optimize import curve_fit


sys.path.append(abspath(dirname(__file__)))

rc("font", **{"family": "sans-serif", "sans-serif": ["Helvetica"]})
# for Palatino and other serif fonts use:
# rc('font',**{'family':'serif','serif':['Palatino']})
rc("text", usetex=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Script to calculate the proliferation profile and fit the meanfield (MF) parameters."
    )
    parser.add_argument(
        "-i",
        "--input_trajectory",
        type=str,
        help="The input path pointing to the full trajectory to be analyzed.",
    )

    parser.add_argument(
        "-b",
        "--begin_time",
        default=0.0,
        type=float,
        help="Time at which the analysis should be started.",
    )

    parser.add_argument(
        "-o",
        "--output_prefix",
        default="data/",
        help="Output prefix (e.g. path) for the resulting files.",
    )

    parser.add_argument(
        "-g",
        "--graphical_ui",
        default=False,
        action="store_true",
        help="Flag to enable visual output of intermediate results.",
    )

    args = parser.parse_args()

    # Get the parameters

    # Input trajectory path
    input_path = args.input_trajectory

    # Time at which the analysis should be started
    begin_time = args.begin_time

    # The prefix to prepend to all output paths
    output_prefix = args.output_prefix

    # Flag to enable graphical output
    visual_enabled = args.graphical_ui

    if visual_enabled:
        plt.ion()

    full_input = np.loadtxt(input_path, ndmin=2)
    radii = full_input[0]
    radii = radii[1:]
    n = int(len(radii)/2)

    r_vals = radii[:n]

    times = full_input[:, 0]
    time_filter = times > begin_time

    filtered_data = full_input[:, 1:]
    filtered_data = filtered_data[time_filter]

    times = times[time_filter]
    rho_g = filtered_data[:, :n]
    rho_s = filtered_data[:, n:]
    rho = rho_g+rho_s
    per_system_dim = len(r_vals)

    classic_radius = np.zeros_like(times)
    integral_radius = np.zeros_like(times)

    for i_t in range(len(times)):
        t_max = rho[i_t][0]
        half_max = t_max / 2.0

        filter = (rho[i_t] >= half_max)
        rmax = np.max(r_vals[filter])

        classic_radius[i_t] = rmax

        integ_dens = 2 * math.pi * r_vals * rho[i_t]

        full_volume = sp_int.simps(integ_dens, r_vals)

        r_vol = np.sqrt(full_volume/math.pi)
        integral_radius[i_t] = r_vol

    with open(output_prefix + "radial_results.dat", "w") as radius_out:
        radius_out.write("{0}\t{1}\t{2}\n".format(
            "#t[a.u.]", "#r_half_center", "#r_volume"))

        for i_t in range(len(times)):
            radius_out.write("{0:.8e}\t{1:.8e}\t{2:.8e}\n".format(
                times[i_t], classic_radius[i_t], integral_radius[i_t]))

    if visual_enabled:
        plt.clf()
        plt.title("Proliferation ratio")
        plt.plot(times, classic_radius, label="half-max radius")
        plt.plot(times, integral_radius, label="circle-volume radius")
        plt.legend()
        plt.savefig(
            output_prefix +
            "radius_evolution.pdf",
            bbox_inches="tight",
        )

    sys.exit(0)
