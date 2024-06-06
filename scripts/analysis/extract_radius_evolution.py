#!/usr/bin/env python3

from calendar import c
import math
import matplotlib.pyplot as plt
from matplotlib import rc
import argparse
import numpy as np
import sys
from os.path import dirname, abspath
import scipy.integrate as sp_int


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
        "-e",
        "--end_time",
        default=10.0,
        type=float,
        help="Time at which the analysis should be stopped.",
    )

    parser.add_argument(
        "-o",
        "--output_prefix",
        default="data/",
        help="Output prefix (e.g. path) for the resulting files.",
    )

    args = parser.parse_args()

    # Get the parameters

    # Input trajectory path
    input_path = args.input_trajectory

    # Time at which the analysis should be started and stopped
    begin_time = args.begin_time
    end_time = args.end_time

    # The prefix to prepend to all output paths
    output_prefix = args.output_prefix

    plt.ion()

    full_input = np.loadtxt(input_path, ndmin=2)
    radii = full_input[0]
    radii = radii[1:]
    n = int(len(radii)/3)

    r_vals = radii[:n]

    times = full_input[1:, 0]
    time_filter = (times > begin_time) & (times <= end_time)

    filtered_data = full_input[1:, 1:]
    filtered_data = filtered_data[time_filter]

    times = times[time_filter]
    rho_g = filtered_data[:, :(n)]
    rho_s = filtered_data[:, (n):(2*n)]
    rho_chem = filtered_data[:, (2*n):]
    rho = rho_g+rho_s
    per_system_dim = len(r_vals)

    classic_radius = np.zeros_like(times)
    integral_radius = np.zeros_like(times)
    classic_radius_chem = np.zeros_like(times)
    integral_radius_chem = np.zeros_like(times)

    for i_t in range(len(times)):
        rho_max = np.max(rho[i_t])
        half_max = rho_max / 2.0

        filter = (rho[i_t] >= half_max)
        rmax = np.max(r_vals[filter])

        classic_radius[i_t] = rmax

        integ_dens = 2 * math.pi * r_vals * rho[i_t]

        full_volume = sp_int.simpson(integ_dens, r_vals)

        r_vol = np.sqrt(full_volume/math.pi)
        integral_radius[i_t] = r_vol

        
        rho_max = np.max(rho_chem[i_t])
        half_max = rho_max / 2.0

        filter = (rho_chem[i_t] >= half_max)
        rmax = np.max(r_vals[filter])

        classic_radius_chem[i_t] = rmax

        integ_dens = 2 * math.pi * r_vals * rho_chem[i_t]

        full_volume = sp_int.simpson(integ_dens, r_vals)

        r_vol = np.sqrt(full_volume/math.pi)
        integral_radius_chem[i_t] = r_vol

    with open(output_prefix + "radial_results.dat", "w") as radius_out:
        radius_out.write("{0}\t{1}\t{2}\t{3}\t{4}\n".format(
            "#t[a.u.]", "#r_half_max", "#r_volume", "#r_half_max (chem)", "#r_volume (chem)"))

        for i_t in range(len(times)):
            radius_out.write("{0:.8e}\t{1:.8e}\t{2:.8e}\t{3:.8e}\t{4:.8e}\n".format(
                times[i_t], classic_radius[i_t], integral_radius[i_t], classic_radius_chem[i_t], integral_radius_chem[i_t]))

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
    plt.clf()
    plt.title("Proliferation ratio chemical")
    plt.plot(times, classic_radius_chem, label="half-max radius")
    plt.plot(times, integral_radius_chem, label="circle-volume radius")
    plt.legend()
    plt.savefig(
        output_prefix +
        "radius_evolution_chem.pdf",
        bbox_inches="tight",
    )

    sys.exit(0)
