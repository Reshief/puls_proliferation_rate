#!/usr/bin/env python3

from audioop import reverse
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
        "--output_path",
        type=str,
        help="Output path for the resulting file.",
    )

    parser.add_argument(
        "-rs",
        "--sampling_rate",
        type=float,
        help="Time between frames to be sampled.",
    )

    args = parser.parse_args()

    # Get the parameters

    # Input path
    input_path = args.input_path

    # The path to put the resulting file in
    output_path = args.output_path

    # Time between frames to be output
    sampling_rate = args.sampling_rate

    with open(output_path, "w") as output:
        output.write(
            "#Subsampled file with time step: {0:.5f}\n".format(sampling_rate))

        with open(input_path, "r") as input:
            last_time = None

            while True:
                line = input.readline()

                if not line:
                    output.flush()
                    break

                if line.startswith("#"):
                    output.write(line)
                else:
                    stripped_line = line.strip()

                    if len(stripped_line) == 0:
                        continue

                    data = [float(x.strip())
                            for x in stripped_line.split("\t")]

                    t = data[0]
                    if t < 0:
                        # Radius information
                        output.write(line)
                    else:
                        if last_time is None or last_time + sampling_rate <= t:
                            output.write(line)
                            last_time = t
                        else:
                            continue
    print("Done")