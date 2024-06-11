#!/usr/bin/env python3

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
from tissue_state import read_tissue_state_chemical, write_tissue_state_chemical


sys.path.append(abspath(dirname(__file__)))

rc("font", **{"family": "sans-serif", "sans-serif": ["Helvetica"]})
# for Palatino and other serif fonts use:
# rc('font',**{'family':'serif','serif':['Palatino']})
rc("text", usetex=True)

# Routine to initialize the required arrays, simplifies extension of the domain


def get_r_vals(total_discrete_steps, total_radius):
    # Generate the discrete r positions
    l_r_vals = np.linspace(
        start=0,
        stop=total_radius,
        num=total_discrete_steps + 1,
    )

    return l_r_vals


# calculate the gradient in r with reflecting boundary conditions at r=0 and
# the dirichlet condition at the outer edge (hence we toss that gradient value)


def get_nabla_reflecting(r, f):
    res = f * 0.0
    dr = r[1] - r[0]

    # get stencil for discret nabla
    r_left = f[0:-2]
    r_right = f[2:]

    res[1:-1] = (r_right - r_left) / (2 * dr)

    res[0] = 0  # we assume reflecting boundary conditions at zero
    # res[-1] = (-f[-2]) / (2 * dr)
    res[-1] = 0  # we force this to zero, we do not need the gradient

    return res


# Calculate the second derivative incorporating the vanishing gradient at zero and
# the forced zero at the outer edge


def get_laplace(r, f):
    res = f * 0.0
    dr = r[1] - r[0]
    dr2 = dr * dr

    # Correct the second order deriative at r=0 that needs to be calculated differently
    res[0] = (f[1] - f[0]) / (dr2)

    # get stencil elements for discrete laplace
    r_left = f[0:-2]
    r_center = f[1:-1]
    r_right = f[2:]

    res[1:-1] = (r_right - 2 * r_center + r_left) / (dr2)

    res[-1] = 0

    return res

    # return get_nabla(r, get_nabla_reflecting(r, f))


def dfk_dde_model(
    Y,
    t,
    Delta_t_s,
    D_base,
    D_slope,
    b,
    rho_0,
    c,
    r_vals,
    chem_rate,
    chem_decay_rate,
    nabla_rho_t,
    D_chem,
    apoptosis_start_time=-1,
    apoptosis_start_rate=1.0,
    apoptosis_decay_rate=0.0,
):
    per_system_dim = len(r_vals)
    # Get current state
    current = Y(t)
    rho_g = current[:per_system_dim]
    rho_s = current[per_system_dim:2*per_system_dim]
    rho_chem = current[2*per_system_dim:]
    # Get delayed state
    delayed = Y(t - Delta_t_s)
    rho_g_d = delayed[:per_system_dim]
    rho_s_d = delayed[per_system_dim:2*per_system_dim]

    # Get gradient and laplace for differential calculation
    gradient_g = get_nabla_reflecting(r_vals, rho_g)
    laplace_g = get_laplace(r_vals, rho_g)

    nabla_chemical = get_nabla_reflecting(r_vals, rho_chem)
    laplace_chem = get_laplace(r_vals, rho_chem)

    def enter_s(p_b, p_rho_0, p_c, p_rho_g, p_rho_s):
        rho_total = p_rho_g + p_rho_s
        return (
            p_b
            * np.power(np.clip(1.0 - rho_total / p_rho_0, a_min=0.0, a_max=1.0), p_c)
            * p_rho_g
        )

    enter_division = enter_s(b, rho_0, c, rho_g, rho_s)

    # Before time t=0 there was no proliferation onset according to our model
    if t - Delta_t_s >= 0:
        enter_division_d = enter_s(b, rho_0, c, rho_g_d, rho_s_d)
    else:
        enter_division_d = rho_s * 0.0

    diff_t_g = np.zeros_like(rho_g)
    diff_t_s = np.zeros_like(rho_s)
    diff_t_chem = np.zeros_like(rho_chem)

    # calculate d_t for all but the r=0 position

    # Calculate spatially dependent diffusion
    D = D_slope * rho_chem + D_base
    gradient_D = get_nabla_reflecting(r_vals, D)

    diff_t_g[1:] = (
        (D[1:]) * (laplace_g[1:] + gradient_g[1:] / r_vals[1:])
        # Deal with D being non-constant
        + (gradient_D[1:]) * (gradient_g[1:])
        # The g-state cells entering the s-phase
        - enter_division[1:]
        # The previous s-phase cells now proliferating
        + 2 * enter_division_d[1:]
    )

    dr = r_vals[1] - r_vals[0]
    dr2 = dr**2
    # Deal with the pole at r=0 which cancels with the reflecting boundary condition gradient=0
    diff_t_g[0] = (
        # (D[0]) * (laplace_g[0] + 0.0)
        # FIXED: The laplace-term at r=0 has been fixed
        (D[0]) * (4. * (rho_g[1]-rho_g[0])/dr2)
        # Deal with D being non-constant
        + (gradient_D[0]) * (gradient_g[0])
        # The g-state cells entering the s-phase
        - enter_division[0]
        # The previous s-phase cells now proliferating
        + 2 * enter_division_d[0]
    )

    # chalculate chemical diffusion at all but origin
    rho_cell_total = rho_g + rho_s
    nabla_full_cell = get_nabla_reflecting(r_vals, rho_cell_total)

    # TODO: Make minimum gradient configurable
    spawn_filter = np.abs(nabla_full_cell) > nabla_rho_t

    # Determine the full gradient as a sort of measure for the pressure gradient

    # TODO: Add constant dissipation rate for chemical
    diff_t_chem[1:] = (
        (D_chem) * (laplace_chem[1:] + nabla_chemical[1:] / r_vals[1:])
        # Cells secreting the chemical
    )

    # Deal with the pole at r=0 which cancels with the reflecting boundary condition gradient=0
    # FIXME: maybe there is a term that needs to be plugged instead of 0.0 (gradient/r \to ? for r\to 0)
    diff_t_chem[0] = (
        # (D_chem) * (laplace_chem[0] + 0.0)
        # FIXED: The laplace-term at r=0 has been fixed
        (D_chem) * (4. * (rho_chem[1]-rho_chem[0])/dr2)
        # Cells secreting the chemical
    )

    # Only spawn chemical where filter has determined
    diff_t_chem[spawn_filter] += chem_rate * rho_cell_total[spawn_filter]
    # Make chemical disappear over time
    diff_t_chem -= chem_decay_rate * rho_chem

    # Apply apoptosis if configured
    if apoptosis_start_time >= 0 and t > apoptosis_start_time:
        # Apoptosis rate decays over time
        curr_apoptosis_rate = apoptosis_start_rate * np.exp(
            -(t - apoptosis_start_time) * apoptosis_decay_rate
        )
        # Add apoptosis on top of other influences
        diff_t_g -= rho_g * curr_apoptosis_rate

    diff_t_s = enter_division - enter_division_d

    # No change at last position
    diff_t_g[-1] = 0.0
    diff_t_s[-1] = 0.0
    diff_t_chem[-1] = 0.0

    return np.concatenate(
        (
            diff_t_g,
            diff_t_s,
            diff_t_chem
        )
    )


def dfk_dde_model_1D(
    Y,
    t,
    Delta_t_s,
    D_base,
    D_slope,
    b,
    rho_0,
    c,
    r_vals,
    chem_rate,
    chem_decay_rate,
    nabla_rho_t,
    D_chem,
    apoptosis_start_time=-1,
    apoptosis_start_rate=1.0,
    apoptosis_decay_rate=0.0,
):
    per_system_dim = len(r_vals)
    # Get current state
    current = Y(t)
    rho_g = current[:per_system_dim]
    rho_s = current[per_system_dim:2*per_system_dim]
    rho_chem = current[2*per_system_dim:]
    # Get delayed state
    delayed = Y(t - Delta_t_s)
    rho_g_d = delayed[:per_system_dim]
    rho_s_d = delayed[per_system_dim:2*per_system_dim]

    # Get gradient and laplace for differential calculation
    gradient_g = get_nabla_reflecting(r_vals, rho_g)
    laplace_g = get_laplace(r_vals, rho_g)

    nabla_chemical = get_nabla_reflecting(r_vals, rho_chem)
    laplace_chem = get_laplace(r_vals, rho_chem)

    def enter_s(p_b, p_rho_0, p_c, p_rho_g, p_rho_s):
        rho_total = p_rho_g + p_rho_s
        return (
            p_b
            * np.power(np.clip(1.0 - rho_total / p_rho_0, a_min=0.0, a_max=1.0), p_c)
            * p_rho_g
        )

    enter_division = enter_s(b, rho_0, c, rho_g, rho_s)

    # Before time t=0 there was no proliferation onset according to our model
    if t - Delta_t_s >= 0:
        enter_division_d = enter_s(b, rho_0, c, rho_g_d, rho_s_d)
    else:
        enter_division_d = rho_s * 0.0

    diff_t_g = np.zeros_like(rho_g)
    diff_t_s = np.zeros_like(rho_s)
    diff_t_chem = np.zeros_like(rho_chem)

    # calculate d_t for all but the r=0 position

    # Calculate spatially dependent diffusion
    D = D_slope * rho_chem + D_base
    gradient_D = get_nabla_reflecting(r_vals, D)

    diff_t_g[1:] = (
        (D[1:]) * (laplace_g[1:])
        # Deal with D being non-constant
        + (gradient_D[1:]) * (gradient_g[1:])
        # The g-state cells entering the s-phase
        - enter_division[1:]
        # The previous s-phase cells now proliferating
        + 2 * enter_division_d[1:]
    )

    dr = r_vals[1] - r_vals[0]
    dr2 = dr**2
    # Deal with the pole at r=0 which cancels with the reflecting boundary condition gradient=0
    diff_t_g[0] = (
        # (D[0]) * (laplace_g[0] + 0.0)
        # FIXED: The laplace-term at r=0 has been fixed
        (D[0]) * (2. * (rho_g[1]-rho_g[0])/dr2)
        # Deal with D being non-constant
        + (gradient_D[0]) * (gradient_g[0])
        # The g-state cells entering the s-phase
        - enter_division[0]
        # The previous s-phase cells now proliferating
        + 2 * enter_division_d[0]
    )

    # chalculate chemical diffusion at all but origin
    """rho_cell_total = rho_g + rho_s
    diff_t_chem[1:] = (
        (D_chem) * (laplace_chem[1:])
        # Cells secreting the chemical
        + chem_rate * rho_cell_total[1:]
    )

    # Deal with the pole at r=0 which cancels with the reflecting boundary condition gradient=0
    # FIXME: maybe there is a term that needs to be plugged instead of 0.0 (gradient/r \to ? for r\to 0)
    diff_t_chem[0] = (
        # (D_chem) * (laplace_chem[0] + 0.0)
        # FIXED: The laplace-term at r=0 has been fixed
        (D_chem) * (2. * (rho_chem[1]-rho_chem[0])/dr2)
        # Cells secreting the chemical
        + chem_rate * rho_cell_total[0]
    )"""

    # chalculate chemical diffusion at all but origin
    rho_cell_total = rho_g + rho_s
    nabla_full_cell = get_nabla_reflecting(r_vals, rho_cell_total)

    # TODO: Make minimum gradient configurable
    spawn_filter = np.abs(nabla_full_cell) > nabla_rho_t

    # TODO: Add constant dissipation rate for chemical
    diff_t_chem[1:] = (
        (D_chem) * (laplace_chem[1:])
        # Cells secreting the chemical
    )

    # Deal with the pole at r=0 which cancels with the reflecting boundary condition gradient=0
    # FIXME: maybe there is a term that needs to be plugged instead of 0.0 (gradient/r \to ? for r\to 0)
    diff_t_chem[0] = (
        # (D_chem) * (laplace_chem[0] + 0.0)
        # FIXED: The laplace-term at r=0 has been fixed
        (D_chem) * (2. * (rho_chem[1]-rho_chem[0])/dr2)
        # Cells secreting the chemical
    )

    # Only spawn chemical where filter has determined
    diff_t_chem[spawn_filter] += chem_rate * rho_cell_total[spawn_filter]
    # Make chemical disappear over time
    diff_t_chem -= chem_decay_rate * rho_chem

    # Apply apoptosis if configured
    if apoptosis_start_time >= 0 and t > apoptosis_start_time:
        # Apoptosis rate decays over time
        curr_apoptosis_rate = apoptosis_start_rate * np.exp(
            -(t - apoptosis_start_time) * apoptosis_decay_rate
        )
        # Add apoptosis on top of other influences
        diff_t_g -= rho_g * curr_apoptosis_rate

    diff_t_s = enter_division - enter_division_d

    # No change at last position
    diff_t_g[-1] = 0.0
    diff_t_s[-1] = 0.0
    diff_t_chem[-1] = 0.0

    return np.concatenate(
        (
            diff_t_g,
            diff_t_s,
            diff_t_chem
        )
    )


def dfk_dde_model_1D_edgemult(
    Y,
    t,
    Delta_t_s,
    D_base,
    D_slope,
    b,
    rho_0,
    c,
    r_vals,
    chem_rate,
    chem_decay_rate,
    nabla_rho_t,
    D_chem,
    apoptosis_start_time=-1,
    apoptosis_start_rate=1.0,
    apoptosis_decay_rate=0.0,
):
    per_system_dim = len(r_vals)
    # Get current state
    current = Y(t)
    rho_g = current[:per_system_dim]
    rho_s = current[per_system_dim:2*per_system_dim]
    rho_chem = current[2*per_system_dim:]
    # Get delayed state
    delayed = Y(t - Delta_t_s)
    rho_g_d = delayed[:per_system_dim]
    rho_s_d = delayed[per_system_dim:2*per_system_dim]

    # Get gradient and laplace for differential calculation
    gradient_g = get_nabla_reflecting(r_vals, rho_g)
    laplace_g = get_laplace(r_vals, rho_g)

    nabla_chemical = get_nabla_reflecting(r_vals, rho_chem)
    laplace_chem = get_laplace(r_vals, rho_chem)

    def enter_s(p_b, p_rho_0, p_c, p_rho_g, p_rho_s):
        rho_total = p_rho_g + p_rho_s
        return (
            p_b
            * np.power(np.clip(1.0 - rho_total / p_rho_0, a_min=0.0, a_max=1.0), p_c)
            * p_rho_g
        )

    enter_division = enter_s(b, rho_0, c, rho_g, rho_s)

    # Before time t=0 there was no proliferation onset according to our model
    if t - Delta_t_s >= 0:
        enter_division_d = enter_s(b, rho_0, c, rho_g_d, rho_s_d)
    else:
        enter_division_d = rho_s * 0.0

    diff_t_g = np.zeros_like(rho_g)
    diff_t_s = np.zeros_like(rho_s)
    diff_t_chem = np.zeros_like(rho_chem)

    # calculate d_t for all but the r=0 position

    # Calculate spatially dependent diffusion
    D = D_slope * rho_chem + D_base
    gradient_D = get_nabla_reflecting(r_vals, D)

    diff_t_g[1:] = (
        (D[1:]) * (laplace_g[1:])
        # Deal with D being non-constant
        + (gradient_D[1:]) * (gradient_g[1:])
        # The g-state cells entering the s-phase
        - enter_division[1:]
        # The previous s-phase cells now proliferating
        + 2 * enter_division_d[1:]
    )

    dr = r_vals[1] - r_vals[0]
    dr2 = dr**2
    # Deal with the pole at r=0 which cancels with the reflecting boundary condition gradient=0
    diff_t_g[0] = (
        # (D[0]) * (laplace_g[0] + 0.0)
        # FIXED: The laplace-term at r=0 has been fixed
        (D[0]) * (2. * (rho_g[1]-rho_g[0])/dr2)
        # Deal with D being non-constant
        + (gradient_D[0]) * (gradient_g[0])
        # The g-state cells entering the s-phase
        - enter_division[0]
        # The previous s-phase cells now proliferating
        + 2 * enter_division_d[0]
    )

    # chalculate chemical diffusion at all but origin
    """rho_cell_total = rho_g + rho_s
    diff_t_chem[1:] = (
        (D_chem) * (laplace_chem[1:])
        # Cells secreting the chemical
        + chem_rate * rho_cell_total[1:]
    )

    # Deal with the pole at r=0 which cancels with the reflecting boundary condition gradient=0
    # FIXME: maybe there is a term that needs to be plugged instead of 0.0 (gradient/r \to ? for r\to 0)
    diff_t_chem[0] = (
        # (D_chem) * (laplace_chem[0] + 0.0)
        # FIXED: The laplace-term at r=0 has been fixed
        (D_chem) * (2. * (rho_chem[1]-rho_chem[0])/dr2)
        # Cells secreting the chemical
        + chem_rate * rho_cell_total[0]
    )"""

    # chalculate chemical diffusion at all but origin
    rho_cell_total = rho_g + rho_s
    nabla_full_cell = get_nabla_reflecting(r_vals, rho_cell_total)

    # TODO: Make minimum gradient configurable
    spawn_filter = np.abs(nabla_full_cell) > nabla_rho_t
    spawn_multiplier = rho_cell_total * np.abs(nabla_full_cell)

    # TODO: Add constant dissipation rate for chemical
    diff_t_chem[1:] = (
        (D_chem) * (laplace_chem[1:])
        # Cells secreting the chemical
    )

    # Deal with the pole at r=0 which cancels with the reflecting boundary condition gradient=0
    # FIXME: maybe there is a term that needs to be plugged instead of 0.0 (gradient/r \to ? for r\to 0)
    diff_t_chem[0] = (
        # (D_chem) * (laplace_chem[0] + 0.0)
        # FIXED: The laplace-term at r=0 has been fixed
        (D_chem) * (2. * (rho_chem[1]-rho_chem[0])/dr2)
        # Cells secreting the chemical
    )

    # Only spawn chemical where filter has determined
    diff_t_chem[spawn_filter] += chem_rate * spawn_multiplier[spawn_filter]
    # Make chemical disappear over time
    diff_t_chem -= chem_decay_rate * rho_chem

    # Apply apoptosis if configured
    if apoptosis_start_time >= 0 and t > apoptosis_start_time:
        # Apoptosis rate decays over time
        curr_apoptosis_rate = apoptosis_start_rate * np.exp(
            -(t - apoptosis_start_time) * apoptosis_decay_rate
        )
        # Add apoptosis on top of other influences
        diff_t_g -= rho_g * curr_apoptosis_rate

    diff_t_s = enter_division - enter_division_d

    # No change at last position
    diff_t_g[-1] = 0.0
    diff_t_s[-1] = 0.0
    diff_t_chem[-1] = 0.0

    return np.concatenate(
        (
            diff_t_g,
            diff_t_s,
            diff_t_chem
        )
    )


# Generate initial history conditions from a list of times and associated density data
def get_history_init_condition(t_values, rho_data):
    def history_init(t):
        if t >= t_values[-1]:
            return rho_data[-1]

        # We cannot go further than the oldest state
        if t <= t_values[0]:
            return rho_data[0]

        for i in range(1, len(t_values)):
            if t >= t_values[-(i + 1)]:
                t_prev = t_values[-(i + 1)]
                t_next = t_values[-i]

                t_diff = t_next - t_prev
                return (
                    rho_data[-(i + 1)] * (t_next - t) +
                    rho_data[-i] * (t - t_prev)
                ) / t_diff

    return history_init


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Script to simulate time progression of the front of a tissue using a split s-phase/g-phase density and delay differential equations (DDE)."
        "Based on python ddeint package to simplify development."
    )
    parser.add_argument(
        "-b",
        "--growth_constant",
        default=1,
        type=float,
        help="The constant with which the intrinsic cell growth is scaled.",
    )

    parser.add_argument(
        "-b_chem",
        "--chemical_creation_rate",
        default=1,
        type=float,
        help="The rate at which cells secrete the chemical.",
    )

    parser.add_argument(
        "-a_chem",
        "--chemical_decay_rate",
        default=1,
        type=float,
        help="The rate at which the chemical evaporates/decays.",
    )

    parser.add_argument(
        "-nabla_t",
        "--nabla_rho_threshold_chemical",
        default=0.1,
        type=float,
        help="The minimum absolute derivative of density for cells to produce the chemical.",
    )

    parser.add_argument(
        "-c",
        "--curvature_exponent",
        default=1,
        type=float,
        help="The exponent to the growth term according to our findings from proliferation theory.",
    )

    parser.add_argument(
        "-D_0",
        "--diff_constant_base",
        default=1,
        type=float,
        help="The effective diffusion constant scale base value with which the tissue spreads.",
    )
    parser.add_argument(
        "-D_c",
        "--diff_constant_slope",
        default=1,
        type=float,
        help="The slope of the effective diffusion constant relative to chemical concentration with which the tissue spreads.",
    )

    parser.add_argument(
        "-D_chem",
        "--diff_constant_chemical",
        default=1,
        type=float,
        help="The effective diffusion constant of the chemical.",
    )

    parser.add_argument(
        "-r_i",
        "--radius_initial",
        default=1,
        type=float,
        help="The initial radius up to which the density should be initialized to non-zero.",
    )

    parser.add_argument(
        "-r",
        "--radius",
        default=200,
        type=float,
        help="The simulation radius to start with. will be extended as soon as the propagation reaches the outer edge.",
    )

    parser.add_argument(
        "-dr",
        "--radius_step",
        default=0.1,
        type=float,
        help="The discretization step to apply to the radius dimension.",
    )

    parser.add_argument(
        "-d_i",
        "--density_initial",
        default=1,
        type=float,
        help="The initial density the density should be initialized with within r_i.",
    )
    parser.add_argument(
        "-d_c",
        "--density_critical",
        default=1.0,
        type=float,
        help="The critical density \\rho_0 that the calculation should be performed with.",
    )

    parser.add_argument(
        "-dt",
        "--timestep",
        default=0.002,
        type=float,
        help="The maximum time step delta used for numerical simulation.",
    )

    parser.add_argument(
        "-ts",
        "--Delta_ts",
        default=0.1,
        type=float,
        help="The time spent in the s-phase by a cell before proliferation.",
    )

    parser.add_argument(
        "--smooth",
        default=False,
        action="store_true",
        help="Flag to trigger smooth seeding",
    )

    parser.add_argument(
        "-rs",
        "--report_step",
        default=0.2,
        type=float,
        help="The time delta between reports.",
    )

    parser.add_argument(
        "-t",
        "--total_time",
        default=4,
        type=float,
        help="The total time to simulate.",
    )

    parser.add_argument(
        "-o",
        "--output_prefix",
        default="data/",
        help="Output prefix (e.g. path) for the resulting files.",
    )

    parser.add_argument(
        "-i",
        "--input_config",
        default=None,
        help="Path to an input tissue configuration. Some settings will be ignored and read from the input configuration instead like the initial radius and the discretisation step",
    )

    parser.add_argument(
        "-g",
        "--graphical_ui",
        default=False,
        action="store_true",
        help="Flag to enable visual output of intermediate results.",
    )

    parser.add_argument(
        "-a_s",
        "--apoptosis_start",
        default=-1.0,
        type=float,
        help="Time at which apoptosis should kick in.",
    )

    parser.add_argument(
        "-a_r",
        "--apoptosis_rate",
        default=1.0,
        type=float,
        help="Initial apoptosis rate.",
    )

    parser.add_argument(
        "-a_d",
        "--apoptosis_decay_rate",
        default=0.0,
        type=float,
        help="Rate at which the apoptosis rate decays. This is an exponential decay rate over time.",
    )

    args = parser.parse_args()

    # Get the parameters

    import pathlib

    # The coefficient to scale the growth term b(1-\rho)^c*\rho
    b = args.growth_constant

    # The exponent to modify the growth term according to proliferation ratio curvature in b(1-\rho)^c*\rho
    c = args.curvature_exponent

    # The constant diffusion coefficient in radial direction
    D_base = args.diff_constant_base
    D_slope = args.diff_constant_slope

    # The initial cutoff radius
    R = args.radius

    # the initial initialization radius
    R_i = args.radius_initial

    # The initial density
    d_i = args.density_initial

    # The radial discretization step
    dr = args.radius_step

    # Number of simulation steps to perform
    total_sim_time = args.total_time

    # Interval for output dumps
    report_step = args.report_step

    # The discrete time step between each iteration
    dt = args.timestep

    # The discrete time duration of cell division
    Delta_t_s = args.Delta_ts

    # The prefix to prepend to all output paths
    output_prefix = args.output_prefix

    # Ensure the output path exists

    output_test = pathlib.Path(output_prefix+"test.txt")
    parent_dir = output_test.parents[0]
    parent_dir.mkdir(exist_ok=True, parents=True)

    # The prefix to prepend to all output paths
    input_path = args.input_config

    # Flag to enable graphical output
    visual_enabled = args.graphical_ui

    # Homeostatic density
    rho_0 = args.density_critical

    # Apoptosis related parameters
    # The time at which apoptosis starts to kick in
    param_apoptosis_start_time = args.apoptosis_start

    # Initial rate of apoptosis
    param_apoptosis_start_rate = args.apoptosis_rate

    # Time rate at which apoptosis decays
    param_apoptosis_decay_rate = args.apoptosis_decay_rate

    # Counter to keep track of how many extensions we have performed
    num_extensions = 1

    # Flag to smoothe out the edge of the initial seeding range
    flag_smooth_seeding = args.smooth

    # Chemical parameters
    chem_rate = args.chemical_creation_rate
    D_chem = args.diff_constant_chemical
    chem_decay_rate = args.chemical_decay_rate
    nabla_rho_t = args.nabla_rho_threshold_chemical

    # Enforce that there are enough positions for the initialization:
    R = max(R_i + 2 * dr, R)

    # The number of discrete steps within one radius R
    num_discretization_steps = int(np.ceil(R / dr))

    # The number of positions to initialize with d_i
    initialization_radius_limit = int(np.ceil(R_i / dr))

    total_radius = R

    # Routine for initialization of initial conditions:
    def rect_init_condition(t):
        global r_vals, R_i, d_i
        rho_g = np.zeros_like(r_vals)
        rho_s = np.zeros_like(r_vals)
        rho_chem = np.zeros_like(r_vals)

        # initialize the density distribution
        for i in range(len(r_vals)):
            if r_vals[i] <= R_i:
                rho_g[i] = d_i

        return np.concatenate((rho_g, rho_s, rho_chem))

    def smooth_init_condition(t):
        global r_vals, R_i, d_i
        rho_g = np.zeros_like(r_vals)
        rho_s = np.zeros_like(r_vals)
        rho_chem = np.zeros_like(r_vals)

        lower_cutoff = R_i * 0.7
        upper_cutoff = R_i

        dist = upper_cutoff-lower_cutoff

        rho_g[r_vals <= lower_cutoff] = d_i

        r_filter = (r_vals > lower_cutoff) & (r_vals <= upper_cutoff)
        rho_g[r_filter] = d_i * (1 - ((r_vals[r_filter]-lower_cutoff)/dist)
                                 ** 2) * ((upper_cutoff-r_vals[r_filter])/dist)**2

        return np.concatenate((rho_g, rho_s, rho_chem))

    if input_path is not None:
        r_vals, total_rho_history, total_t_history = read_tissue_state_chemical(
            input_path)

        total_radius = np.max(r_vals)
        num_discretization_steps = len(r_vals)-1

        # Build the initial condition from the provided history
        init_condition = get_history_init_condition(
            total_t_history, total_rho_history
        )

        last_time = np.max(total_t_history)
    else:
        r_vals = get_r_vals(num_discretization_steps, total_radius)

        ratio_density_max = 1.5 * rho_0

        total_t_history = None
        total_rho_history = None

        if flag_smooth_seeding:
            init_condition = smooth_init_condition
        else:
            init_condition = rect_init_condition

        last_time = 0.0

    if dt > dr*dr/4.:
        dt = dr*dr/4.
        print(
            "Reduced the time step to {0:.3e} to increase stability".format(dt))

    final_sim_time = last_time+total_sim_time

    with open(output_prefix + "config.txt", "w") as config_out:
        config_out.write("{0}\t=\t{1:.5e}\n".format("D_base (D_0)", D_base))
        config_out.write("{0}\t=\t{1:.5e}\n".format("D_slope (D_c)", D_slope))
        config_out.write("{0}\t=\t{1:.5e}\n".format("b", b))
        config_out.write("{0}\t=\t{1:.5e}\n".format("Delta_t_s", Delta_t_s))
        config_out.write("{0}\t=\t{1:.5e}\n".format("Curvature (c)", c))
        config_out.write("{0}\t=\t{1:.5e}\n".format(
            "Simulated Radius limit (R)", R))
        config_out.write("{0}\t=\t{1:.5e}\n".format(
            "Initialized Radius (R_i)", R_i))
        config_out.write("{0}\t=\t{1:.5e}\n".format(
            "Initialized density (d_i)", d_i))
        config_out.write(
            "{0}\t=\t{1:.5e}\n".format("Homeostatic density (rho_0)", rho_0)
        )
        config_out.write("{0}\t=\t{1:.5e}\n".format(
            "Simulation time step (dt)", dt))
        config_out.write(
            "{0}\t=\t{1:.5e}\n".format(
                "Simulation time total (t)", total_sim_time)
        )
        config_out.write(
            "{0}\t=\t{1:.5e}\n".format(
                "Simulation time start (t_0)", last_time)
        )
        config_out.write(
            "{0}\t=\t{1:.5e}\n".format(
                "Simulation time end (t_e)", final_sim_time)
        )
        config_out.write(
            "{0}\t=\t{1:.5e}\n".format("Radial discretisation step (dr)", dr)
        )
        config_out.write(
            "{0}\t=\t{1:.5e}\n".format(
                "Number of radial positions (n)", num_discretization_steps + 1
            )
        )
        config_out.write(
            "{0}\t=\t{1:.5e}\n".format("Report time delta (rs)", report_step)
        )

        config_out.write(
            "{0}\t=\t{1:.5e}\n".format("b[chemical] (b_chem)", chem_rate)
        )
        config_out.write(
            "{0}\t=\t{1:.5e}\n".format("a[chemical] (a_chem)", chem_decay_rate)
        )
        config_out.write(
            "{0}\t=\t{1:.5e}\n".format(
                "nabla_rho_thresh[chemical] (nabla_rho_t)", nabla_rho_t)
        )

        config_out.write(
            "{0}\t=\t{1:.5e}\n".format("D[chemical] (D_chem)", D_chem)
        )

        config_out.write("{0}\t=\t{1}\n".format(
            "Output prefix", output_prefix))

        if input_path is not None:
            config_out.write("{0}\t=\t{1}\n".format(
                "Input configuration", input_path))

        config_out.write(
            "{0}\t=\t{1:.5e}\n".format(
                "apoptosis_start_time", param_apoptosis_start_time
            )
        )
        config_out.write(
            "{0}\t=\t{1:.5e}\n".format(
                "apoptosis_start_rate", param_apoptosis_start_rate
            )
        )
        config_out.write(
            "{0}\t=\t{1:.5e}\n".format(
                "apoptosis_decay_rate", param_apoptosis_decay_rate
            )
        )

    if visual_enabled:
        plt.ion()
        report_fig, report_axs = plt.subplots(2)

    if report_step is None:
        report_step = (total_sim_time-last_time) / 100.0

    # Make sure, sufficient resolution is used for the sampling rate
    dt = min(dt, report_step / 10.0)

    report_iteration = 0

    print("Simulating a total of t=", total_sim_time)
    print("Starting at t_0=", last_time)
    print("With a report interval /\\t_report=", report_step)

    rho_spread = np.linspace(0.0, rho_0, num=40)
    rho_delta = rho_spread[1]
    rho_spread = rho_spread[1:]

    rho_statistics_mean = np.zeros_like(rho_spread)
    rho_statistics_mean_sq = np.zeros_like(rho_spread)
    rho_statistics_hits = np.zeros_like(rho_spread)
    upper_rho_spread_index = len(rho_spread)
    ratio_density_max = 1.1 * rho_0

    per_system_dim = len(r_vals)

    def progressBar(current_value, max_value, suffix=''):
        bar_length = 50
        filled_up_Length = int(
            round(bar_length * current_value / float(max_value)))
        percentage = round(100.0 * current_value/float(max_value), 1)
        bar = '=' * filled_up_Length + '-' * (bar_length - filled_up_Length)
        sys.stdout.write('[%s] %s%s ...%s\r' % (bar, percentage, '%', suffix))
        sys.stdout.flush()

    with open(output_prefix + "full_trajectory.txt", "w") as trajectory_out:
        n = len(r_vals)
        trajectory_out.write("# Full trajectory of simulation\n")
        trajectory_out.write("# Format:\n")
        trajectory_out.write(
            "# The simulation has a total of n={0} positions for each density\n".format(
                n
            )
        )
        trajectory_out.write(
            "# First line: -1 then tab-separated list of r coordinates once for \\rho_g(r), once for \\rho_s(r) and once for \\rho_chem(r). (total of 3n+1 entries)\n"
        )
        trajectory_out.write(
            "# Then after that each line has 3n+1 entries until the end of the file with the format:\n"
        )
        trajectory_out.write(
            "# <time t>\t<n tab-separated values of \\rho_g(r)>\t<n tab-separated values of \\rho_s(r)>\t<n tab-separated values of \\rho_chem(r)>\n"
        )
        trajectory_out.write("# n={0}\n".format(n))

        def write_entry(out, time, rho_g, rho_s, rho_chem):
            out.write("{0:1.8e}".format(time))
            n = len(rho_g)
            for i in range(n):
                out.write("\t{0:.8e}".format(rho_g[i]))
            for i in range(n):
                out.write("\t{0:.8e}".format(rho_s[i]))
            for i in range(n):
                out.write("\t{0:.8e}".format(rho_chem[i]))
            out.write("\n")

        write_entry(trajectory_out, -1, r_vals, r_vals, r_vals)
        trajectory_out.flush()

        first_time = last_time
        progressBar(0., final_sim_time-first_time)

        while last_time + 1.0001*dt < final_sim_time:
            # Calculate the next report step end time and required simulation time
            next_end_time = min(final_sim_time, last_time + report_step)
            sim_delta = next_end_time - last_time

            # print("simulate from ", last_time, " to ", next_end_time)

            # Sample the time interval to be simulated
            t_data = np.linspace(
                last_time, next_end_time, num=max(
                    int(np.ceil(sim_delta / dt)), 2)
            )

            # Do the simulation
            res_data = solve_dde(
                # dfk_dde_model_1D,
                dfk_dde_model_1D_edgemult,
                init_condition,
                t_data,
                fargs=(
                    Delta_t_s,
                    D_base,
                    D_slope,
                    b,
                    rho_0,
                    c,
                    r_vals,
                    chem_rate,
                    chem_decay_rate,
                    nabla_rho_t,
                    D_chem,
                    param_apoptosis_start_time,
                    param_apoptosis_start_rate,
                    param_apoptosis_decay_rate,
                ),
            )

            progressBar(last_time - first_time, final_sim_time-first_time,
                        suffix="({:.2f}/{:.2f})".format(last_time, final_sim_time))

            last_time = next_end_time

            # build the history back:
            if total_rho_history is None:
                total_t_history = t_data
                total_rho_history = res_data
            else:
                total_t_history = np.append(total_t_history, t_data, axis=0)
                total_rho_history = np.append(
                    total_rho_history, res_data, axis=0)

                # Accumulate statistics for proliferation ratio
                for step in range(len(res_data)):
                    rho_g = res_data[step][:per_system_dim]
                    rho_s = res_data[step][per_system_dim:2*per_system_dim]
                    rho_chem = res_data[step][2*per_system_dim:]

                    write_entry(trajectory_out,
                                t_data[step], rho_g, rho_s, rho_chem)
                    rho = rho_g + rho_s

                    filter_mask = rho > 1e-3
                    ratios = rho_s[filter_mask] / rho[filter_mask]
                    base_rho = rho[filter_mask]

            # Filter the history and truncate entries to free up memory
            history_time_filter = total_t_history >= last_time - 2 * Delta_t_s

            # copy the filtered arrays to drop references to original large array
            total_t_history = total_t_history[history_time_filter].copy()
            total_rho_history = total_rho_history[history_time_filter].copy()

            # print(len(total_t_history), "entries in history")

            # Rebuild the new initial condition
            init_condition = get_history_init_condition(
                total_t_history, total_rho_history
            )

            c_data = res_data[-1]

            rho_g = c_data[:per_system_dim]
            rho_s = c_data[per_system_dim:2*per_system_dim]
            rho_chem = c_data[2*per_system_dim:]

            rho = rho_g + rho_s

            if visual_enabled:
                report_axs[0].clear()
                report_axs[0].set_title(
                    "Density distribution (t={0})".format(last_time)
                )
                report_axs[0].plot(r_vals, rho_s, label=r"$\rho_s$", c="g")
                report_axs[0].plot(r_vals, rho_g, label=r"$\rho_g$", c="r")
                report_axs[0].plot(r_vals, rho, label=r"$\rho$", c="b")
                report_axs[0].set_ylim((0, rho_0 * 1.1))
                report_axs[0].legend()

                report_axs[1].clear()
                report_axs[1].set_title("Chemical distribution")
                report_axs[1].plot(
                    r_vals, rho_chem, label=r"$\rho_{chem}$", c="k")
                # report_axs[1].set_ylim((1e-4, 1.1))
                # report_axs[1].set_xlim((1e-4, ratio_density_max))

                report_fig.tight_layout()
                report_fig.savefig(
                    output_prefix
                    + "density_profile_i{0:05d}_t{1:.3f}.pdf".format(report_iteration+1, last_time),
                    bbox_inches="tight",
                )
            report_iteration += 1

    print("Writing final state")
    write_tissue_state_chemical(output_prefix+"final_state.txt",
                                r_vals, total_rho_history, total_t_history)
    print("Done")

    sys.exit(0)
