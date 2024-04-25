#!/usr/bin/env python3
import numpy as np
from ddesolver import solve_dde


class Simple_DFK_Settings:
    def __init__(self) -> None:
        self.simulation_total_time = None
        self.simulation_time_step = None
        self.simulation_reporting_step = None
        self.simulation_D = None
        self.simulation_b = None
        self.simulation_rho_c = None
        self.simulation_c = None
        self.simulation_tau_delay = None
        self.radius_max = None
        self.radius_delta = None
        self.init_radius_max = None
        self.init_density = None
        self.apoptosis_start_time = -1
        self.apoptosis_start_rate = 1.0
        self.apoptosis_decay_rate = 0.0
        self.input_path = None
        self.output_path = None

    def write(self, config_out) -> None:
        config_out.write("{0}\t=\t{1:.5e}\n".format("D", self.simulation_D))
        config_out.write("{0}\t=\t{1:.5e}\n".format("b", self.simulation_b))
        config_out.write("{0}\t=\t{1:.5e}\n".format(
            "Curvature (c)", self.simulation_c))
        config_out.write(
            "{0}\t=\t{1:.5e}\n".format(
                "Homeostatic density (rho_c)", self.simulation_rho_c)
        )

        config_out.write("{0}\t=\t{1:.5e}\n".format(
            "Delta_t_s", self.simulation_tau_delay))

        config_out.write("{0}\t=\t{1:.5e}\n".format(
            "Simulation time step (dt)", self.simulation_time_step))
        config_out.write(
            "{0}\t=\t{1:.5e}\n".format(
                "Report time delta (rs)", self.simulation_reporting_step)
        )
        config_out.write(
            "{0}\t=\t{1:.5e}\n".format(
                "Simulation time total (t)", self.simulation_total_time)
        )

        config_out.write("{0}\t=\t{1:.5e}\n".format(
            "Initialized Radius (R_i)", self.init_radius_max))
        config_out.write("{0}\t=\t{1:.5e}\n".format(
            "Initialized density (d_i)", self.init_density))

        # config_out.write(
        #    "{0}\t=\t{1:.5e}\n".format(
        #        "Simulation time start (t_0)", last_time)
        # )
        # config_out.write(
        #    "{0}\t=\t{1:.5e}\n".format(
        #        "Simulation time end (t_e)", final_sim_time)
        # )

        config_out.write("{0}\t=\t{1:.5e}\n".format(
            "Simulated Radius limit (R)", self.radius_max))
        config_out.write(
            "{0}\t=\t{1:.5e}\n".format(
                "Radial discretisation step (dr)", self.radius_delta)
        )

        # config_out.write(
        #    "{0}\t=\t{1:.5e}\n".format(
        #        "Number of radial positions (n)", num_discretization_steps + 1
        #    )
        # )

        if self.output_path is not None:
            config_out.write("{0}\t=\t{1}\n".format(
                "Output prefix", self.output_path))

        if self.input_path is not None:
            config_out.write("{0}\t=\t{1}\n".format(
                "Input configuration", self.input_path))

        config_out.write(
            "{0}\t=\t{1:.5e}\n".format(
                "apoptosis_start_time", self.apoptosis_start_time
            )
        )
        config_out.write(
            "{0}\t=\t{1:.5e}\n".format(
                "apoptosis_start_rate", self.apoptosis_start_rate
            )
        )
        config_out.write(
            "{0}\t=\t{1:.5e}\n".format(
                "apoptosis_decay_rate", self.apoptosis_decay_rate
            )
        )


class Simple_DFK_Sim:
    def __init__(self) -> None:
        self.r_vals = None

    def get_r_vals(self, config: Simple_DFK_Settings):
        # Generate the discrete r positions
        total_discrete_steps = np.ceil(config.radius_max / config.radius_delta)
        l_r_vals = np.linspace(
            start=0,
            stop=config.radius_max,
            num=total_discrete_steps + 1,
        )
        return l_r_vals

    def get_nabla_reflecting(self, r, f):
        # calculate the gradient in r with reflecting boundary conditions at r=0 and
        # the dirichlet condition at the outer edge (hence we toss that gradient value)
        res = f * 0.0
        dr = r[1] - r[0]

        # get stencil for discret nabla
        r_m1 = f[0:-2]
        r_p1 = f[2:]

        res[1:-1] = (r_p1 - r_m1) / (2 * dr)

        res[0] = 0  # we assume reflecting boundary conditions at zero
        # res[-1] = (-f[-2]) / (2 * dr)
        res[-1] = 0  # we force this to zero, we do not need the gradient

        return res

    def get_laplace(self, r, f):
        # Calculate the second derivative incorporating the vanishing gradient at zero and
        # the forced zero at the outer edge
        res = f * 0.0
        dr = r[1] - r[0]
        dr2 = dr * dr

        # Correct the second order deriative at r=0 that needs to be calculated differently
        res[0] = (f[1] - f[0]) / (dr2)

        # get stencil elements for discrete laplace
        r_m1 = f[0:-2]
        r_c = f[1:-1]
        r_p1 = f[2:]

        res[1:-1] = (r_p1 - 2 * r_c + r_m1) / (dr2)

        res[-1] = 0

        return res

    def get_dfk_dde_model(self, config: Simple_DFK_Settings):
        def dfk_dde_model(Y, t):
            per_system_dim = len(self.r_vals)
            # Get current state
            current = Y(t)
            rho_g = current[:per_system_dim]
            rho_s = current[per_system_dim:]
            # Get delayed state
            delayed = Y(t - config.simulation_tau_delay)
            rho_g_d = delayed[:per_system_dim]
            rho_s_d = delayed[per_system_dim:]

            # Get gradient and laplace for differential calculation
            gradient_g = self.get_nabla_reflecting(self.r_vals, rho_g)
            laplace_g = self.get_laplace(self.r_vals, rho_g)

            def enter_s(p_b, p_rho_0, p_c, p_rho_g, p_rho_s):
                rho_total = p_rho_g + p_rho_s
                return (
                    p_b
                    * np.power(np.clip(1.0 - rho_total / p_rho_0, a_min=0.0, a_max=1.0), p_c)
                    * p_rho_g
                )

            enter_division = enter_s(
                config.simulation_b, config.simulation_rho_c, config.simulation_c, rho_g, rho_s)

            # Before time t=0 there was no proliferation onset according to our model
            if t - config.simulation_tau_delay >= 0:
                enter_division_d = enter_s(
                    config.simulation_b, config.simulation_rho_c, config.simulation_c, rho_g_d, rho_s_d)
            else:
                enter_division_d = rho_s * 0.0

            diff_t_g = np.zeros_like(rho_g)
            diff_t_s = np.zeros_like(rho_s)

            # calculate d_t for all but the r=0 position
            diff_t_g[1:] = (
                (config.simulation_D) *
                (laplace_g[1:] + gradient_g[1:] / self.r_vals[1:])
                # The g-state cells entering the s-phase
                - enter_division[1:]
                # The previous s-phase cells now proliferating
                + 2 * enter_division_d[1:]
            )

            # Deal with the pole at r=0 which cancels with the reflecting boundary condition gradient=0
            # FIXME: maybe there is a term that needs to be plugged instead of 0.0 (gradient/r \to ? for r\to 0)
            diff_t_g[0] = (
                (config.simulation_DD) * (laplace_g[0] + 0.0)
                # The g-state cells entering the s-phase
                - enter_division[0]
                # The previous s-phase cells now proliferating
                + 2 * enter_division_d[0]
            )

            # Apply apoptosis if configured
            if config.apoptosis_start_time >= 0 and t > config.apoptosis_start_time:
                # Apoptosis rate decays over time
                curr_apoptosis_rate = config.apoptosis_start_rate * np.exp(
                    -(t - config.apoptosis_start_time) *
                    config.apoptosis_decay_rate
                )
                # Add apoptosis on top of other influences
                diff_t_g -= rho_g * curr_apoptosis_rate

            diff_t_s = enter_division - enter_division_d

            # No change at last position
            diff_t_g[-1] = 0.0
            diff_t_s[-1] = 0.0

            return np.concatenate(
                (
                    diff_t_g,
                    diff_t_s,
                )
            )
        return dfk_dde_model

    # Generate initial history conditions from a list of times and associated density data

    def get_history_init_condition(self, t_values, rho_data):
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

    def rect_init_condition(self, config: Simple_DFK_Settings):
        # Routine for initialization of box-shaped initial conditions:
        r_vals = self.get_r_vals(config)
        rho_g = np.zeros_like(r_vals)
        rho_s = np.zeros_like(r_vals)

        # initialize the density distribution
        for i in range(len(r_vals)):
            if r_vals[i] <= config.init_radius_max:
                rho_g[i] = config.init_density

        return r_vals, np.concatenate((rho_g, rho_s))

    def linear_init_condition(self, config: Simple_DFK_Settings):
        # Routine for initialization of linearly decreasing initial conditions:
        r_vals = self.get_r_vals(config)
        rho_g = np.zeros_like(r_vals)
        rho_s = np.zeros_like(r_vals)

        # initialize the density distribution
        for i in range(len(r_vals)):
            if r_vals[i] <= config.init_radius_max:
                rho_g[i] = config.init_density * \
                    (config.init_radius_max - r_vals[i])/config.init_radius_max

        return r_vals, np.concatenate((rho_g, rho_s))

    def simulate_uninitialized_system(self, config: Simple_DFK_Settings):
        pass

    def simulate_simplified_linear_system(self, config: Simple_DFK_Settings):
        self.r_vals, state = self.linear_init_condition(config)

        total_t_history = None
        total_rho_history = None
        last_time = 0.0
        final_sim_time = last_time+config.simulation_total_time
        per_system_dim = len(self.r_vals)

        print("Simulating a total of t=", config.simulation_total_time)
        print("Starting at t_0=", last_time)
        print("With a report interval /\\t_report=",
              config.simulation_reporting_step)
        print("And a time step /\\t=", config.simulation_time_step)

        while last_time + 1.0001*config.simulation_time_step < final_sim_time:
            # Calculate the next report step end time and required simulation time
            next_end_time = min(final_sim_time, last_time +
                                config.simulation_reporting_step)
            sim_delta = next_end_time - last_time

            # print("simulate from ", last_time, " to ", next_end_time)

            # Sample the time interval to be simulated
            t_data = np.linspace(
                last_time, next_end_time, num=max(
                    int(np.ceil(sim_delta / config.simulation_time_step)), 2)
            )

            # Do the simulation
            res_data = solve_dde(
                self.get_dfk_dde_model(config),
                init_condition,
                t_data,
                fargs=(
                    # Delta_t_s,
                    # D,
                    # b,
                    # rho_0,
                    # c,
                    # r_vals,
                    # param_apoptosis_start_time,
                    # param_apoptosis_start_rate,
                    # param_apoptosis_decay_rate,
                ),
            )

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
                # for step in range(len(res_data)):
                #    #rho_g = res_data[step][:per_system_dim]
                #    #rho_s = res_data[step][per_system_dim:]

                #    #write_entry(trajectory_out, t_data[step], rho_g, rho_s)
                #    #rho = rho_g + rho_s

            # Filter the history and truncate entries to free up memory
            history_time_filter = total_t_history >= last_time - 2 * config.simulation_tau_delay

            # copy the filtered arrays to drop references to original large array
            total_t_history = total_t_history[history_time_filter].copy()
            total_rho_history = total_rho_history[history_time_filter].copy()

            # print(len(total_t_history), "entries in history")

            # Rebuild the new initial condition
            init_condition = self.get_history_init_condition(
                total_t_history, total_rho_history
            )

            c_data = res_data[-1]

            rho_g = c_data[:per_system_dim]
            rho_s = c_data[per_system_dim:]

            rho = rho_g + rho_s

        return self.r_vals, total_t_history, total_rho_history
