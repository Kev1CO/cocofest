"""
This example demonstrates the way of identifying an experimental muscle force model based on Ding 2007 model.
"""

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import pickle
import pandas as pd
import ast

from bioptim import SolutionMerge, OdeSolver, OptimalControlProgram, ObjectiveFcn, Node, ControlType, ObjectiveList, \
    Solver

from cocofest import (
    DingModelPulseWidthFrequency,
    IvpFes,
    ModelMaker,
    OcpFesId, FES_plot,
)

from cocofest.identification.identification_method import DataExtraction
from examples.data_process.c3d_to_force import C3dToForce



def set_time_to_zero(stim_time, time_list):
    first_stim = stim_time[0]
    if first_stim > time_list[0]:
        raise ValueError("Time list should begin at the first stimulation")
    stim_time = list(np.array(stim_time) - first_stim)
    time_list = list(np.array(time_list) - first_stim)

    return stim_time, time_list


def prepare_ocp(
    model,
    final_time,
    pulse_width_values,
    key_parameter_to_identify,
    tracked_data,
    stim_time
):
    n_shooting = model.get_n_shooting(final_time)

    force_at_node = DataExtraction.force_at_node_in_ocp(tracked_data["time"], tracked_data["force"], n_shooting, final_time)

    numerical_data_time_series, stim_idx_at_node_list = model.get_numerical_data_time_series(n_shooting, final_time)
    dynamics = OcpFesId.declare_dynamics(model=model, numerical_data_timeseries=numerical_data_time_series)

    x_bounds, x_init = OcpFesId.set_x_bounds(
        model=model,
        force_tracking=force_at_node,
    )
    u_bounds, u_init = OcpFesId.set_u_bounds(
        model=model,
        control_value=pulse_width_values,
        stim_idx_at_node_list=stim_idx_at_node_list,
        n_shooting=n_shooting,
    )

    objective_functions = ObjectiveList()
    objective_functions.add(
        ObjectiveFcn.Lagrange.TRACK_STATE,
        key="F",
        weight=1,
        target=np.array(force_at_node)[np.newaxis, :],
        node=Node.ALL,
        quadratic=True,
    )
    additional_key_settings = OcpFesId.set_default_values(model)

    parameters, parameters_bounds, parameters_init = OcpFesId.set_parameters(
        parameter_to_identify=key_parameter_to_identify,
        parameter_setting=additional_key_settings,
        use_sx=True,
    )
    OcpFesId.update_model_param(model, parameters)

    return OptimalControlProgram(
        bio_model=[model],
        dynamics=dynamics,
        n_shooting=n_shooting,
        phase_time=final_time,
        x_init=x_init,
        x_bounds=x_bounds,
        u_init=u_init,
        u_bounds=u_bounds,
        objective_functions=objective_functions,
        parameters=parameters,
        parameter_bounds=parameters_bounds,
        parameter_init=parameters_init,
        control_type=ControlType.CONSTANT,
        use_sx=True,
        ode_solver=OdeSolver.RK4(n_integration_steps=10),
        n_threads=20,
    )

def separate_trains(pickle_path, n):
    with open(pickle_path, "rb") as f:
        data = pickle.load(f)
    stim_time = data["stim_time"]
    force_biclong = data["force_biclong"]
    force_bicshort = data["force_bicshort"]
    time = data["time"]

    if n < 0 or n > len(time):
        raise ValueError(f"n must be between 0 and {len(time)}")

    stim_time_check = stim_time[n]
    force_biclong_check = force_biclong[n]
    force_bicshort_check = force_bicshort[n]
    time_check = time[n]

    data_check = {"stim_time": stim_time_check, "time": time_check, "force_biclong": force_biclong_check, "force_bicshort": force_bicshort_check}

    stim_time_train = stim_time[:n] + stim_time[n+1:]
    force_biclong_train = force_biclong[:n] + force_biclong[n+1:]
    force_bicshort_train = force_bicshort[:n] + force_bicshort[n+1:]
    time_train = time[:n] + time[n+1:]

    data_train = {"stim_time": stim_time_train, "time": time_train, "force_biclong": force_biclong_train, "force_bicshort": force_bicshort_train}
    return data_train, data_check

def set_time_continuity(stim_time, time):
    stim_time[0] = np.array(stim_time[0]) - time[0][0]
    time[0] = np.array(time[0]) - time[0][0]
    for i in range(len(stim_time) - 1):
        stim_time[i + 1] = stim_time[i + 1] + (time[i][-1] - time[i + 1][0])
        time[i + 1] = time[i + 1] + (time[i][-1] - time[i + 1][0])

    return stim_time, time

def get_pickle_paths_from_participant(p_n):
    p_data = pd.read_csv("/home/mickaelbegon/Documents/Stage_Florine/Data/data_participants.csv", sep=";")
    freq_str = p_data.iloc[p_n - 1]["freq_force"]
    freq_list = ast.literal_eval(freq_str)
    seeds_str = p_data.iloc[p_n - 1]["seed_force"]
    seeds_list = ast.literal_eval(seeds_str)

    data = {"freq": freq_list, "seeds": seeds_list}

    p_nb = str(p_n) if len(str(p_n)) == 2 else "0" + str(p_n)
    file_path = "/home/mickaelbegon/Documents/Stage_Florine/pkl_files_force/"
    pickle_path_list = [f"{file_path}/p{p_nb}_force_{freq_list[i]}Hz_{seeds_list[i]}.pkl" for i in range(len(freq_list))]

    return pickle_path_list, data

def main(plot=True, p_n_list=None):
    for p_n in p_n_list:
        pickle_path_list, param = get_pickle_paths_from_participant(p_n)
        for i, pickle_path in enumerate(pickle_path_list):
            n_sep = 0
            data_train, data_check = separate_trains(pickle_path, n_sep)
            stim_time, time = set_time_continuity(data_train["stim_time"], data_train["time"])
            time = np.concatenate(time)

            force_biclong = np.concatenate(data_train["force_biclong"])
            force_bicshort = np.concatenate(data_train["force_bicshort"])
            tracked_data_biclong = {"time": time, "force": force_biclong}
            tracked_data_bicshort = {"time": time, "force": force_bicshort}

            final_time = np.round(time[-1], 2)

            # pulse width values
            pulse_width_path = "../data_process/seeds_pulse_width.pkl"
            with open(pulse_width_path, "rb") as f:
                pulse_width_dict = pickle.load(f)
            pulse_width_values = pulse_width_dict[param["seeds"][i]]
            pulse_width_values = list(np.array(pulse_width_values)/1e6)
            pulse_width_values_train = pulse_width_values[:n_sep] + pulse_width_values[n_sep+1:]
            for j in range(len(pulse_width_values_train)):
                pulse_width_values_train[j] = [pulse_width_values_train[j]] * len(stim_time[j])
            pulse_width_values_train = np.concatenate(pulse_width_values_train)
            pulse_width_values_check = pulse_width_values[n_sep]

            stim_time = np.concatenate(stim_time)
            stim_time = list(np.round(stim_time, 2))
            model = ModelMaker.create_model("ding2007", stim_time=stim_time, sum_stim_truncation=10)

            ocp = prepare_ocp(
                model,
                final_time,
                pulse_width_values_train,
                tracked_data=tracked_data_biclong,
                stim_time=stim_time,
                key_parameter_to_identify=[
                    "km_rest",
                    "tau1_rest",
                    "tau2",
                    "pd0",
                    "pdt",
                    "a_scale",
                ],
            )
            sol = ocp.solve(Solver.IPOPT(_max_iter=1000))

            sol_time = sol.stepwise_time(to_merge=SolutionMerge.NODES).T[0]
            sol_force = sol.stepwise_states(to_merge=SolutionMerge.NODES)["F"][0]

            if plot:
                default_model = DingModelPulseWidthFrequency()

                FES_plot(data=sol).plot(
                    title=f"Identification of experimetal force - Participant {p_n} - Freq {param['freq'][i]}Hz - Seed {param['seeds'][i]}",
                    tracked_data=tracked_data_biclong,
                    default_model=default_model,
                    show_bounds=False,
                    show_stim=True,
                    stim_time=stim_time
                )


if __name__ == "__main__":
    main(plot=True, p_n_list=[1])
