import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import pickle
import pandas as pd
import ast

from bioptim import SolutionMerge, OdeSolver, OptimalControlProgram, ObjectiveFcn, Node, ControlType, ObjectiveList, \
    Solver

from cocofest import IvpFes, ModelMaker, OcpFesId, FES_plot

from cocofest.identification.identification_method import DataExtraction


def set_time_to_zero(stim_time, time_list):
    first_stim = stim_time[0]
    #if first_stim > time_list[0]:
        #raise ValueError("Time list should begin at the first stimulation")
    stim_time = list(np.array(stim_time) - first_stim)
    time_list = list(np.array(time_list) - first_stim)

    return stim_time, time_list


def prepare_ocp(
    model,
    final_time,
    pulse_width_values,
    key_parameter_to_identify,
    tracked_data,
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
    """
    This function separates the train of data into a train to identify the parameters and a train to test the identified parameters.
    Parameters
    ----------
    pickle_path
        The path to the pickle file containing the processed experiental data
    n: int
        The index of the train to separate (0 to len(time)-1)

    Returns
    -------
    Two dictionaries: the first one contains the time, stim_time and forces for the training trains, the second one
    contains the time, stim_time and forces for the test train

    """
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
    """
    This function retrieves the pickle paths and data for a specific participant based on their number.
    Parameters
    ----------
    p_n: int
        Participant number

    Returns
    -------
    The list of pickle path to the participant's c3d files, and a dictionary containing the frequency and seeds used
    for the participant.
    """
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

def optim_all_concat(p_n, muscle_name, plot=True, save=True):
    """
        This function identifies the parameters of the Ding 2007 model based on experimental data for a specific participant and muscle.
        It identifies for all frequencies, meaning it will create a model for all frequencies together.
        Parameters
        ----------
        p_n: int
            Participant number
        muscle_name: str
            Muscle you want to use to identify parameters
        plot: bool
            If True, plot the results
        save: bool
            If True, save the results in a pickle file
        """
    pickle_path_list, param = get_pickle_paths_from_participant(p_n)
    force_train_list = []
    time_train_list = []
    stim_time_train_list = []
    pulse_width_train_list = []
    force_test_list = []
    time_test_list = []
    stim_time_test_list = []
    pulse_width_test_list = []

    for i, pickle_path in enumerate(pickle_path_list):
        n_sep = 0
        data_train, data_test = separate_trains(pickle_path, n_sep)
        stim_time, time = set_time_continuity(data_train["stim_time"], data_train["time"])

        # pulse width values
        pulse_width_path = "../data_process/seeds_pulse_width.pkl"
        with open(pulse_width_path, "rb") as f:
            pulse_width_dict = pickle.load(f)
        pulse_width_values = pulse_width_dict[param["seeds"][i]]
        pulse_width_values = list(np.array(pulse_width_values) / 1e6) #convert into s
        pulse_width_values_train = pulse_width_values[:n_sep] + pulse_width_values[n_sep + 1:]
        for j in range(len(pulse_width_values_train)):
            pulse_width_values_train[j] = [pulse_width_values_train[j]] * len(stim_time[j])
        pulse_width_values_train = np.concatenate(pulse_width_values_train)
        pulse_width_values_test = [pulse_width_values[n_sep]] * len(data_test["stim_time"])

        time = np.concatenate(time)
        force = np.concatenate(data_train[f"force_{muscle_name}"])
        stim_time = np.concatenate(stim_time)

        force_train_list.append(force)
        time_train_list.append(time)
        stim_time_train_list.append(stim_time)
        pulse_width_train_list.append(pulse_width_values_train)
        force_test_list.append(data_test[f"force_{muscle_name}"])
        time_test_list.append(data_test["time"])
        stim_time_test_list.append(data_test["stim_time"])
        pulse_width_test_list.append(pulse_width_values_test)

    stim_time, time = set_time_continuity(stim_time_train_list, time_train_list)

    time = np.concatenate(time)
    stim_time = np.concatenate(stim_time)
    pulse_width = np.concatenate(pulse_width_train_list)
    force = np.concatenate(force_train_list)

    data_train = {"force":force, "stim_time":stim_time, "time":time, "pulse_width":pulse_width}

    stim_time_test, time_test = set_time_continuity(stim_time_test_list, time_test_list)
    data_test = {"force":np.concatenate(force_test_list), "stim_time": np.concatenate(stim_time_test), "time": np.concatenate(time_test), "pulse_width":np.concatenate(pulse_width_test_list)}

    tracked_data = {"time": time, "force": force}

    stim_time = list(np.round(stim_time, 2))
    model = ModelMaker.create_model("ding2007", stim_time=stim_time, sum_stim_truncation=10)

    final_time = np.round(time[-1], 2)

    ocp = prepare_ocp(
        model,
        final_time,
        pulse_width,
        tracked_data=tracked_data,
        key_parameter_to_identify=[
            "km_rest",
            "tau1_rest",
            "tau2",
            "pd0",
            "pdt",
            "a_scale",
        ],
    )
    sol = ocp.solve(Solver.IPOPT(_max_iter=10000))

    sol_time = sol.stepwise_time(to_merge=SolutionMerge.NODES).T[0]
    sol_force = sol.stepwise_states(to_merge=SolutionMerge.NODES)["F"][0]
    for key, value in sol.parameters.items():
        param[key] = value
    solution = {"time": sol_time, "force": sol_force, "parameters": param, "data_test": data_test, "data_train":data_train, "muscle_name":muscle_name}

    if save:
        p_nb = str(p_n) if len(str(p_n)) == 2 else "0" + str(p_n)
        current_file_dir = Path(__file__).parent
        saving_pkl_path = f"{current_file_dir}/id_force/p{p_nb}_force_{muscle_name}.pkl"
        with open(saving_pkl_path, "wb") as f:
            pickle.dump(solution, f)

    if plot:

        FES_plot(data=sol).plot(
            title=f"Identification of experimental force - Participant {p_n} - Muscle {muscle_name}",
            tracked_data=tracked_data,
            show_bounds=False,
            show_stim=True,
            stim_time=stim_time,
        )

def optim_per_freq(p_n, muscle_name, plot=True, save=True):
    """
    This function identifies the parameters of the Ding 2007 model based on experimental data for a specific participant and muscle.
    It identifies per frequency, meaning it will create a model for each frequency separately.
    Parameters
    ----------
    p_n: int
        Participant number
    muscle_name: str
        Muscle you want to use to identify parameters
    plot: bool
        If True, plot the results
    save: bool
        If True, save the results in a pickle file
    """
    pickle_path_list, param = get_pickle_paths_from_participant(p_n)

    for i, pickle_path in enumerate(pickle_path_list):
        n_sep = 0
        data_train, data_test = separate_trains(pickle_path, n_sep)
        stim_time, time = set_time_continuity(data_train["stim_time"], data_train["time"])

        # pulse width values
        pulse_width_path = "../data_process/seeds_pulse_width.pkl"
        with open(pulse_width_path, "rb") as f:
            pulse_width_dict = pickle.load(f)
        pulse_width_values = pulse_width_dict[param["seeds"][i]]
        pulse_width_values = list(np.array(pulse_width_values) / 1e6) #convert into s
        pulse_width_values_train = pulse_width_values[:n_sep] + pulse_width_values[n_sep + 1:]
        for j in range(len(pulse_width_values_train)):
            pulse_width_values_train[j] = [pulse_width_values_train[j]] * len(stim_time[j])
        pulse_width_values_train = np.concatenate(pulse_width_values_train)
        pulse_width_values_test = [pulse_width_values[n_sep]] * len(data_test["stim_time"])

        time = np.concatenate(time)
        force = np.concatenate(data_train[f"force_{muscle_name}"])
        stim_time = np.concatenate(stim_time)


        data_train = {"force":force, "stim_time":stim_time, "time":time, "pulse_width":pulse_width_values_train}

        stim_time_test, time_test = set_time_to_zero(data_test["stim_time"], data_test["time"])
        data_test = {"force":data_test[f"force_{muscle_name}"], "stim_time": stim_time_test, "time": time_test, "pulse_width":pulse_width_values_test}

        tracked_data = {"time": time, "force": force}

        stim_time = list(np.round(stim_time, 2))
        model = ModelMaker.create_model("ding2007", stim_time=stim_time, sum_stim_truncation=10)

        final_time = np.round(time[-1], 2)

        ocp = prepare_ocp(
            model,
            final_time,
            pulse_width_values_train,
            tracked_data=tracked_data,
            key_parameter_to_identify=[
                "km_rest",
                "tau1_rest",
                "tau2",
                "pd0",
                "pdt",
                "a_scale",
            ],
        )
        sol = ocp.solve(Solver.IPOPT(_max_iter=10000))

        sol_time = sol.stepwise_time(to_merge=SolutionMerge.NODES).T[0]
        sol_force = sol.stepwise_states(to_merge=SolutionMerge.NODES)["F"][0]
        parameters = {}
        for key, value in sol.parameters.items():
            parameters[key] = value
        parameters["freq"] = param["freq"][i]
        freq = parameters["freq"]
        parameters["seeds"] = param["seeds"][i]
        seed = parameters["seeds"]
        pulse_width_plot = pulse_width_dict[seed][:n_sep] + pulse_width_dict[seed][n_sep + 1:]
        pulse_width_plot = [int(x) for x in pulse_width_plot]
        parameters["pulse_width"] = pulse_width_plot
        solution = {"time": sol_time, "force": sol_force, "parameters": parameters, "data_test": data_test, "data_train":data_train, "muscle_name":muscle_name}

        if save:
            p_nb = str(p_n) if len(str(p_n)) == 2 else "0" + str(p_n)
            current_file_dir = Path(__file__).parent
            saving_pkl_path = f"{current_file_dir}/id_force/p{p_nb}_force_{muscle_name}_{freq}Hz_{seed}.pkl"
            with open(saving_pkl_path, "wb") as f:
                pickle.dump(solution, f)

        if plot:
            FES_plot(data=sol).plot(
                title=f"Identification of experimental force - Participant {p_n} - Muscle {muscle_name} - Freq {freq} - Pulse Width values {pulse_width_plot}",
                tracked_data=tracked_data,
                show_bounds=False,
                show_stim=True,
                stim_time=stim_time,
            )

def id_auto(p_n_list=None, muscle_name_list=None, plot=True, save=False, per_freq=False):
    """
    This function automatically identifies the parameters of the Ding 2007 model based on experimental data.
    Parameters
    ----------
    p_n_list: list
        Participant number list
    muscle_name_list: list
        Muscle you want to use to identify parameters
    plot: bool
        If True plot the results
    save: bool
        If True save the results in a pickle file
    per_freq: bool
        If True, identify parameters for each frequency separately, otherwise identify parameters for all frequencies concatenated
    """
    for p_n in p_n_list:
        for muscle_name in muscle_name_list:
            if per_freq:
                optim_per_freq(p_n, muscle_name=muscle_name, plot=plot, save=save)
            else:
                optim_all_concat(p_n, muscle_name=muscle_name, plot=plot, save=save)

def check_data_id(p_n_list, muscle_name_list, per_freq=False):
    """
    This function plots the identified parameters from the pickle files created by the identification process.
    Parameters
    ----------
    p_n_list: list
        Participant numbers list
    muscle_name_list: list
        Muscle names list
    per_freq: bool
        If True, identify parameters for each frequency separately, otherwise identify parameters for all frequencies concatenated
    """
    for p_n in p_n_list:
        for muscle_name in muscle_name_list:
            p_nb = str(p_n) if len(str(p_n)) == 2 else "0" + str(p_n)
            current_file_dir = Path(__file__).parent
            if per_freq :
                p_data = pd.read_csv("/home/mickaelbegon/Documents/Stage_Florine/Data/data_participants.csv", sep=";")
                freq_str = p_data.iloc[p_n - 1]["freq_force"]
                freq_list = ast.literal_eval(freq_str)
                seeds_str = p_data.iloc[p_n - 1]["seed_force"]
                seeds_list = ast.literal_eval(seeds_str)

                for i in range(len(freq_list)):
                    pickle_path = f"{current_file_dir}/id_force/p{p_nb}_force_{muscle_name}_{freq_list[i]}Hz_{seeds_list[i]}.pkl"
                    with open(pickle_path, "rb") as f:
                        data = pickle.load(f)
                    param_dict = data["parameters"]
                    pulse_width = param_dict["pulse_width"]
                    print("Identified parameters :")
                    for key in param_dict.keys():
                        print(f"{key} : {param_dict[key]}")
                    plt.plot(data["time"], data["force"], label="identified", color="red")
                    plt.plot(data["data_train"]["time"], data["data_train"]["force"], label="tracked", color="blue")
                    data_stim = np.interp(data["data_train"]["stim_time"], data["data_train"]["time"],data["data_train"]["force"])
                    plt.scatter(data["data_train"]["stim_time"], data_stim, label="stimulations", color="green", alpha=0.5)
                    plt.title(f"Identification from experimental force - Participant {p_nb} - Muscle {muscle_name} - Frequency {freq_list[i]} - Pulse Width values {pulse_width}")
                    plt.xlabel("Time (s)")
                    plt.ylabel("Force (N)")
                    plt.legend()
                    plt.show()
            else:
                pickle_path = f"{current_file_dir}/id_force/p{p_nb}_force_{muscle_name}.pkl"
                with open(pickle_path, "rb") as f:
                    data = pickle.load(f)
                param_dict = data["parameters"]
                print("Identified parameters :")
                for key in param_dict.keys():
                    print(f"{key} : {param_dict[key]}")
                plt.plot(data["time"], data["force"], label="identified", color="red")
                plt.plot(data["data_train"]["time"], data["data_train"]["force"], label="tracked", color="blue")
                data_stim = np.interp(data["data_train"]["stim_time"], data["data_train"]["time"], data["data_train"]["force"])
                plt.scatter(data["data_train"]["stim_time"], data_stim, label="stimulations", color="green", alpha=0.5)
                plt.title(f"Identification from experimental force - Participant {p_nb} - Muscle {muscle_name}")
                plt.xlabel("Time (s)")
                plt.ylabel("Force (N)")
                plt.legend()
                plt.show()

def get_force_from_id_param(param_dict:dict, data_test:dict):
    """
    This function generates the force from the identified parameters (saved in a pickle file).
    Parameters
    ----------
    param_dict: dict
        Dictionary containing the identified parameters
    data_test: dict
        Dictionary containing the train of experimental data separated to test the identified parameters

    Returns
    -------
     A dictionary that contains the generated force, time and stim_time.
    """
    stim_time = data_test["stim_time"]
    stim_time = list(np.round(stim_time, 2))
    model = ModelMaker.create_model("ding2007", stim_time=stim_time, sum_stim_truncation=10)
    model.a_scale = param_dict["a_scale"]
    model.km_rest = param_dict["km_rest"]
    model.tau1_rest = param_dict["tau1_rest"]
    model.tau2 = param_dict["tau2"]
    model.pdt = param_dict["pdt"]
    model.pd0 = min(param_dict["pd0"], 250 * 1e-6) #to make sure pulse width 250µs is possible

    fes_parameters = {"model": model, "pulse_width": list(data_test["pulse_width"])}

    final_time = np.round(data_test["time"][-1], 2)
    ivp_parameters = {
        "final_time": final_time,
        "use_sx": True,
        "ode_solver": OdeSolver.RK4(n_integration_steps=10),
    }
    ivp = IvpFes(fes_parameters, ivp_parameters)

    result, time = ivp.integrate()
    data = {
        "time": time,
        "force": result["F"][0],
        "stim_time": stim_time,
    }
    return data

def loo_auto(p_n_list, muscle_name_list, plot=True, save=False, per_freq=False):
    """
    This function automatically runs the leave-one-out (LOO) method (compute_out() function) to compute the RMSE of the Out for each participant
    and muscle in the lists. It can plot or save the results.
    Parameters
    ----------
    p_n_list: list
        Participant numbers list
    muscle_name_list: list
        Muscle names list
    plot: bool
        If True plot the results
    save: bool
        If True save the results in a pickle file
    per_freq: bool
        If True, compute the generated force for each frequency separately, otherwise compute the generated force for all frequencies concatenated
    """
    for p_n in p_n_list:
        p_nb = str(p_n) if len(str(p_n)) == 2 else "0" + str(p_n)
        for muscle_name in muscle_name_list:
            if per_freq:
                p_data = pd.read_csv("/home/mickaelbegon/Documents/Stage_Florine/Data/data_participants.csv", sep=";")
                freq_str = p_data.iloc[p_n - 1]["freq_force"]
                freq_list = ast.literal_eval(freq_str)
                seeds_str = p_data.iloc[p_n - 1]["seed_force"]
                seeds_list = ast.literal_eval(seeds_str)

                for i in range(len(freq_list)):
                    path = f"p{p_nb}_force_{muscle_name}_{freq_list[i]}Hz_{seeds_list[i]}"
                    compute_out(p_nb=p_nb, muscle_name=muscle_name, path=path, save=save, plot=plot, per_freq=per_freq)
            else:
                path = f"p{p_nb}_force_{muscle_name}"
                compute_out(p_nb=p_nb, muscle_name=muscle_name, path=path, save=save, plot=plot, per_freq=per_freq)

def compute_out(p_nb, muscle_name, path, plot, save, per_freq):
    """
    This function computes the RMSE of the Out for each participant and muscle.
    It can plot or save the results.
    Parameters
    ----------
    p_nb: str
        Participant number as a string
    muscle_name: str
        Muscle name
    path: str
        Path to the pickle file containing the identified parameters
    plot: bool
        If True plot the results
    save: bool
        If True save the results in a pickle file
    per_freq: bool
        If True, compute the generated force for each frequency separately, otherwise compute the generated force for all frequencies concatenated
    """
    current_file_dir = Path(__file__).parent
    pickle_path = f"{current_file_dir}/id_force/{path}.pkl"
    with open(pickle_path, "rb") as f:
        data = pickle.load(f)
    param_dict = data["parameters"]
    data_test = data["data_test"]
    generated_data = get_force_from_id_param(param_dict, data_test)

    exp_data = np.interp(generated_data["time"], data_test["time"], data_test["force"])
    rmse = np.sqrt(np.mean((exp_data - generated_data["force"]) ** 2))
    mean_exp_data = np.mean(exp_data)
    rmse_percent = (rmse / mean_exp_data) * 100
    rmse = np.round(rmse, 2)
    rmse_percent = np.round(rmse_percent, 2)

    print(f"Root Mean Square Error : {rmse}")
    print(f"Root Mean Square Error in % : {rmse_percent}%")

    if plot:
        plt.plot(data_test["time"], data_test["force"], label="experimental", color="blue")
        plt.plot(generated_data["time"], generated_data["force"], label="simulated", color="red")
        data_stim = np.interp(data_test["stim_time"], data_test["time"], data_test["force"])
        plt.scatter(data_test["stim_time"], data_stim, label="stimulations", color="green", alpha=0.5)
        if per_freq:
            freq = param_dict["freq"]
            pulse_width = int(data_test["pulse_width"][0] * 1e6)
            plt.title(f"Simulated force from identified parameters - Participant {p_nb} - Muscle {muscle_name} - Freq {freq} - Pulse Width values {pulse_width} µs")
        else:
            plt.title(f"Simulated force from identified parameters - Participant {p_nb} - Muscle {muscle_name}")
        plt.xlabel("Time (s)")
        plt.ylabel("Force (N)")
        plt.legend()
        plt.show()

    if save:
        current_file_dir = Path(__file__).parent
        saving_pkl_path = f"{current_file_dir}/force_test/{path}.pkl"
        dict = {"generated_data": generated_data, "data_test": data_test, "rmse": rmse, "rmse%": rmse_percent}
        with open(saving_pkl_path, "wb") as f:
            pickle.dump(dict, f)

def check_data_loo(p_n_list, muscle_name_list, per_freq=False):
    """
    This function plots the results of the leave-one-out (LOO) method.
    Parameters
    ----------
    p_n_list: list
        Participant numbers list
    muscle_name_list: list
        Muscle names list
    per_freq: bool
        If True, plot the results for each frequency separately, otherwise plot the results for all frequencies
    """
    for p_n in p_n_list:
        for muscle_name in muscle_name_list:
            p_nb = str(p_n) if len(str(p_n)) == 2 else "0" + str(p_n)
            current_file_dir = Path(__file__).parent
            if per_freq:
                p_data = pd.read_csv("/home/mickaelbegon/Documents/Stage_Florine/Data/data_participants.csv", sep=";")
                freq_str = p_data.iloc[p_n - 1]["freq_force"]
                freq_list = ast.literal_eval(freq_str)
                seeds_str = p_data.iloc[p_n - 1]["seed_force"]
                seeds_list = ast.literal_eval(seeds_str)

                for i in range(len(freq_list)):
                    pickle_path = f"{current_file_dir}/force_test/p{p_nb}_force_{muscle_name}_{freq_list[i]}Hz_{seeds_list[i]}.pkl"

                    with open(pickle_path, "rb") as f:
                        data = pickle.load(f)
                    rmse = data["rmse"]
                    rmse_percent = data["rmse%"]
                    print(f"Root Mean Square Error : {rmse}")
                    print(f"Root Mean Square Error in % : {rmse_percent}%")
                    plt.plot(data["data_test"]["time"], data["data_test"]["force"], label="experimental", color="blue")
                    plt.plot(data["generated_data"]["time"], data["generated_data"]["force"], label="simulated",
                             color="red")
                    data_stim = np.interp(data["data_test"]["stim_time"], data["data_test"]["time"],
                                          data["data_test"]["force"])
                    plt.scatter(data["data_test"]["stim_time"], data_stim, label="stimulations", color="green",
                                alpha=0.5)
                    plt.title(f"Simulated force from identified parameters - Participant {p_nb} - Muscle {muscle_name} - Freq {freq_list[i]}")
                    plt.xlabel("Time (s)")
                    plt.ylabel("Force (N)")
                    plt.legend()
                    plt.show()
            else:
                pickle_path = f"{current_file_dir}/force_test/p{p_nb}_force_{muscle_name}.pkl"

            with open(pickle_path, "rb") as f:
                data = pickle.load(f)
            rmse = data["rmse"]
            rmse_percent = data["rmse%"]
            print(f"Root Mean Square Error : {rmse}")
            print(f"Root Mean Square Error in % : {rmse_percent}%")
            plt.plot(data["data_test"]["time"], data["data_test"]["force"], label="experimental", color="blue")
            plt.plot(data["generated_data"]["time"], data["generated_data"]["force"], label="simulated", color="red")
            data_stim = np.interp(data["data_test"]["stim_time"], data["data_test"]["time"], data["data_test"]["force"])
            plt.scatter(data["data_test"]["stim_time"], data_stim, label="stimulations", color="green", alpha=0.5)
            plt.title(f"Simulated force from identified parameters - Participant {p_nb} - Muscle {muscle_name}")
            plt.xlabel("Time (s)")
            plt.ylabel("Force (N)")
            plt.legend()
            plt.show()



if __name__ == "__main__":
    id_auto(p_n_list=[8], muscle_name_list=["biclong", "bicshort"], plot=False, save=True, per_freq=True)
    #check_data_id(p_n_list=[3], muscle_name_list=["biclong", "bicshort"], per_freq=False)
    #loo_auto(p_n_list=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20], muscle_name_list=["biclong", "bicshort"], plot=False, save=True, per_freq=False)
    check_data_loo(p_n_list=[3], muscle_name_list=["biclong", "bicshort"], per_freq=False)
